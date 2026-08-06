"""Middleware de logging HTTP que persiste solicitudes en la colección ``logs``."""
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import LogEntry

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()

MAX_BODY_LENGTH = 1000


async def _capture_body(request: Request) -> Optional[str]:
    """Captura el body de la solicitud limitado a 1000 caracteres.

    Args:
        request: Solicitud FastAPI/Starlette.

    Returns:
        Body como texto (truncado) o ``None`` si no se pudo leer.
    """
    try:
        if request.method in ("GET", "HEAD"):
            return None
        body = await request.body()
        text = body.decode("utf-8", errors="replace")
        if len(text) > MAX_BODY_LENGTH:
            text = text[:MAX_BODY_LENGTH] + "...[truncado]"
        return text
    except Exception as e:
        logger.debug("No se pudo capturar body: %s", e)
        return None


async def _save_log_entry(entry: LogEntry) -> None:
    """Persiste una entrada de log en MongoDB (best-effort).

    Args:
        entry: Entrada de log a guardar.
    """
    try:
        db = await _connection.connect()
        await db[settings.MONGODB_COLL_LOGS].insert_one(entry.model_dump())
    except Exception as e:
        logger.debug("Log HTTP no guardado en MongoDB (no bloquea): %s", e)


def _extract_session_id(request: Request, body: Optional[str]) -> str:
    """Intenta extraer session_id del header, query o body.

    Args:
        request: Solicitud.
        body: Body capturado (opcional).

    Returns:
        session_id encontrado o "unknown".
    """
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    try:
        query_session = request.query_params.get("session_id")
        if query_session:
            return query_session
    except Exception:
        pass
    if body:
        try:
            data = json.loads(body)
            if isinstance(data, dict) and data.get("session_id"):
                return str(data["session_id"])
        except Exception:
            pass
    return "unknown"


class LoggingMiddleware(BaseHTTPMiddleware):
    """Registra cada solicitud HTTP en la colección ``logs``.

    Captura timestamp, method, path, client_ip, user_agent, status_code,
    response_time_ms y body (máx 1000 caracteres). No bloquea la respuesta
    ni la persistencia del log.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        body = await _capture_body(request)

        response = await call_next(request)

        try:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            client_ip = request.client.host if request.client else None

            entry = LogEntry(
                method=request.method,
                path=request.url.path,
                client_ip=client_ip,
                user_agent=request.headers.get("user-agent"),
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                body=body,
                session_id=_extract_session_id(request, body),
            )
            await _save_log_entry(entry)
        except Exception as e:
            logger.debug("Error capturando log HTTP: %s", e)

        return response
