"""Servicios para la colección de métricas."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from motor.motor_asyncio import AsyncIOMotorCollection

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import ConversationDocument, MetricCreate
from mongodb.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()


async def _get_collection() -> AsyncIOMotorCollection:
    """Retorna la colección de métricas, conectando si es necesario."""
    db = await _connection.connect()
    return db[settings.MONGODB_COLL_METRICS]


async def record_metric(metric: MetricCreate) -> str:
    """Registra una métrica de rendimiento.

    Args:
        metric: Datos de la métrica a registrar.

    Returns:
        El ID (string) del documento insertado.
    """
    collection = await _get_collection()
    result = await collection.insert_one(metric.model_dump())
    logger.info("📊 Métrica registrada: %s", result.inserted_id)
    return str(result.inserted_id)


async def get_endpoint_metrics(endpoint: str, hours: int = 24) -> Dict[str, Any]:
    """Calcula métricas agregadas para un endpoint en un periodo de horas.

    Args:
        endpoint: Nombre del endpoint (ej: ``/chat``).
        hours: Ventana de tiempo hacia atrás en horas (default: 24).

    Returns:
        Diccionario con total de solicitudes, latencia promedio, tokens,
        cache hits, respuestas RAG y confianza promedio.
    """
    collection = await _get_collection()
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    pipeline = [
        {"$match": {"endpoint": endpoint, "request_timestamp": {"$gte": start}}},
        {
            "$group": {
                "_id": None,
                "total_requests": {"$sum": 1},
                "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
                "total_tokens": {"$sum": {"$ifNull": ["$tokens_used", 0]}},
                "cache_hits": {"$sum": {"$cond": [{"$eq": ["$cache_hit", True]}, 1, 0]}},
                "rag_responses": {"$sum": {"$cond": [{"$eq": ["$is_rag_response", True]}, 1, 0]}},
                "avg_confidence": {"$avg": {"$ifNull": ["$confidence_score", 0]}},
            }
        },
    ]
    results = await collection.aggregate(pipeline).to_list(length=1)
    if not results:
        return {
            "endpoint": endpoint,
            "hours": hours,
            "total_requests": 0,
            "avg_latency_ms": 0.0,
            "total_tokens": 0,
            "cache_hits": 0,
            "rag_responses": 0,
            "avg_confidence": 0.0,
        }
    r = results[0]
    return {
        "endpoint": endpoint,
        "hours": hours,
        "total_requests": r.get("total_requests", 0),
        "avg_latency_ms": round(r.get("avg_latency_ms", 0.0), 2),
        "total_tokens": r.get("total_tokens", 0),
        "cache_hits": r.get("cache_hits", 0),
        "rag_responses": r.get("rag_responses", 0),
        "avg_confidence": round(r.get("avg_confidence", 0.0), 3),
    }


async def get_system_health() -> Dict[str, Any]:
    """Reporta la salud general del sistema MongoDB.

    Hace ping al servidor y cuenta documentos en las colecciones principales.

    Returns:
        Diccionario con estado, conteos y latencia promedio de la última hora.
    """
    db = await _connection.connect()
    await _connection.ping()

    conversations_col = db[settings.MONGODB_COLL_CONVERSATIONS]
    metrics_col = db[settings.MONGODB_COLL_METRICS]
    feedback_col = db[settings.MONGODB_COLL_FEEDBACK]

    repo_conversations: BaseRepository[ConversationDocument] = BaseRepository(conversations_col, ConversationDocument)
    total_conversations = await repo_conversations.count()

    start = datetime.now(timezone.utc) - timedelta(hours=1)
    metrics_last_hour = await metrics_col.count_documents({"request_timestamp": {"$gte": start}})
    total_feedback = await feedback_col.count_documents({})

    latency_pipeline = [
        {"$match": {"request_timestamp": {"$gte": start}}},
        {"$group": {"_id": None, "avg": {"$avg": {"$ifNull": ["$latency_ms", 0]}}}},
    ]
    latency_results = await metrics_col.aggregate(latency_pipeline).to_list(length=1)
    avg_latency_ms = round(latency_results[0]["avg"], 2) if latency_results else 0.0

    return {
        "status": "healthy",
        "db_name": settings.MONGODB_DB_NAME,
        "total_conversations": total_conversations,
        "metrics_last_hour": metrics_last_hour,
        "total_feedback": total_feedback,
        "avg_latency_ms_last_hour": avg_latency_ms,
    }


class MetricsService:
    """Servicio de alto nivel para métricas."""

    async def record_metric(self, metric: MetricCreate) -> str:
        """Registra una métrica de rendimiento."""
        return await record_metric(metric)

    async def get_endpoint_metrics(self, endpoint: str, hours: int = 24) -> Dict[str, Any]:
        """Calcula métricas agregadas para un endpoint."""
        return await get_endpoint_metrics(endpoint, hours)

    async def get_system_health(self) -> Dict[str, Any]:
        """Reporta la salud general del sistema MongoDB."""
        return await get_system_health()
