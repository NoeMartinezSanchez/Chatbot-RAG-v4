"""Helpers de zona horaria centralizados (CDMX por defecto).

El sistema almacena en MongoDB en UTC (práctica estándar) pero la
presentación de fechas, la conciencia temporal del chatbot, el
reinicio diario de tokens y los conteos del dashboard deben operar
en la zona horaria oficial configurada via TIMEZONE (default
``America/Mexico_City``). Estas funciones son el único punto de
conversión.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config.settings import settings


def get_tz():
    """Retorna el ZoneInfo de la zona configurada (fallback CDMX)."""
    tz_name = settings.TIMEZONE or "America/Mexico_City"
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        pass
    # Fallback: offset fijo UTC-6 (CDMX sin horario de verano) si el entorno
    # no dispone de la base de datos de zonas (p.ej. Windows sin tzdata).
    return timezone(timedelta(hours=-6))


def now_local() -> datetime:
    """Fecha/hora actual en la zona horaria configurada (aware)."""
    return datetime.now(get_tz())


def today_local() -> date:
    """Fecha de hoy en la zona horaria configurada."""
    return now_local().date()


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normaliza un datetime naive (UTC implícito) a timezone-aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def start_of_today_utc() -> datetime:
    """Medianoche de HOY en la zona configurada, expresada en UTC.

    Útil para queries de Mongo (``$match: {"field": {"$gte": start}}``)
    que deben reiniciarse a las 00:00 de CDMX.
    """
    local_midnight = now_local().replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc)


def to_local(dt: Optional[datetime]) -> Optional[datetime]:
    """Convierte un datetime (naive UTC o aware) a la zona configurada."""
    dt = _ensure_utc(dt)
    if dt is None:
        return None
    return dt.astimezone(get_tz())


def format_local(dt: Optional[datetime], fmt: str = "%d/%m/%Y %H:%M") -> str:
    """Formatea un datetime (UTC naive/aware) en la zona configurada."""
    local = to_local(dt)
    if local is None:
        return "-"
    return local.strftime(fmt)