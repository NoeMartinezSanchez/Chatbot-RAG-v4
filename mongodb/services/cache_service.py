"""Servicios para la colección de caché de respuestas RAG."""
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import RAGCacheEntry

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()


async def _get_collection() -> AsyncIOMotorCollection:
    """Retorna la colección de caché, conectando si es necesario."""
    db = await _connection.connect()
    return db[settings.MONGODB_COLL_RAG_CACHE]


def _query_hash(query: str, context: Optional[Dict] = None) -> str:
    """Genera un hash MD5 de la consulta y su contexto.

    Args:
        query: Texto de la consulta del usuario.
        context: Contexto adicional (opcional) que afecta la respuesta.

    Returns:
        Hash MD5 hexadecimal (32 caracteres).
    """
    payload = {"query": (query or "").strip().lower()}
    if context:
        payload["context"] = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normaliza un datetime a UTC aware.

    MongoDB devuelve datetimes naive (UTC implícito). Esta función los
    convierte a timezone-aware para poder compararlos con ``datetime.now(timezone.utc)``.

    Args:
        dt: Datetime a normalizar.

    Returns:
        Datetime UTC aware o ``None`` si la entrada era ``None``.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def get_cached_response(query: str, context: Optional[Dict] = None) -> Optional[RAGCacheEntry]:
    """Recupera una respuesta en caché si existe y no está expirada.

    Args:
        query: Texto de la consulta del usuario.
        context: Contexto adicional (opcional).

    Returns:
        La entrada de caché encontrada o ``None`` si no existe/expiró.
    """
    if not settings.RAG_CACHE_ENABLED:
        return None

    collection = await _get_collection()
    qh = _query_hash(query, context)
    now = datetime.now(timezone.utc)

    entry = await collection.find_one({"query_hash": qh})
    if entry is None:
        return None

    expires_at = _ensure_utc(entry.get("expires_at"))
    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at < now:
        await collection.delete_one({"query_hash": qh})
        logger.info("🗑️ Caché expirada eliminada: %s", qh)
        return None

    entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
    entry["last_accessed"] = now
    await collection.update_one(
        {"query_hash": qh},
        {"$set": {"hit_count": entry["hit_count"], "last_accessed": now}},
    )
    logger.info("🎯 Cache HIT: %s (hits=%d)", qh, entry["hit_count"])
    return RAGCacheEntry.model_validate(entry)


async def cache_response(
    query: str,
    response: str,
    sources: list,
    confidence: float,
    context: Optional[Dict] = None,
) -> RAGCacheEntry:
    """Guarda una respuesta en caché.

    Args:
        query: Texto de la consulta del usuario.
        response: Respuesta generada.
        sources: Fuentes usadas (lista de dicts).
        confidence: Confianza de la respuesta.
        context: Contexto adicional (opcional).

    Returns:
        La entrada de caché guardada.
    """
    collection = await _get_collection()
    qh = _query_hash(query, context)
    ttl_hours = settings.RAG_CACHE_TTL_HOURS
    now = datetime.now(timezone.utc)

    entry = RAGCacheEntry(
        query_hash=qh,
        query=query,
        response=response,
        sources=list(sources) if isinstance(sources, list) else [],
        confidence=float(confidence) if confidence is not None else None,
        context=context,
        hit_count=0,
        created_at=now,
        expires_at=now + timedelta(hours=ttl_hours) if ttl_hours else None,
        last_accessed=now,
    )

    await collection.replace_one(
        {"query_hash": qh},
        entry.model_dump(),
        upsert=True,
    )
    logger.info("📦 Respuesta cacheada: %s (TTL %dh)", qh, ttl_hours)
    return entry


async def get_cache_stats() -> Dict[str, Any]:
    """Calcula estadísticas de la caché.

    Returns:
        Diccionario con total de entradas, total de hits y hits promedio.
    """
    collection = await _get_collection()
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_entries": {"$sum": 1},
                "total_hits": {"$sum": {"$ifNull": ["$hit_count", 0]}},
            }
        },
    ]
    results = await collection.aggregate(pipeline).to_list(length=1)
    if not results:
        return {"total_entries": 0, "total_hits": 0, "avg_hits": 0.0}
    r = results[0]
    total_entries = r.get("total_entries", 0)
    total_hits = r.get("total_hits", 0)
    return {
        "total_entries": total_entries,
        "total_hits": total_hits,
        "avg_hits": round(total_hits / total_entries, 2) if total_entries else 0.0,
    }


async def get_expired_entries(limit: int = 1000) -> List[Dict[str, Any]]:
    """Recupera entradas de caché expiradas (para limpieza).

    Args:
        limit: Número máximo de entradas a recuperar.

    Returns:
        Lista de entradas expiradas.
    """
    collection = await _get_collection()
    now = datetime.now(timezone.utc)
    entries = await collection.find({"expires_at": {"$lt": now}}).limit(limit).to_list(length=limit)
    for entry in entries:
        if "expires_at" in entry:
            entry["expires_at"] = _ensure_utc(entry["expires_at"])
        if "created_at" in entry:
            entry["created_at"] = _ensure_utc(entry["created_at"])
        if "last_accessed" in entry:
            entry["last_accessed"] = _ensure_utc(entry["last_accessed"])
    return entries


async def cleanup_expired() -> int:
    """Elimina entradas de caché expiradas.

    Returns:
        Número de entradas eliminadas.
    """
    collection = await _get_collection()
    now = datetime.now(timezone.utc)
    result = await collection.delete_many({"expires_at": {"$lt": now}})
    if result.deleted_count:
        logger.info("🧹 %d entradas de caché expiradas eliminadas", result.deleted_count)
    return result.deleted_count


class RAGCacheService:
    """Servicio de alto nivel para la caché de respuestas RAG."""

    async def get_cached_response(self, query: str, context: Optional[Dict] = None) -> Optional[RAGCacheEntry]:
        """Recupera una respuesta en caché si existe y no está expirada."""
        return await get_cached_response(query, context)

    async def cache_response(
        self,
        query: str,
        response: str,
        sources: list,
        confidence: float,
        context: Optional[Dict] = None,
    ) -> RAGCacheEntry:
        """Guarda una respuesta en caché."""
        return await cache_response(query, response, sources, confidence, context)

    async def get_stats(self) -> Dict[str, Any]:
        """Calcula estadísticas de la caché."""
        return await get_cache_stats()

    async def cleanup_expired(self) -> int:
        """Elimina entradas de caché expiradas."""
        return await cleanup_expired()
