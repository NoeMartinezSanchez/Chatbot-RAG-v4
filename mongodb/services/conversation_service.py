"""Servicios para la colección de conversaciones."""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import DESCENDING

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import ConversationCreate, ConversationDocument
from mongodb.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()


async def _get_collection() -> AsyncIOMotorCollection:
    """Retorna la colección de conversaciones, conectando si es necesario."""
    db = await _connection.connect()
    return db[settings.MONGODB_COLL_CONVERSATIONS]


async def save_conversation(conv_data: ConversationCreate) -> ConversationDocument:
    """Guarda (inserta o actualiza) una conversación.

    Si ya existe una conversación con el mismo ``conversation_id`` se
    actualiza; en caso contrario se inserta.

    Args:
        conv_data: Datos de la conversación a guardar.

    Returns:
        La conversación persistida.

    Raises:
        RuntimeError: Si no se puede recuperar la conversación tras guardarla.
    """
    collection = await _get_collection()
    data = conv_data.model_dump()
    now = datetime.now(timezone.utc)
    existing = await collection.find_one({"conversation_id": conv_data.conversation_id})

    if existing is not None:
        await collection.update_one(
            {"conversation_id": conv_data.conversation_id},
            {"$set": {**data, "updated_at": now}},
            upsert=True,
        )
        logger.info("🔄 Conversación actualizada: %s", conv_data.conversation_id)
    else:
        await collection.insert_one({**data, "updated_at": now})
        logger.info("📝 Conversación guardada: %s", conv_data.conversation_id)

    doc = await get_conversation(conv_data.conversation_id)
    if doc is None:
        raise RuntimeError(f"No se pudo recuperar la conversación guardada: {conv_data.conversation_id}")
    return doc


async def get_conversation(conversation_id: str) -> Optional[ConversationDocument]:
    """Recupera una conversación por su ID.

    Args:
        conversation_id: ID único de la conversación.

    Returns:
        La conversación encontrada o ``None`` si no existe.
    """
    collection = await _get_collection()
    repo: BaseRepository[ConversationDocument] = BaseRepository(collection, ConversationDocument)
    return await repo.find_one({"conversation_id": conversation_id})


async def get_conversations_by_session(session_id: str, limit: int = 50) -> List[ConversationDocument]:
    """Recupera las conversaciones más recientes de una sesión.

    Args:
        session_id: ID de la sesión.
        limit: Número máximo de resultados (default: 50).

    Returns:
        Lista de conversaciones ordenadas por fecha de creación (desc).
    """
    collection = await _get_collection()
    repo: BaseRepository[ConversationDocument] = BaseRepository(collection, ConversationDocument)
    return await repo.find_many(
        {"session_id": session_id},
        limit=limit,
        sort=[("created_at", DESCENDING)],
    )


async def get_conversations_by_user(user_id: str, limit: int = 50) -> List[ConversationDocument]:
    """Recupera las conversaciones más recientes de un usuario.

    Args:
        user_id: ID del usuario.
        limit: Número máximo de resultados (default: 50).

    Returns:
        Lista de conversaciones ordenadas por fecha de creación (desc).
    """
    collection = await _get_collection()
    repo: BaseRepository[ConversationDocument] = BaseRepository(collection, ConversationDocument)
    return await repo.find_many(
        {"user_id": user_id},
        limit=limit,
        sort=[("created_at", DESCENDING)],
    )


async def get_conversation_stats(session_id: str) -> Dict[str, Any]:
    """Calcula estadísticas agregadas de una sesión.

    Args:
        session_id: ID de la sesión.

    Returns:
        Diccionario con total de conversaciones, mensajes, latencia promedio,
        tokens totales y conteo de respuestas RAG.
    """
    collection = await _get_collection()
    pipeline = [
        {"$match": {"session_id": session_id}},
        {
            "$group": {
                "_id": None,
                "total_conversations": {"$sum": 1},
                "total_messages": {"$sum": {"$size": {"$ifNull": ["$messages", []]}}},
                "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
                "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
                "rag_responses": {"$sum": {"$cond": [{"$eq": ["$is_rag_response", True]}, 1, 0]}},
            }
        },
    ]
    results = await collection.aggregate(pipeline).to_list(length=1)
    if not results:
        return {
            "session_id": session_id,
            "total_conversations": 0,
            "total_messages": 0,
            "avg_latency_ms": 0.0,
            "total_tokens": 0,
            "rag_responses": 0,
        }
    r = results[0]
    return {
        "session_id": session_id,
        "total_conversations": r.get("total_conversations", 0),
        "total_messages": r.get("total_messages", 0),
        "avg_latency_ms": round(r.get("avg_latency_ms", 0.0), 2),
        "total_tokens": r.get("total_tokens", 0),
        "rag_responses": r.get("rag_responses", 0),
    }


async def get_daily_stats(days: int = 7) -> List[Dict[str, Any]]:
    """Agrupa conversaciones por día para los últimos ``days`` días.

    Args:
        days: Número de días hacia atrás (default: 7).

    Returns:
        Lista de diccionarios con ``date`` (YYYY-MM-DD) y ``count``.
    """
    collection = await _get_collection()
    start = datetime.now(timezone.utc) - timedelta(days=days)
    pipeline = [
        {"$match": {"created_at": {"$gte": start}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    results = await collection.aggregate(pipeline).to_list(length=None)
    return [{"date": r["_id"], "count": r["count"]} for r in results]


async def get_recent_conversations(limit: int = 200) -> List[ConversationDocument]:
    """Recupera las conversaciones más recientes de todas las sesiones.

    Args:
        limit: Número máximo de resultados (default: 200).

    Returns:
        Lista de conversaciones ordenadas por fecha de creación (desc).
    """
    collection = await _get_collection()
    repo: BaseRepository[ConversationDocument] = BaseRepository(collection, ConversationDocument)
    return await repo.find_many(
        {},
        limit=limit,
        sort=[("created_at", DESCENDING)],
    )


async def search_conversations(query: str, limit: int = 10) -> List[ConversationDocument]:
    """Busca conversaciones cuyo contenido de mensajes contenga el texto.

    Args:
        query: Texto a buscar dentro de los mensajes.
        limit: Número máximo de resultados (default: 10).

    Returns:
        Lista de conversaciones que contienen el texto en algún mensaje.
    """
    collection = await _get_collection()
    repo: BaseRepository[ConversationDocument] = BaseRepository(collection, ConversationDocument)
    pattern = re.escape(query)
    filter = {
        "messages": {
            "$elemMatch": {"content": {"$regex": pattern, "$options": "i"}}
        }
    }
    return await repo.find_many(filter, limit=limit, sort=[("created_at", DESCENDING)])


class ConversationService:
    """Servicio de alto nivel para conversaciones."""

    async def save_conversation(self, conv_data: ConversationCreate) -> ConversationDocument:
        """Guarda (inserta o actualiza) una conversación."""
        return await save_conversation(conv_data)

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationDocument]:
        """Recupera una conversación por su ID."""
        return await get_conversation(conversation_id)

    async def get_conversations_by_session(self, session_id: str, limit: int = 50) -> List[ConversationDocument]:
        """Recupera las conversaciones más recientes de una sesión."""
        return await get_conversations_by_session(session_id, limit)

    async def get_conversations_by_user(self, user_id: str, limit: int = 50) -> List[ConversationDocument]:
        """Recupera las conversaciones más recientes de un usuario."""
        return await get_conversations_by_user(user_id, limit)

    async def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Calcula estadísticas agregadas de una sesión."""
        return await get_conversation_stats(session_id)

    async def get_recent_conversations(self, limit: int = 200) -> List[ConversationDocument]:
        """Recupera las conversaciones más recientes de todas las sesiones."""
        return await get_recent_conversations(limit)

    async def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Agrupa conversaciones por día para los últimos ``days`` días."""
        return await get_daily_stats(days)

    async def search_conversations(self, query: str, limit: int = 10) -> List[ConversationDocument]:
        """Busca conversaciones cuyo contenido contenga el texto."""
        return await search_conversations(query, limit)
