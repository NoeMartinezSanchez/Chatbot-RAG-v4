"""Servicios para la colección de feedback."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import FeedbackCreate

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()


async def _get_collection() -> AsyncIOMotorCollection:
    """Retorna la colección de feedback, conectando si es necesario."""
    db = await _connection.connect()
    return db[settings.MONGODB_COLL_FEEDBACK]


async def record_feedback(feedback: FeedbackCreate) -> str:
    """Registra el feedback de un usuario y lo refleja en la conversación.

    Args:
        feedback: Datos del feedback a registrar.

    Returns:
        El ID (string) del documento insertado.
    """
    collection = await _get_collection()
    result = await collection.insert_one(feedback.model_dump())
    logger.info("⭐ Feedback registrado: %s", result.inserted_id)

    if feedback.conversation_id is not None and feedback.message_index is not None:
        try:
            await _update_conversation_feedback(
                feedback.conversation_id, feedback.message_index, feedback.user_rating
            )
        except Exception as e:
            logger.warning("⚠️ No se pudo reflejar el feedback en la conversación: %s", e)

    return str(result.inserted_id)


async def _update_conversation_feedback(
    conversation_id: str,
    message_index: int,
    rating: Optional[int],
) -> None:
    """Actualiza el feedback del mensaje indicado dentro de una conversación.

    Args:
        conversation_id: ID de la conversación.
        message_index: Índice del mensaje que recibe el feedback.
        rating: Calificación otorgada (1-5).
    """
    db = await _connection.connect()
    conversations = db[settings.MONGODB_COLL_CONVERSATIONS]
    feedback_data = {
        "user_rating": rating,
        "timestamp": datetime.now(timezone.utc),
    }
    await conversations.update_one(
        {"conversation_id": conversation_id},
        {"$set": {f"messages.{message_index}.feedback": feedback_data}},
    )
    logger.info("💬 Feedback aplicado al mensaje %d de %s", message_index, conversation_id)


async def get_feedback_stats(days: int = 30) -> Dict[str, Any]:
    """Calcula estadísticas de feedback para los últimos ``days`` días.

    Args:
        days: Ventana de tiempo hacia atrás en días (default: 30).

    Returns:
        Diccionario con total de feedback, calificación promedio, distribución
        de calificaciones y conteo de correctos/incorrectos.
    """
    collection = await _get_collection()
    start = datetime.now(timezone.utc) - timedelta(days=days)
    match = {"created_at": {"$gte": start}}

    totals = await collection.aggregate([
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "total": {"$sum": 1},
                "avg_rating": {"$avg": {"$ifNull": ["$user_rating", 0]}},
                "correct_count": {"$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}},
                "incorrect_count": {"$sum": {"$cond": [{"$eq": ["$is_correct", False]}, 1, 0]}},
            }
        },
    ]).to_list(length=1)

    distribution = await collection.aggregate([
        {"$match": {**match, "user_rating": {"$ne": None}}},
        {"$group": {"_id": "$user_rating", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]).to_list(length=None)

    if not totals:
        return {
            "days": days,
            "total_feedback": 0,
            "avg_rating": 0.0,
            "correct_count": 0,
            "incorrect_count": 0,
            "rating_distribution": {},
        }

    t = totals[0]
    return {
        "days": days,
        "total_feedback": t.get("total", 0),
        "avg_rating": round(t.get("avg_rating", 0.0), 2),
        "correct_count": t.get("correct_count", 0),
        "incorrect_count": t.get("incorrect_count", 0),
        "rating_distribution": {str(d["_id"]): d["count"] for d in distribution},
    }


class FeedbackService:
    """Servicio de alto nivel para feedback."""

    async def record_feedback(self, feedback: FeedbackCreate) -> str:
        """Registra feedback y lo refleja en la conversación."""
        return await record_feedback(feedback)

    async def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """Calcula estadísticas de feedback."""
        return await get_feedback_stats(days)
