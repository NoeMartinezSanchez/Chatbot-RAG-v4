"""Prueba de los servicios de MongoDB.

Uso:
    python scripts/test_services.py
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import (
    ConversationCreate,
    ConversationMessage,
    FeedbackCreate,
    MessageRole,
    MetricCreate,
)
from mongodb.services import (
    get_conversation,
    get_conversation_stats,
    get_conversations_by_session,
    get_daily_stats,
    get_endpoint_metrics,
    get_feedback_stats,
    get_system_health,
    record_feedback,
    record_metric,
    save_conversation,
    search_conversations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Ejecuta la batería de pruebas de los servicios MongoDB."""
    connection = MongoDBConnection()
    await connection.connect()

    test_session = f"test-session-{uuid.uuid4().hex[:8]}"
    test_conv_id = f"conv-{uuid.uuid4().hex[:8]}"

    try:
        # 1. Guardar una conversación de prueba
        conv = ConversationCreate(
            conversation_id=test_conv_id,
            session_id=test_session,
            user_id="test-user",
            messages=[
                ConversationMessage(role=MessageRole.USER, content="¿Qué es Prepa en Línea?", tokens=12),
                ConversationMessage(role=MessageRole.ASSISTANT, content="Es un servicio educativo gratuito de nivel medio superior.", tokens=24),
            ],
            total_tokens=36,
            latency_ms=120.5,
            is_rag_response=True,
            confidence_score=0.93,
        )
        saved = await save_conversation(conv)
        logger.info("1️⃣ Conversación guardada: %s (id=%s, %d mensajes)",
                    saved.conversation_id, saved.id, len(saved.messages))

        # 2. Recuperarla por ID
        found = await get_conversation(test_conv_id)
        assert found is not None, "No se recuperó la conversación por ID"
        assert found.conversation_id == test_conv_id
        logger.info("2️⃣ Conversación recuperada por ID: %s", found.conversation_id)

        # 3. Recuperar por sesión
        by_session = await get_conversations_by_session(test_session)
        assert any(c.conversation_id == test_conv_id for c in by_session), "No apareció en la búsqueda por sesión"
        logger.info("3️⃣ Conversaciones por sesión: %d", len(by_session))

        # 4. Guardar una métrica
        metric = MetricCreate(
            session_id=test_session,
            endpoint="/chat",
            latency_ms=95.0,
            tokens_used=1500,
            is_rag_response=True,
            confidence_score=0.90,
            cache_hit=False,
        )
        metric_id = await record_metric(metric)
        logger.info("4️⃣ Métrica guardada: %s", metric_id)

        # 5. Guardar feedback
        feedback = FeedbackCreate(
            session_id=test_session,
            conversation_id=test_conv_id,
            message_index=1,
            user_rating=5,
            user_comment="Excelente respuesta",
            is_correct=True,
        )
        feedback_id = await record_feedback(feedback)
        logger.info("5️⃣ Feedback guardado: %s", feedback_id)

        # 6. Verificar estadísticas
        session_stats = await get_conversation_stats(test_session)
        assert session_stats["total_conversations"] >= 1
        logger.info("6️⃣ Estadísticas de sesión: %s", session_stats)

        ep_metrics = await get_endpoint_metrics("/chat", hours=24)
        assert ep_metrics["total_requests"] >= 1
        logger.info("6️⃣ Métricas de endpoint /chat: %s", ep_metrics)

        fb_stats = await get_feedback_stats(days=30)
        assert fb_stats["total_feedback"] >= 1
        logger.info("6️⃣ Estadísticas de feedback: %s", fb_stats)

        daily = await get_daily_stats(days=7)
        logger.info("6️⃣ Estadísticas diarias (7 días): %d días con datos", len(daily))

        health = await get_system_health()
        assert health["status"] == "healthy"
        logger.info("6️⃣ Salud del sistema: %s", health)

        search = await search_conversations("Prepa", limit=10)
        assert len(search) >= 1
        logger.info("6️⃣ Búsqueda 'Prepa': %d resultados", len(search))

        logger.info("✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
    finally:
        # Limpieza de datos de prueba
        db = connection.get_db()
        await db[settings.MONGODB_COLL_CONVERSATIONS].delete_many({"session_id": test_session})
        await db[settings.MONGODB_COLL_METRICS].delete_many({"session_id": test_session})
        await db[settings.MONGODB_COLL_FEEDBACK].delete_many({"session_id": test_session})
        logger.info("🧹 Datos de prueba eliminados (%s)", test_session)
        await connection.disconnect()
        logger.info("🔌 Conexión cerrada correctamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("❌ La prueba de servicios falló: %s", e, exc_info=True)
        sys.exit(1)
