"""Prueba de integración completa: endpoints HTTP con persistencia MongoDB.

Uso:
    python scripts/test_integration.py
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
import httpx

from config.settings import settings
from api import endpoints
from langchain_layer.wrappers import LangChainRAGWrapper
from mongodb.connection import MongoDBConnection
from mongodb.services import ConversationService, MetricsService, FeedbackService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class StubRAG:
    """Sistema RAG ficticio para evitar dependencias pesadas y llamadas externas."""

    def process_query(self, question):
        return (
            "El módulo propedéutico es obligatorio para todos los aspirantes de nuevo ingreso.",
            True,
            0.95,
            [{"metadata": {"source_file": "stub_source"}}],
        )


def _stub_wrapper():
    return LangChainRAGWrapper(StubRAG(), memory_enabled=False, mongodb_enabled=True)


async def main() -> None:
    """Ejecuta la batería de integración HTTP + MongoDB."""
    connection = MongoDBConnection()
    await connection.connect()

    test_session = f"itest-session-{uuid.uuid4().hex[:8]}"

    # Reemplazar el wrapper pesado por uno con RAG stub (evita FAISS/Groq reales)
    endpoints.get_langchain_wrapper = _stub_wrapper

    app = FastAPI(title="Test Integration MongoDB")
    app.include_router(endpoints.mongodb_router)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # ===== 1. Endpoint /chat (schema question) =====
            logger.info("1️⃣ Probando POST /chat (question)...")
            r = await client.post("/chat", json={
                "question": "¿El módulo propedéutico es obligatorio?",
                "session_id": test_session,
                "user_id": "itest-user",
            })
            assert r.status_code == 200, f"/chat status {r.status_code}: {r.text}"
            data = r.json()
            assert data["response"], "Respuesta vacía en /chat"
            assert data["conversation_id"], "Falta conversation_id en /chat"
            conv_id = data["conversation_id"]
            logger.info("   ✅ /chat respondió (conversation_id=%s)", conv_id)

            # Esperar a que el guardado en background termine
            await asyncio.sleep(2.5)

            # ===== 2. Verificar que la conversación se guardó en MongoDB =====
            logger.info("2️⃣ Verificando persistencia en MongoDB...")
            convs = await ConversationService().get_conversations_by_session(test_session)
            assert any(c.conversation_id == conv_id for c in convs), "Conversación no guardada en MongoDB"
            logger.info("   ✅ Conversación encontrada en MongoDB (%d en sesión)", len(convs))

            # ===== 3. Endpoint /chat con campo legacy `message` (backward compat) =====
            logger.info("3️⃣ Probando POST /chat con `message` (backward compat)...")
            r2 = await client.post("/chat", json={
                "message": "¿Cuánto dura el trayecto?",
                "session_id": test_session,
            })
            assert r2.status_code == 200, f"/chat legacy status {r2.status_code}: {r2.text}"
            assert r2.json()["response"]
            logger.info("   ✅ /chat con `message` funcionó")
            await asyncio.sleep(2.5)

            # ===== 4. Endpoint /feedback =====
            logger.info("4️⃣ Probando POST /feedback...")
            r = await client.post("/feedback", json={
                "session_id": test_session,
                "conversation_id": conv_id,
                "message_index": 1,
                "user_rating": 5,
                "user_comment": "Excelente respuesta",
                "is_correct": True,
            })
            assert r.status_code == 200, f"/feedback status {r.status_code}: {r.text}"
            fb = r.json()
            assert fb["status"] == "success"
            assert fb["feedback_id"], "Falta feedback_id"
            logger.info("   ✅ /feedback registrado (feedback_id=%s)", fb["feedback_id"])

            # ===== 5. Endpoint /analytics =====
            logger.info("5️⃣ Probando GET /analytics...")
            r = await client.get("/analytics", params={"session_id": test_session, "days": 7})
            assert r.status_code == 200, f"/analytics status {r.status_code}: {r.text}"
            ana = r.json()
            assert ana["status"] == "success", ana
            assert ana["analytics"]["system_health"]["status"] == "healthy"
            assert ana["analytics"]["conversation_stats"]["total_conversations"] >= 1
            assert ana["analytics"]["feedback_stats"]["total_feedback"] >= 1
            logger.info("   ✅ /analytics devolvió estadísticas combinadas")

            # ===== 6. Métricas registradas =====
            logger.info("6️⃣ Verificando métricas en MongoDB...")
            ep_metrics = await MetricsService().get_endpoint_metrics("/chat", hours=24)
            assert ep_metrics["total_requests"] >= 2
            logger.info("   ✅ Métricas del endpoint /chat: %d solicitudes", ep_metrics["total_requests"])

        logger.info("✅ TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON CORRECTAMENTE")

    finally:
        # Limpieza de datos de prueba
        db = connection.get_db()
        await db[settings.MONGODB_COLL_CONVERSATIONS].delete_many({"session_id": test_session})
        await db[settings.MONGODB_COLL_METRICS].delete_many({"session_id": test_session})
        await db[settings.MONGODB_COLL_FEEDBACK].delete_many({"session_id": test_session})
        logger.info("🧹 Datos de integración eliminados (%s)", test_session)
        await connection.disconnect()
        logger.info("🔌 Conexión cerrada correctamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("❌ La prueba de integración falló: %s", e, exc_info=True)
        sys.exit(1)
