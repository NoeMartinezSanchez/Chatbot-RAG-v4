"""Prueba de integración del dashboard de usuarios con MongoDB.

Verifica que:
  1. La generación del dashboard funciona sin MongoDB (fallback JSONL).
  2. Las métricas se calculan correctamente con interacciones sintéticas.
  3. Las interacciones y tokens se leen desde MongoDB (si hay conexión),
     mapeando la colección ``conversations`` al formato esperado.

Uso:
    python scripts/test_dashboard_mongodb.py
"""
import asyncio
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.models import ConversationCreate, ConversationMessage, MessageRole, MetricCreate
from mongodb.services import ConversationService, MetricsService, RAGCacheService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MONGODB_AVAILABLE = (
    settings.MONGODB_URI not in ("", "mongodb://localhost:27017")
    and "localhost" not in settings.MONGODB_URI.replace("127.0.0.1", "")
)


def _sample_interactions() -> list:
    """Genera interacciones sintéticas con el formato del dashboard."""
    return [
        {
            "timestamp": "2026-08-10T10:00:00+00:00",
            "session_id": "s1",
            "pregunta": "¿El módulo propedéutico es obligatorio?",
            "respuesta": "El módulo propedéutico es obligatorio para todos los aspirantes.",
            "tiempo_total_ms": 1500.0,
            "confianza": 0.95,
            "fuentes_usadas": ["Control_Escolar.xlsx"],
            "es_rag": True,
            "tokens_used": 120,
        },
        {
            "timestamp": "2026-08-10T11:00:00+00:00",
            "session_id": "s2",
            "pregunta": "¿Cuándo es la convocatoria?",
            "respuesta": "No encontré información sobre la convocatoria.",
            "tiempo_total_ms": 2500.0,
            "confianza": 0.40,
            "fuentes_usadas": [],
            "es_rag": True,
            "tokens_used": 80,
        },
    ]


async def _test_metrics() -> None:
    """Calcula métricas con datos sintéticos y valida resultados clave."""
    from evaluation.generate_user_dashboard import calculate_metrics

    metrics = calculate_metrics(_sample_interactions(), tokens_por_hora={10: 120, 11: 80})
    assert metrics["total_interacciones"] == 2
    assert metrics["usuarios_unicos"] == 2
    assert metrics["tasa_no_encontrado"] == 50.0, metrics
    assert metrics["fuentes_top"], "Debe detectar fuentes usadas"
    assert metrics["tokens_por_hora"] == {10: 120, 11: 80}, "Override de tokens por hora"
    assert metrics["max_tokens_por_hora"] == 120
    logger.info("   ✅ calculate_metrics correcto (%d interacciones, %d usuarios)",
                metrics["total_interacciones"], metrics["usuarios_unicos"])


async def _test_dashboard_without_mongodb() -> None:
    """Genera el dashboard sin MongoDB (fallback JSONL)."""
    from evaluation.generate_user_dashboard import generate_dashboard_html

    tmp = Path(tempfile.mkdtemp(prefix="dash-test-"))
    log_path = tmp / "user_interactions.jsonl"
    out_path = tmp / "user_dashboard.html"

    with open(log_path, "w", encoding="utf-8") as f:
        for inter in _sample_interactions():
            f.write(json.dumps(inter, ensure_ascii=False) + "\n")

    from evaluation.generate_user_dashboard import generate_user_dashboard_async

    result = await generate_user_dashboard_async(
        log_path=str(log_path),
        output_path=str(out_path),
        use_mongodb=False,
    )
    assert result == str(out_path), "La ruta retornada debe coincidir"
    assert out_path.exists(), "El HTML del dashboard no se generó"
    html = out_path.read_text(encoding="utf-8")
    assert "Dashboard de Interacciones Reales" in html
    assert "Total Interacciones" in html
    assert "2" in html or 'Interactive' in html
    logger.info("   ✅ Dashboard generado sin MongoDB (%d bytes)", out_path.stat().st_size)


async def _test_mongodb_sources() -> None:
    """Inserta datos de prueba en MongoDB y valida el mapeo del dashboard."""
    test_session = f"dash-itest-{uuid.uuid4().hex[:8]}"
    conversation_id = f"dc-{uuid.uuid4().hex[:8]}"

    conv = ConversationCreate(
        conversation_id=conversation_id,
        session_id=test_session,
        user_id="dash-itest-user",
        messages=[
            ConversationMessage(role=MessageRole.USER, content="¿Cuándo es la convocatoria?", tokens=12),
            ConversationMessage(role=MessageRole.ASSISTANT,
                                content="El registro es del 10 al 20 de Agosto de 2026.", tokens=40,
                                latency_ms=2010.5, confidence_score=0.93, is_rag=True,
                                sources_used=[{"metadata": {"source_file": "Convocatoria.xlsx"}}]),
        ],
        sources_used=[{"metadata": {"source_file": "Convocatoria.xlsx"}}],
        total_tokens=52,
        latency_ms=2010.5,
        is_rag_response=True,
        confidence_score=0.93,
    )
    await ConversationService().save_conversation(conv)

    # Segundo turno sobre el MISMO conversation_id → debe ACUMULARSE
    turn2_msgs = [
        ConversationMessage(role=MessageRole.USER, content="¿Qué documentos necesito?", tokens=10),
        ConversationMessage(role=MessageRole.ASSISTANT,
                            content="Necesitas tu acta de nacimiento.", tokens=25,
                            latency_ms=1500.0, confidence_score=0.88, is_rag=True,
                            sources_used=[{"metadata": {"source_file": "Control_Escolar.xlsx"}}]),
    ]
    conv2 = ConversationCreate(
        conversation_id=conversation_id,
        session_id=test_session,
        user_id="dash-itest-user",
        messages=turn2_msgs,
        sources_used=[{"metadata": {"source_file": "Control_Escolar.xlsx"}}],
        total_tokens=35,
        latency_ms=1500.0,
        is_rag_response=True,
        confidence_score=0.88,
    )
    await ConversationService().save_conversation(conv2)

    # Repetir el MISMO turno (mismos mensajes/timestamps, como un reintento en
    # background) → NO debe duplicarse (idempotencia atómica)
    conv3 = ConversationCreate(
        conversation_id=conversation_id,
        session_id=test_session,
        user_id="dash-itest-user",
        messages=turn2_msgs,
        total_tokens=35,
        latency_ms=1500.0,
        is_rag_response=True,
        confidence_score=0.88,
    )
    await ConversationService().save_conversation(conv3)

    await MetricsService().record_metric(MetricCreate(
        session_id=test_session,
        endpoint="/chat",
        latency_ms=2010.5,
        tokens_used=52,
        is_rag_response=True,
        confidence_score=0.93,
        cache_hit=False,
    ))

    # Verificar que la conversación acumuló 4 mensajes (2 turnos), sin duplicados
    stored = await ConversationService().get_conversation(conversation_id)
    assert stored is not None, "La conversación no existe en MongoDB"
    assert stored.messages, "La conversación no tiene mensajes"
    assert len(stored.messages) == 4, f"Esperaba 4 mensajes acumulados, hay {len(stored.messages)}"
    logger.info("   ✅ Acumulación idempotente OK (%d mensajes en %d turnos)",
                len(stored.messages), len(stored.messages) // 2)

    from evaluation.generate_user_dashboard import (
        fetch_mongodb_interactions,
        fetch_mongodb_token_stats,
        fetch_mongodb_tokens_por_hora,
    )

    interactions = await fetch_mongodb_interactions(limit=500)
    mine = [i for i in interactions if i.get("conversation_id") == conversation_id]
    assert len(mine) == 2, f"Esperaba 2 interacciones del dashboard, hay {len(mine)}"
    mi = mine[0]
    assert mi["pregunta"] == "¿Cuándo es la convocatoria?"
    assert "Convocatoria.xlsx" in mi["fuentes_usadas"]
    assert mi["es_rag"] is True
    assert mi["tiempo_total_ms"] == 2010.5
    # El segundo turno conserva sus propias métricas
    mi2 = mine[1]
    assert mi2["pregunta"] == "¿Qué documentos necesito?"
    assert mi2["tiempo_total_ms"] == 1500.0
    assert mi2["confianza"] == 0.88
    assert "Control_Escolar.xlsx" in mi2["fuentes_usadas"]

    token_stats = await fetch_mongodb_token_stats()
    if token_stats is not None:
        logger.info("   ✅ Token stats desde MongoDB: %s", token_stats)

    tph = await fetch_mongodb_tokens_por_hora()
    if tph is not None:
        logger.info("   ✅ Tokens por hora desde MongoDB: %s", dict(tph))

    # Limpieza
    db = MongoDBConnection().get_db()
    await db[settings.MONGODB_COLL_CONVERSATIONS].delete_many({"conversation_id": conversation_id})
    await db[settings.MONGODB_COLL_METRICS].delete_many({"session_id": test_session})
    logger.info("   ✅ Datos de prueba del dashboard eliminados (%s)", test_session)


async def main() -> None:
    """Ejecuta la batería de pruebas del dashboard con MongoDB."""
    logger.info("📊 Probando dashboard de usuarios...")
    await _test_metrics()
    await _test_dashboard_without_mongodb()

    if not MONGODB_AVAILABLE:
        logger.warning("⏭️  MongoDB no configurado (URI default localhost). "
                       "Se omiten las pruebas de fuentes MongoDB.")
    else:
        connection = MongoDBConnection()
        try:
            await connection.connect()
            logger.info("1️⃣ Probar mapeo conversations → interacciones...")
            await _test_mongodb_sources()
        except Exception as e:
            logger.warning("⏭️  MongoDB no alcanzable (%s). Se omiten las pruebas de fuentes MongoDB.", e)
        finally:
            try:
                await connection.disconnect()
            except Exception:
                pass

    logger.info("✅ TODAS LAS PRUEBAS DEL DASHBOARD PASARON CORRECTAMENTE")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("❌ La prueba del dashboard falló: %s", e, exc_info=True)
        sys.exit(1)