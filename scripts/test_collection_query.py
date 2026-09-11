"""Prueba del servicio de consultas de colecciones (pestaña "Consulta").

Verifica que:
  1. ``list_collections`` devuelve la whitelist con conteos.
  2. ``count`` cuenta documentos.
  3. ``find`` con sort y limit devuelve documentos serializados.
  4. ``distinct`` devuelve valores únicos de un campo.
  5. Filtro por fecha (date_field/date_from/date_to) funciona.
  6. Rechazos: colección no permitida, operador $where, distinct sin campo.

Uso:
    PYTHONIOENCODING=utf-8 python scripts/test_collection_query.py
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.services import CollectionQueryService, collection_query_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MONGODB_AVAILABLE = (
    settings.MONGODB_URI not in ("", "mongodb://localhost:27017")
    and "localhost" not in settings.MONGODB_URI.replace("127.0.0.1", "")
)


async def _insert_test_metrics(session_id: str, count: int = 5) -> None:
    """Inserta métricas de prueba en la colección real de métricas."""
    from datetime import datetime, timezone
    db = await MongoDBConnection().connect()
    col = db[settings.MONGODB_COLL_METRICS]
    docs = [
        {
            "session_id": session_id,
            "endpoint": "/chat",
            "request_timestamp": datetime(2026, 9, 10, h, 0, 0, tzinfo=timezone.utc),
            "latency_ms": 100.0 * (i + 1),
            "tokens_used": 50 * (i + 1),
            "is_rag_response": True,
            "confidence_score": 0.9,
            "cache_hit": False,
        }
        for i, h in enumerate(range(10, 10 + count))
    ]
    if docs:
        await col.insert_many(docs)


async def _insert_test_conversations(conversation_id: str, session_id: str) -> None:
    """Inserta una conversación de prueba con 2 turnos user→assistant."""
    from datetime import datetime, timezone
    db = await MongoDBConnection().connect()
    col = db[settings.MONGODB_COLL_CONVERSATIONS]

    def _msg(role: str, content: str, **extra):
        m = {
            "role": role,
            "content": content,
            "timestamp": datetime(2026, 9, 10, 16, 30, 0, tzinfo=timezone.utc),
        }
        m.update(extra)
        return m

    doc = {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": "qq-itest-user",
        "created_at": datetime(2026, 9, 10, 16, 30, 0, tzinfo=timezone.utc),
        "is_rag_response": True,
        "total_tokens": 500,
        "latency_ms": 2000.0,
        "messages": [
            _msg("user", "¿Cómo me inscribo al módulo 1?", timestamp=datetime(2026, 9, 10, 16, 30, 0, tzinfo=timezone.utc)),
            _msg("assistant",
                 "Debes ingresar a la plataforma con tu usuario y contraseña para inscribirte al módulo 1. "
                 "El periodo de inscripción queda abierto según la convocatoria vigente.",
                 timestamp=datetime(2026, 9, 10, 16, 30, 10, tzinfo=timezone.utc),
                 tokens=120, latency_ms=726.0, is_rag=True),
            _msg("user", "hola", timestamp=datetime(2026, 9, 10, 16, 31, 0, tzinfo=timezone.utc)),
            _msg("assistant",
                 "¡Hola! ¿En qué puedo ayudarte hoy? Recuerda que puedes preguntar sobre trámites, "
                 "inscripciones y el funcionamiento de Prepa en Línea SEP.",
                 timestamp=datetime(2026, 9, 10, 16, 31, 5, tzinfo=timezone.utc),
                 tokens=25, latency_ms=400.0, is_rag=False),
        ],
    }
    await col.insert_one(doc)


async def _cleanup(session_id: str) -> None:
    db = MongoDBConnection().get_db()
    await db[settings.MONGODB_COLL_METRICS].delete_many({"session_id": session_id})
    await db[settings.MONGODB_COLL_CONVERSATIONS].delete_many({"session_id": session_id})


async def _test_validation() -> None:
    logger.info("1️⃣ Validación/seguridad del servicio...")
    svc = collection_query_service

    # Colección no permitida
    try:
        await svc.run_query(collection="no_existe", operation="count")
        raise AssertionError("Debería rechazar colección no permitida")
    except ValueError as e:
        assert "no permitida" in str(e), str(e)
    logger.info("   ✅ Rechaza colección no permitida")

    # Operador bloqueado
    try:
        await svc.run_query(collection=settings.MONGODB_COLL_METRICS,
                            operation="find", filter_doc={"$where": "this.a > 1"})
        raise AssertionError("Debería rechazar $where")
    except ValueError as e:
        assert "$where" in str(e), str(e)
    logger.info("   ✅ Rechaza operador $where")

    # distinct sin campo
    try:
        await svc.run_query(collection=settings.MONGODB_COLL_METRICS,
                            operation="distinct")
        raise AssertionError("Debería requerir distinct_field")
    except ValueError as e:
        assert "distinct_field" in str(e), str(e)
    logger.info("   ✅ Requiere distinct_field")

    # Operación inválida
    try:
        await svc.run_query(collection=settings.MONGODB_COLL_METRICS,
                            operation="drop")
        raise AssertionError("Debería rechazar operación no soportada")
    except ValueError as e:
        assert "no soportada" in str(e), str(e)
    logger.info("   ✅ Rechaza operación no soportada")


async def _test_with_mongo() -> None:
    session_id = "qq-itest-" + uuid.uuid4().hex[:6]
    try:
        logger.info("4️⃣ Probando consultas reales en MongoDB...")
        await _insert_test_metrics(session_id)

        svc = collection_query_service

        # list_collections
        cols = await svc.list_collections()
        names = [c["name"] for c in cols]
        assert settings.MONGODB_COLL_METRICS in names, names
        logger.info("   ✅ list_collections incluye %d colecciones",
                    len(cols))

        # count
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="count",
            filter_doc={"session_id": session_id})
        assert res["total"] == 5, res
        assert res["operation"] == "count"
        logger.info("   ✅ count devuelve %d", res["total"])

        # find con sort desc por request_timestamp
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="find",
            filter_doc={"session_id": session_id},
            sort_doc={"request_timestamp": -1},
            limit=3)
        assert res["returned"] == 3, res
        assert res["total"] == 5, res
        assert "tokens_used" in res["fields"], res["fields"]
        # Más reciente primero: 14:00 UTC = 08:00 CDMX (se muestra en CDMX)
        assert res["docs"][0]["request_timestamp"].startswith("2026-09-10 08"), res["docs"][0]
        # Datetimes serializados a string ISO
        assert isinstance(res["docs"][0]["request_timestamp"], str)
        logger.info("   ✅ find con sort/limit (3 de 5, más reciente primero)")

        # Límite tope
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="find",
            filter_doc={"session_id": session_id},
            limit=99999)
        assert res["returned"] <= 500, res["returned"]
        logger.info("   ✅ Límite tope aplicado (<=500)")

        # distinct
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="distinct",
            filter_doc={"session_id": session_id},
            distinct_field="endpoint")
        assert res["total"] == 1, res
        assert res["docs"][0]["valor"] == "/chat", res
        logger.info("   ✅ distinct de endpoint")

        # Filtro por fecha: 04:00-06:00 = 10:00 UTC, 09:00-06:00 = 15:00 UTC
        # cubre todos los docs (10..14 UTC)
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="find",
            filter_doc={"session_id": session_id},
            date_field="request_timestamp",
            date_from="2026-09-10T04:00:00-06:00",
            date_to="2026-09-10T09:00:00-06:00")
        assert res["total"] == 5, res

        # Sub-rango: 15:00-06:00 = 21:00 UTC, 17:00-06:00 = 23:00 UTC → 0 docs
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_METRICS,
            operation="find",
            filter_doc={"session_id": session_id},
            date_field="request_timestamp",
            date_from="2026-09-10T15:00:00-06:00",
            date_to="2026-09-10T17:00:00-06:00")
        assert res["total"] == 0, res
        logger.info("   ✅ Filtro por fecha (rango completo y sub-rango vacío)")

        # ===== Conversaciones: vista "tipo historial" =====
        conv_id = "qq-conv-" + uuid.uuid4().hex[:6]
        await _insert_test_conversations(conv_id, session_id)

        res = await svc.run_query(
            collection=settings.MONGODB_COLL_CONVERSATIONS,
            operation="find",
            filter_doc={"session_id": session_id})
        assert res["fields"] == ["fecha", "pregunta", "respuesta", "tiempo", "tokens", "rag"], res["fields"]
        assert res["total"] == 2, res  # 2 turnos user→assistant
        assert res["returned"] == 2, res
        rows = res["docs"]
        # El texto de pregunta/respuesta se conserva COMPLETO (no truncado)
        assert rows[0]["pregunta"] == "¿Cómo me inscribo al módulo 1?", rows[0]
        assert rows[0]["respuesta"].startswith("Debes ingresar a la plataforma"), rows[0]
        assert rows[0]["tiempo"] == "726ms", rows[0]
        assert rows[0]["tokens"] == 120, rows[0]
        assert rows[0]["rag"] == "Sí", rows[0]
        assert rows[1]["pregunta"] == "hola", rows[1]
        assert rows[1]["rag"] == "No", rows[1]
        assert rows[1]["tiempo"] == "400ms", rows[1]
        # La fecha usa hora CDMX (16:30 UTC = 10:30 CDMX)
        assert rows[0]["fecha"].startswith("10/09/2026 10:30"), rows[0]
        logger.info("   ✅ find conversations aplana turnos (total 2, texto completo)")

        # count sobre conversations NO aplana (sigue contando documentos)
        res = await svc.run_query(
            collection=settings.MONGODB_COLL_CONVERSATIONS,
            operation="count",
            filter_doc={"session_id": session_id})
        assert res["total"] == 1, res
        logger.info("   ✅ count sobre conversations cuenta documentos")

        logger.info("5️⃣ Pruebas de consulta PASSED")
    finally:
        await _cleanup(session_id)


async def main() -> None:
    """Ejecuta la batería de pruebas del servicio de consulta."""
    logger.info("🗄️ Probando service de consulta de colecciones...")
    await _test_validation()

    if not MONGODB_AVAILABLE:
        logger.warning("⏭️  MongoDB no configurado (URI default localhost). "
                       "Se omiten las pruebas reales de consultas.")
    else:
        await _test_with_mongo()

    logger.info("✅ TODAS LAS PRUEBAS DE COLLECTION QUERY PASARON CORRECTAMENTE")


if __name__ == "__main__":
    asyncio.run(main())