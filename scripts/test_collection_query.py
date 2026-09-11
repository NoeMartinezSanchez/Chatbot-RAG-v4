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


async def _cleanup(session_id: str) -> None:
    db = MongoDBConnection().get_db()
    await db[settings.MONGODB_COLL_METRICS].delete_many({"session_id": session_id})


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