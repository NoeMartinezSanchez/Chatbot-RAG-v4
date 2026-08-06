"""Prueba del servicio de caché de respuestas RAG.

Uso:
    python scripts/test_cache.py
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mongodb.connection import MongoDBConnection
from mongodb.services import RAGCacheService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Ejecuta la batería de pruebas del cache service."""
    connection = MongoDBConnection()
    await connection.connect()

    test_query = f"¿Qué es el módulo propedéutico? {uuid.uuid4().hex[:6]}"
    test_response = "El módulo propedéutico es obligatorio para todos los aspirantes."
    test_sources = [{"metadata": {"source_file": "test.xlsx"}}]
    test_confidence = 0.95

    service = RAGCacheService()

    try:
        # ===== 1. Guardar respuesta en caché =====
        logger.info("1️⃣ Guardando respuesta en caché...")
        entry = await service.cache_response(
            query=test_query,
            response=test_response,
            sources=test_sources,
            confidence=test_confidence,
        )
        assert entry.query_hash, "Falta query_hash"
        logger.info("   ✅ Respuesta cacheada (hash=%s)", entry.query_hash[:8])
        await asyncio.sleep(1)

        # ===== 2. Recuperar respuesta (verificar hit) =====
        logger.info("2️⃣ Recuperando respuesta (debería ser HIT)...")
        cached = await service.get_cached_response(query=test_query)
        assert cached is not None, "No se encontró la respuesta en caché"
        assert cached.response == test_response, "La respuesta cacheada no coincide"
        logger.info("   ✅ Cache HIT verificado (hits=%d)", cached.hit_count)

        # ===== 3. Contexto diferente (no debe encontrar) =====
        logger.info("3️⃣ Buscando con contexto diferente (debería ser MISS)...")
        different_context = {"user": "otro-usuario"}
        cached_ctx = await service.get_cached_response(query=test_query, context=different_context)
        assert cached_ctx is None, "No debería encontrar caché con contexto distinto"
        logger.info("   ✅ Cache MISS con contexto diferente verificado")

        # ===== 4. Consulta diferente (no debe encontrar) =====
        logger.info("4️⃣ Buscando con consulta diferente (debería ser MISS)...")
        other_query = "Otra pregunta completamente distinta 12345"
        cached_other = await service.get_cached_response(query=other_query)
        assert cached_other is None, "No debería encontrar caché para otra consulta"
        logger.info("   ✅ Cache MISS para consulta distinta verificado")

        # ===== 5. Verificar incremento de hit_count =====
        logger.info("5️⃣ Verificando incremento de hit_count...")
        await service.get_cached_response(query=test_query)
        await service.get_cached_response(query=test_query)
        cached_final = await service.get_cached_response(query=test_query)
        assert cached_final is not None, "Caché desapareció tras los hits"
        assert cached_final.hit_count >= 4, (
            f"hit_count esperado >= 4, obtenido {cached_final.hit_count}"
        )
        logger.info("   ✅ hit_count incrementado correctamente (hits=%d)", cached_final.hit_count)

        # ===== 6. Estadísticas =====
        logger.info("6️⃣ Consultando estadísticas de caché...")
        stats = await service.get_stats()
        assert stats["total_entries"] >= 1
        logger.info("   ✅ Stats: %d entradas, %d hits", stats["total_entries"], stats["total_hits"])

        logger.info("✅ TODAS LAS PRUEBAS DE CACHÉ PASARON CORRECTAMENTE")

    finally:
        # Limpieza de datos de prueba
        db = connection.get_db()
        await db["rag_cache"].delete_many({"query": test_query})
        logger.info("🧹 Datos de prueba de caché eliminados")
        await connection.disconnect()
        logger.info("🔌 Conexión cerrada correctamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("❌ La prueba de caché falló: %s", e, exc_info=True)
        sys.exit(1)
