"""Prueba de conexión a MongoDB Atlas.

Uso:
    python scripts/test_mongodb_connection.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mongodb.connection import MongoDBConnection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Conecta a MongoDB, hace ping, lista colecciones y muestra estado."""
    connection = MongoDBConnection()

    logger.info("🟢 Conectando a MongoDB Atlas...")
    db = await connection.connect()
    logger.info("✅ Conexión establecida")

    # Hacer ping
    await connection.ping()
    logger.info("✅ Ping al servidor exitoso")

    # Listar colecciones existentes
    collections = await db.list_collection_names()
    logger.info("📚 Colecciones existentes: %d", len(collections))
    for name in sorted(collections):
        logger.info("  - %s", name)

    # Mostrar estado de conexión
    logger.info("🔌 Estado de conexión: %s", "conectado" if connection.is_connected else "desconectado")
    logger.info("🗄️ Base de datos: %s", db.name)

    # Cerrar conexión
    await connection.disconnect()
    logger.info("🔌 Conexión cerrada correctamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("❌ La prueba de conexión falló: %s", e, exc_info=True)
        sys.exit(1)
