"""Política de retención: elimina documentos viejos de MongoDB por colección.

Uso (CLI):
    python scripts/retention_policy.py [--dry-run]

Importable:
    from scripts.retention_policy import run_retention, RETENTION_RULES
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from mongodb.connection import MongoDBConnection

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()

# Reglas: (colección, campo_timestamp, días_a_conservar)
RETENTION_RULES: List[Tuple[str, str, int]] = [
    (settings.MONGODB_COLL_CONVERSATIONS, "created_at", settings.RETENTION_CONVERSATIONS_DAYS),
    (settings.MONGODB_COLL_METRICS, "request_timestamp", settings.RETENTION_METRICS_DAYS),
    (settings.MONGODB_COLL_LOGS, "timestamp", settings.RETENTION_LOGS_DAYS),
    (settings.MONGODB_COLL_FEEDBACK, "created_at", settings.RETENTION_FEEDBACK_DAYS),
]

COLLECTION_LABELS: Dict[str, str] = {
    settings.MONGODB_COLL_CONVERSATIONS: "Conversaciones",
    settings.MONGODB_COLL_METRICS: "Métricas",
    settings.MONGODB_COLL_LOGS: "Logs",
    settings.MONGODB_COLL_FEEDBACK: "Feedback",
}


async def _cleanup_collection(collection: str, timestamp_field: str, days: int) -> int:
    """Elimina documentos más viejos que ``days`` días en una colección.

    Args:
        collection: Nombre de la colección.
        timestamp_field: Campo con el timestamp de creación.
        days: Días a conservar (documentos más viejos se eliminan).

    Returns:
        Número de documentos eliminados.
    """
    db = await _connection.connect()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db[collection].delete_many({timestamp_field: {"$lt": cutoff}})
    return result.deleted_count


async def run_retention(dry_run: bool = False) -> Dict[str, Any]:
    """Ejecuta la limpieza por política de retención.

    Args:
        dry_run: Si True, solo cuenta documentos sin eliminarlos.

    Returns:
        Diccionario con resultados por colección y total eliminado.
    """
    results: Dict[str, Any] = {"dry_run": dry_run, "collections": [], "total_deleted": 0}

    for collection, timestamp_field, days in RETENTION_RULES:
        deleted = await _cleanup_collection(collection, timestamp_field, days)
        label = COLLECTION_LABELS.get(collection, collection)
        results["collections"].append({
            "collection": collection,
            "label": label,
            "retention_days": days,
            "deleted": deleted,
            "field": timestamp_field,
        })
        results["total_deleted"] += deleted
        action = "Eliminaría" if dry_run else "Eliminados"
        logger.info("%s %d documentos de %s (> %d días)", action, deleted, label, days)

    return results


async def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Política de retención de MongoDB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo cuenta documentos sin eliminarlos",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        results = await run_retention(dry_run=args.dry_run)
        print("\n=== RESUMEN DE RETENCIÓN ===")
        print(f"Modo: {'DRY RUN (nada se eliminó)' if results['dry_run'] else 'Ejecución real'}")
        for c in results["collections"]:
            status = "Eliminaría" if results["dry_run"] else "Eliminó"
            print(f"  {c['label']:<16} {status} {c['deleted']:>5} docs  (retención: {c['retention_days']} días)")
        print(f"\nTotal: {results['total_deleted']} documentos")
        print("✅ Limpieza completada")
    except Exception as e:
        print(f"❌ Error en retención: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await _connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
