"""Genera un reporte JSON con estadísticas del sistema.

Uso (CLI):
    python scripts/generate_dashboard_report.py [--output data/dashboard_report.json]

Importable:
    from scripts.generate_dashboard_report import generate_report
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from mongodb.connection import MongoDBConnection
from mongodb.services import ConversationService, MetricsService, FeedbackService, RAGCacheService

logger = logging.getLogger(__name__)

_connection = MongoDBConnection()


async def generate_report(days: int = 7) -> Dict[str, Any]:
    """Genera el reporte ejecutivo con estadísticas del sistema.

    Args:
        days: Ventana de días para las estadísticas diarias (default: 7).

    Returns:
        Diccionario con el reporte completo.
    """
    conv_service = ConversationService()
    metrics_service = MetricsService()
    feedback_service = FeedbackService()
    cache_service = RAGCacheService()

    daily_stats = await conv_service.get_daily_stats(days=days)
    system_health = await metrics_service.get_system_health()
    feedback_stats = await feedback_service.get_feedback_stats(days=days)
    cache_stats = await cache_service.get_stats()

    total_daily = sum(d["count"] for d in daily_stats)

    # Resumen ejecutivo
    avg_rating = feedback_stats.get("avg_rating", 0.0)
    rating_note = (
        "Buen nivel de satisfacción"
        if avg_rating >= 4.0
        else ("Satisfacción aceptable" if avg_rating >= 3.0 else "Satisfacción baja — revisar respuestas")
    )
    health_note = "Sistema saludable" if system_health.get("status") == "healthy" else "Sistema con problemas"
    latency = system_health.get("avg_latency_ms_last_hour", 0.0)
    latency_note = "Latencia óptima" if latency < 3000 else "Latencia elevada — revisar"

    executive_summary = {
        "message": (
            f"{health_note}. {total_daily} conversaciones en los últimos {days} días. "
            f"Calificación promedio {avg_rating}/5 ({rating_note}). "
            f"{latency_note} ({latency:.0f} ms en la última hora). "
            f"{cache_stats['total_hits']} cache hits con {cache_stats['total_entries']} entradas."
        ),
        "status": system_health.get("status", "unknown"),
        "total_conversations_daily": total_daily,
        "avg_rating": avg_rating,
        "avg_latency_ms_last_hour": latency,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_window": days,
        "daily_stats": daily_stats,
        "system_health": system_health,
        "feedback_stats": feedback_stats,
        "cache_stats": cache_stats,
        "executive_summary": executive_summary,
    }


async def save_report(days: int = 7, output_path: str = None) -> Dict[str, Any]:
    """Genera y guarda el reporte en un archivo JSON.

    Args:
        days: Ventana de días para las estadísticas.
        output_path: Ruta del archivo de salida (default: data/dashboard_report.json).

    Returns:
        El reporte generado.
    """
    report = await generate_report(days=days)
    path = output_path or str(Path("data") / "dashboard_report.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("📊 Reporte guardado en: %s", path)
    return report


async def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Genera reporte de dashboard JSON")
    parser.add_argument("--days", type=int, default=7, help="Ventana de días (default: 7)")
    parser.add_argument("--output", type=str, default="data/dashboard_report.json", help="Ruta de salida")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        await _connection.connect()
        report = await save_report(days=args.days, output_path=args.output)
        print(f"\n=== REPORTE DASHBOARD ({args.days} días) ===")
        print(f"Conversaciones en periodo: {sum(d['count'] for d in report['daily_stats'])}")
        print(f"Estado del sistema: {report['system_health']['status']}")
        print(f"Feedback total: {report['feedback_stats']['total_feedback']} (promedio {report['feedback_stats']['avg_rating']}/5)")
        print(f"Cache: {report['cache_stats']['total_entries']} entradas, {report['cache_stats']['total_hits']} hits")
        print(f"\n📌 Resumen: {report['executive_summary']['message']}")
        print(f"\n✅ Reporte guardado en {args.output}")
    except Exception as e:
        print(f"❌ Error generando reporte: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await _connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
