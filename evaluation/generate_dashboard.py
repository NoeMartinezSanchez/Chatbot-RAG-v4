"""Dashboard Generator for Evaluation Results.

Reads evaluation results and generates an HTML dashboard.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# Ruta absoluta para HF Spaces - usar el directorio donde está el script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
EVALUATION_LOG = LOG_DIR / "evaluation_results.jsonl"

# Dashboard se guarda en la misma carpeta que el script (evaluation/)
OUTPUT_PATH = SCRIPT_DIR / "dashboard.html"


def load_results() -> list:
    """Load evaluation results from JSONL file."""
    results = []
    if not EVALUATION_LOG.exists():
        print(f"⚠️ No se encontró: {EVALUATION_LOG}")
        return results
    
    with open(EVALUATION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def calculate_metrics(results: list) -> dict:
    """Calculate summary metrics from results."""
    if not results:
        return {
            "total": 0,
            "correctas": 0,
            "incorrectas": 0,
            "tasa_exito": 0,
            "avg_latency": 0,
            "avg_retrieval": 0,
            "avg_generation": 0
        }
    
    total = len(results)
    correctas = sum(1 for r in results if r.get("correcto", False))
    incorrectas = total - correctas
    tasa_exito = round(correctas / total * 100, 1) if total > 0 else 0
    
    avg_latency = round(sum(r.get("latency_ms", 0) for r in results) / total, 1) if total > 0 else 0
    avg_retrieval = round(sum(r.get("retrieval_time_ms", 0) for r in results) / total, 1) if total > 0 else 0
    avg_generation = round(sum(r.get("generation_time_ms", 0) for r in results) / total, 1) if total > 0 else 0
    
    return {
        "total": total,
        "correctas": correctas,
        "incorrectas": incorrectas,
        "tasa_exito": tasa_exito,
        "avg_latency": avg_latency,
        "avg_retrieval": avg_retrieval,
        "avg_generation": avg_generation,
        "last_evaluation": results[-1].get("timestamp", "N/A") if results else "N/A"
    }


def get_difficulty_stats(results: list) -> dict:
    """Get success rate by difficulty."""
    stats = defaultdict(lambda: {"total": 0, "correctas": 0})
    
    for r in results:
        diff = r.get("dificultad", "unknown")
        stats[diff]["total"] += 1
        if r.get("correcto", False):
            stats[diff]["correctas"] += 1
    
    return {
        diff: {
            "total": data["total"],
            "correctas": data["correctas"],
            "tasa": round(data["correctas"] / data["total"] * 100, 1) if data["total"] > 0 else 0
        }
        for diff, data in stats.items()
    }


def get_category_stats(results: list) -> dict:
    """Get stats by category."""
    stats = defaultdict(lambda: {"total": 0, "correctas": 0})
    
    for r in results:
        cat = r.get("categoria", "unknown")
        stats[cat]["total"] += 1
        if r.get("correcto", False):
            stats[cat]["correctas"] += 1
    
    return {
        cat: {
            "total": data["total"],
            "correctas": data["correctas"],
            "tasa": round(data["correctas"] / data["total"] * 100, 1) if data["total"] > 0 else 0
        }
        for cat, data in stats.items()
    }


def generate_dashboard(output_path: Path = None):
    """Generate HTML dashboard from evaluation results.
    
    Args:
        output_path: Custom path to save dashboard. If None, uses default.
    """
    # Usar ruta personalizada si se proporciona, si no usar la padrão
    if output_path is None:
        output_path = OUTPUT_PATH
    
    results = load_results()
    metrics = calculate_metrics(results)
    diff_stats = get_difficulty_stats(results)
    cat_stats = get_category_stats(results)
    
    # Separate correct and incorrect
    incorrect_results = [r for r in results if not r.get("correcto", False)]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Evaluación - Chatbot RAG</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 30px; font-size: 14px; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
        .metric-card.success .metric-value {{ color: #28a745; }}
        .metric-card.error .metric-value {{ color: #dc3545; }}
        
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 20px; }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; color: #2c3e50; font-weight: 600; }}
        .status-correct {{ color: #28a745; font-weight: bold; }}
        .status-incorrect {{ color: #dc3545; font-weight: bold; }}
        
        .failed-section {{ border-left: 4px solid #dc3545; }}
        .failed-section h2 {{ color: #dc3545; }}
        
        .bar-chart {{ display: flex; flex-direction: column; gap: 15px; }}
        .bar-item {{ display: flex; align-items: center; gap: 15px; }}
        .bar-label {{ width: 80px; font-weight: 500; }}
        .bar-container {{ flex: 1; height: 30px; background: #ecf0f1; border-radius: 5px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: #28a745; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold; font-size: 12px; }}
        .bar-fill.low {{ background: #dc3545; }}
        
        .category-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .category-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .category-name {{ font-weight: 600; color: #2c3e50; margin-bottom: 5px; }}
        .category-stats {{ color: #7f8c8d; font-size: 13px; }}
        
        .method-tag {{ font-size: 11px; padding: 2px 6px; border-radius: 3px; background: #e9ecef; color: #495057; }}
        
        @media (max-width: 768px) {{ 
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .category-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dashboard de Evaluación</h1>
        <p class="subtitle">Última evaluación: {metrics.get('last_evaluation', 'N/A')}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics['total']}</div>
                <div class="metric-label">Total Pruebas</div>
            </div>
            <div class="metric-card success">
                <div class="metric-value">{metrics['correctas']}</div>
                <div class="metric-label">Correctas</div>
            </div>
            <div class="metric-card error">
                <div class="metric-value">{metrics['incorrectas']}</div>
                <div class="metric-label">Incorrectas</div>
            </div>
            <div class="metric-card success">
                <div class="metric-value">{metrics['tasa_exito']}%</div>
                <div class="metric-label">Tasa de Éxito</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Tiempos Promedio</h2>
            <div style="display: flex; gap: 40px;">
                <div><strong>{metrics['avg_retrieval']} ms</strong> - Retrieval</div>
                <div><strong>{metrics['avg_generation']} ms</strong> - Generación</div>
                <div><strong>{metrics['avg_latency']} ms</strong> - Total</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Éxito por Dificultad</h2>
            <div class="bar-chart">
                <div class="bar-item">
                    <span class="bar-label">Fácil</span>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {diff_stats.get('facil', {}).get('tasa', 0)}%">{diff_stats.get('facil', {}).get('tasa', 0)}%</div>
                    </div>
                </div>
                <div class="bar-item">
                    <span class="bar-label">Medio</span>
                    <div class="bar-container">
                        <div class="bar-fill {'low' if diff_stats.get('medio', {}).get('tasa', 0) < 50 else ''}" style="width: {diff_stats.get('medio', {}).get('tasa', 0)}%">{diff_stats.get('medio', {}).get('tasa', 0)}%</div>
                    </div>
                </div>
                <div class="bar-item">
                    <span class="bar-label">Difícil</span>
                    <div class="bar-container">
                        <div class="bar-fill {'low' if diff_stats.get('dificil', {}).get('tasa', 0) < 50 else ''}" style="width: {diff_stats.get('dificil', {}).get('tasa', 0)}%">{diff_stats.get('dificil', {}).get('tasa', 0)}%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Estadísticas por Categoría</h2>
            <div class="category-grid">
"""
    
    # Add category cards
    for cat, stat in sorted(cat_stats.items()):
        html += f"""                <div class="category-card">
                    <div class="category-name">{cat.replace('_', ' ').title()}</div>
                    <div class="category-stats">{stat['correctas']}/{stat['total']} ({stat['tasa']}%)</div>
                </div>
"""
    
    html += """            </div>
        </div>
        
        <div class="section">
            <h2>Resultados Detallados</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Pregunta</th>
                        <th>Respuesta</th>
                        <th>Estado</th>
                        <th>Método</th>
                        <th>Tiempo (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add table rows
    for r in results:
        status_class = "status-correct" if r.get("correcto", False) else "status-incorrect"
        status_text = "✓ Correcto" if r.get("correcto", False) else "✗ Incorrecto"
        respuesta = r.get("respuesta_chatbot", "")[:80] + "..." if len(r.get("respuesta_chatbot", "")) > 80 else r.get("respuesta_chatbot", "")
        
        html += f"""                    <tr>
                        <td>{r.get('test_id', 'N/A')}</td>
                        <td>{r.get('pregunta', '')[:60]}...</td>
                        <td>{respuesta}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td><span class="method-tag">{r.get('metodo_usado', 'N/A')}</span></td>
                        <td>{r.get('latency_ms', 0)}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
"""
    
    # Add failed questions section
    if incorrect_results:
        html += f"""        <div class="section failed-section">
            <h2>Preguntas Fallidas ({len(incorrect_results)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Pregunta</th>
                        <th>Respuesta Obtenida</th>
                        <th>Esperado</th>
                    </tr>
                </thead>
                <tbody>
"""
        for r in incorrect_results:
            html += f"""                    <tr>
                        <td>{r.get('test_id', 'N/A')}</td>
                        <td>{r.get('pregunta', '')}</td>
                        <td>{r.get('respuesta_chatbot', '')[:100]}</td>
                        <td>{r.get('respuesta_esperada', '')}</td>
                    </tr>
"""
        
        html += """                </tbody>
            </table>
        </div>
"""
    
    html += """    </div>
</body>
</html>"""
    
    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    absolute_path = output_path.resolve()
    print(f"✅ Dashboard guardado en: {absolute_path}")
    print(f"✅ Existe archivo: {output_path.exists()}")
    return output_path


if __name__ == "__main__":
    generate_dashboard()