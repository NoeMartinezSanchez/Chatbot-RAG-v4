"""
Generate User Dashboard - Analiza interacciones reales de usuarios con el chatbot.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any


def read_interactions(log_path: str = "/data/user_interactions.jsonl") -> List[Dict[str, Any]]:
    """Lee las interacciones desde el archivo JSONL."""
    path = Path(log_path)
    if not path.exists():
        return []
    
    interactions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                interactions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return interactions


def calculate_metrics(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula las métricas de las interacciones."""
    if not interactions:
        return {
            "total_interacciones": 0,
            "tiempo_promedio_ms": 0,
            "tasa_rag": 0,
            "confianza_promedio": 0,
            "fuentes_top": [],
            "distribucion_dia": {},
            "distribucion_hora": {}
        }
    
    total = len(interactions)
    
    # Tiempos
    tiempos = [i.get("tiempo_total_ms", 0) for i in interactions if i.get("tiempo_total_ms")]
    tiempo_promedio = sum(tiempos) / len(tiempos) if tiempos else 0
    
    # Tasa RAG
    count_rag = sum(1 for i in interactions if i.get("es_rag", False))
    tasa_rag = (count_rag / total * 100) if total > 0 else 0
    
    # Confianza promedio
    confidencias = [i.get("confianza", 0) for i in interactions if i.get("confianza")]
    confianza_promedio = sum(confidencias) / len(confidencias) if confidencias else 0
    
    # Fuentes más usadas
    todas_fuentes = []
    for i in interactions:
        fuentes = i.get("fuentes_usadas", [])
        if fuentes:
            todas_fuentes.extend(fuentes)
    
    fuentes_counter = Counter(todas_fuentes)
    fuentes_top = [{"fuente": f, "conteo": c} for f, c in fuentes_counter.most_common(3)]
    
    # Distribución por día y hora
    distribucion_dia = defaultdict(int)
    distribucion_hora = defaultdict(int)
    
    for i in interactions:
        ts = i.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dia = dt.strftime("%Y-%m-%d")
                hora = dt.hour
                distribucion_dia[dia] += 1
                distribucion_hora[hora] += 1
            except (ValueError, TypeError):
                continue
    
    distribucion_dia = dict(sorted(distribucion_dia.items()))
    distribucion_hora = dict(sorted(distribucion_hora.items()))
    
    return {
        "total_interacciones": total,
        "tiempo_promedio_ms": round(tiempo_promedio, 2),
        "tasa_rag": round(tasa_rag, 2),
        "confianza_promedio": round(confianza_promedio, 4),
        "fuentes_top": fuentes_top,
        "distribucion_dia": distribucion_dia,
        "distribucion_hora": distribucion_hora
    }


def generate_dashboard_html(metrics: Dict[str, Any]) -> str:
    """Genera el HTML del dashboard interactivo."""
    import html
    
    # Preparar datos para gráficos
    horas_labels = list(range(24))
    horas_values = [metrics.get("distribucion_hora", {}).get(h, 0) for h in horas_labels]
    
    dias_labels = list(metrics.get("distribucion_dia", {}).keys())[-7:]  # últimos 7 días
    dias_values = [metrics.get("distribucion_dia", {}).get(d, 0) for d in dias_labels]
    
    fuentes_html = ""
    for f in metrics.get("fuentes_top", []):
        fuentes_html += f'<div class=" fuente-item"><span>{html.escape(f["fuente"])}</span><span class="count">{f["conteo"]}</span></div>'
    
    if not fuentes_html:
        fuentes_html = '<div class="no-data">No hay fuentes registradas</div>'
    
    return f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Usuarios - Prepa en Línea SEP</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 5px; font-size: 1.8rem; }}
        .subtitle {{ color: #666; margin-bottom: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .btn {{ background: #4f46e5; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
        .btn:hover {{ background: #4338ca; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card-label {{ font-size: 13px; color: #6b7280; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 32px; font-weight: 700; color: #1a1a2e; }}
        .card-sub {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
        
        .chart-container {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        .chart-title {{ font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 16px; }}
        
        .fuentes-list {{ display: flex; flex-direction: column; gap: 8px; }}
        .fuente-item {{ display: flex; justify-content: space-between; padding: 10px 12px; background: #f9fafb; border-radius: 6px; font-size: 14px; }}
        .fuente-item .count {{ font-weight: 600; color: #4f46e5; }}
        .no-data {{ color: #9ca3af; font-style: italic; }}
        
        .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 768px) {{ .tables {{ grid-template-columns: 1fr; }} }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; color: #374151; }}
        td {{ color: #4b5563; }}
        tr:last-child td {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Dashboard de Interacciones Reales</h1>
                <p class="subtitle">Métricas de uso del chatbot por estudiantes reales</p>
            </div>
            <button class="btn" onclick="refresh()">🔄 Actualizar</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-label">Total Interacciones</div>
                <div class="card-value">{metrics.get("total_interacciones", 0):,}</div>
                <div class="card-sub">Conversaciones registradas</div>
            </div>
            <div class="card">
                <div class="card-label">Tiempo Promedio</div>
                <div class="card-value">{metrics.get("tiempo_promedio_ms", 0):.0f}ms</div>
                <div class="card-sub">Tiempo de respuesta medio</div>
            </div>
            <div class="card">
                <div class="card-label">Tasa RAG</div>
                <div class="card-value">{metrics.get("tasa_rag", 0):.1f}%</div>
                <div class="card-sub">Consultas respondidas con RAG</div>
            </div>
            <div class="card">
                <div class="card-label">Confianza Promedio</div>
                <div class="card-value">{metrics.get("confianza_promedio", 0)*100:.1f}%</div>
                <div class="card-sub">Confianza media del sistema</div>
            </div>
        </div>
        
        <div class="tables">
            <div class="chart-container">
                <div class="chart-title">Fuentes Más Usadas</div>
                <div class="fuentes-list">
                    {fuentes_html}
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Distribución por Hora (24h)</div>
                <canvas id="chartHora"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Interacciones por Día</div>
            <canvas id="chartDia"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Historial Reciente</div>
            <table id="tablaHistorial">
                <thead>
                    <tr><th>Fecha</th><th>Pregunta</th><th>Respuesta</th><th>Tiempo</th><th>RAG</th></tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    
    <script>
        const horasLabels = {horas_labels};
        const horasValues = {horas_values};
        const diasLabels = {dias_labels};
        const diasValues = {dias_values};
        
        new Chart(document.getElementById('chartHora'), {{
            type: 'bar',
            data: {{
                labels: horasLabels.map(h => h + ':00'),
                datasets: [{{
                    label: 'Consultas',
                    data: horasValues,
                    backgroundColor: '#4f46e5',
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});
        
        new Chart(document.getElementById('chartDia'), {{
            type: 'line',
            data: {{
                labels: diasLabels,
                datasets: [{{
                    label: 'Interacciones',
                    data: diasValues,
                    borderColor: '#4f46e5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }}
                }}
            }}
        }});
        
        async function refresh() {{
            const btn = document.querySelector('.btn');
            btn.textContent = 'Actualizando...';
            await fetch('/user-dashboard/refresh');
            location.reload();
        }}
        
        async function loadRecientes() {{
            try {{
                const res = await fetch('/data/user_interactions.jsonl');
                const text = await res.text();
                const lines = text.trim().split('\\n').filter(l => l).slice(-10);
                const tbody = document.querySelector('#tablaHistorial tbody');
                tbody.innerHTML = lines.reverse().map(line => {{
                    const d = JSON.parse(line);
                    const fecha = new Date(d.timestamp).toLocaleString('es-MX', {{ dateStyle: 'short', timeStyle: 'short' }});
                    const tiempo = d.tiempo_total_ms ? d.tiempo_total_ms.toFixed(0) + 'ms' : '-';
                    const rag = d.es_rag ? '✓' : '-';
                    return `<tr>
                        <td>${{fecha}}</td>
                        <td>${{d.pregunta ? d.pregunta.substring(0, 50) + '...' : '-'}}</td>
                        <td>${{d.respuesta ? d.respuesta.substring(0, 40) + '...' : '-'}}</td>
                        <td>${{tiempo}}</td>
                        <td>${{rag}}</td>
                    </tr>`;
                }}).join('');
            }} catch(e) {{ console.error(e); }}
        }}
        
        loadRecientes();
    </script>
</body>
</html>'''


def generate_user_dashboard(
    log_path: str = "/data/user_interactions.jsonl",
    output_path: str = "/data/user_dashboard.html"
) -> str:
    """Genera el dashboard de interacciones de usuarios."""
    interactions = read_interactions(log_path)
    metrics = calculate_metrics(interactions)
    html = generate_dashboard_html(metrics)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Dashboard generado: {output_path}")
    print(f"   Interacciones procesadas: {metrics['total_interacciones']}")
    return output_path


if __name__ == "__main__":
    generate_user_dashboard()