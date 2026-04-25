"""
Generate User Dashboard - Analiza interacciones reales de usuarios con el chatbot.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any
import re


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


def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calcula el percentil de una lista de valores."""
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def extract_keywords(questions: List[str], top_n: int = 5) -> List[Dict[str, int]]:
    """Extrae palabras clave más frecuentes de las preguntas."""
    stopwords = {
        "de", "la", "el", "en", "y", "a", "que", "es", "por", "con",
        "los", "las", "un", "una", "se", "su", "para", "mi", "me", "como",
        "qué", "cómo", "cuándo", "dónde", "cuál", "cuáles", "cuánto", "cuántos",
        "está", "son", "tiene", "tienen", "hacer", "puedo", "puede", "sí",
        "no", "pero", "del", "al", "le", "les", "esto", "esta", "este",
        "todo", "toda", "todos", "todas", "muy", "más", "menos", "tan",
        "bien", "mal", "solo", "sólo", "ya", "aún", "todavía",
        "cual", "forma", "manera", "razon", "saber", "haber", "estar", "ser",
        "hacer", "tener", "decir", "ir", "ver", "dar", "algo", "alguien",
        "nada", "nadie", "cada", "poco", "mucho", "otro", "mismo"
    }
    
    todas_palabras = []
    
    for pregunta in questions:
        if not pregunta:
            continue
        # Limpiar: minúsculas, eliminar signos de puntuación ynormalizar tildes
        texto_limpio = pregunta.lower()
        texto_limpio = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto_limpio)
        # Normalizar tildes
        texto_limpio = texto_limpio.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        palabras = texto_limpio.split()
        # Filtrar stop words y palabras muy cortas
        palabras_filtradas = [p for p in palabras if p not in stopwords and len(p) > 3]
        todas_palabras.extend(palabras_filtradas)
    
    counter = Counter(todas_palabras)
    return [{"palabra": w, "conteo": c} for w, c in counter.most_common(top_n)]


def calculate_metrics(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula las métricas de las interacciones."""
    if not interactions:
        return {
            "total_interacciones": 0,
            "tiempo_promedio_ms": 0,
            "tiempo_p50_ms": 0,
            "tiempo_p95_ms": 0,
            "tasa_rag": 0,
            "tasa_no_encontrado": 0,
            "confianza_promedio": 0,
            "usuarios_unicos": 0,
            "palabras_clave": [],
            "fuentes_top": [],
            "distribucion_dia": {},
            "distribucion_hora": {}
        }
    
    total = len(interactions)
    
    # Tiempos
    tiempos = [i.get("tiempo_total_ms", 0) for i in interactions if i.get("tiempo_total_ms")]
    tiempo_promedio = sum(tiempos) / len(tiempos) if tiempos else 0
    tiempo_p50 = calculate_percentile(tiempos, 50)
    tiempo_p95 = calculate_percentile(tiempos, 95)
    
    # Tasa RAG
    count_rag = sum(1 for i in interactions if i.get("es_rag", False))
    tasa_rag = (count_rag / total * 100) if total > 0 else 0
    
    # Tasa "No encontré información"
    no_encontrado = sum(
        1 for i in interactions 
        if i.get("respuesta", "") and "no encontré" in i.get("respuesta", "").lower()
    )
    tasa_no_encontrado = (no_encontrado / total * 100) if total > 0 else 0
    
    # Confianza promedio
    confidencias = [i.get("confianza", 0) for i in interactions if i.get("confianza")]
    confianza_promedio = sum(confidencias) / len(confidencias) if confidencias else 0
    
    # Usuarios únicos
    session_ids = set(i.get("session_id", "") for i in interactions if i.get("session_id"))
    usuarios_unicos = len(session_ids)
    
    # Palabras clave
    preguntas = [i.get("pregunta", "") for i in interactions if i.get("pregunta")]
    palabras_clave = extract_keywords(preguntas, 5)
    
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
        "tiempo_p50_ms": round(tiempo_p50, 2),
        "tiempo_p95_ms": round(tiempo_p95, 2),
        "tasa_rag": round(tasa_rag, 2),
        "tasa_no_encontrado": round(tasa_no_encontrado, 2),
        "confianza_promedio": round(confianza_promedio, 4),
        "usuarios_unicos": usuarios_unicos,
        "palabras_clave": palabras_clave,
        "fuentes_top": fuentes_top,
        "distribucion_dia": distribucion_dia,
        "distribucion_hora": distribucion_hora
    }


def formatear_fecha(timestamp_str):
    """Formatea timestamp ISO a DD/MM/YYYY HH:MM."""
    if not timestamp_str:
        return "-"
    try:
        fecha_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return fecha_obj.strftime("%d/%m/%Y %H:%M")
    except:
        return timestamp_str[:16] if len(timestamp_str) > 16 else "-"


def generate_dashboard_html(metrics: Dict[str, Any], interactions: List[Dict[str, Any]] = []) -> str:
    """Genera el HTML del dashboard interactivo."""
    import html
    
    if interactions is None:
        interactions = []
    
    # Generar tabla de historial reciente (últimas 10)
    historial_html = ""
    for i in interactions[-10:]:
        ts = formatear_fecha(i.get("timestamp", ""))
        pregunta = html.escape(i.get("pregunta", "-")[:50])
        if len(i.get("pregunta", "")) > 50:
            pregunta += "..."
        respuesta = html.escape(i.get("respuesta", "-")[:60])
        if len(i.get("respuesta", "")) > 60:
            respuesta += "..."
        tiempo = f"{i.get('tiempo_total_ms', 0):.0f}ms" if i.get("tiempo_total_ms") else "-"
        rag = "Sí" if i.get("es_rag", False) else "No"
        historial_html += f"<tr><td>{ts}</td><td>{pregunta}</td><td>{respuesta}</td><td>{tiempo}</td><td>{rag}</td></tr>"
    
    if not historial_html:
        historial_html = '<tr><td colspan="5" class="no-data">No hay interacciones registradas</td></tr>'
    
    # Preparar datos para gráficos
    horas_labels = list(range(24))
    horas_values = [metrics.get("distribucion_hora", {}).get(h, 0) for h in horas_labels]
    
    dias_labels = list(metrics.get("distribucion_dia", {}).keys())[-7:]  # últimos 7 días
    dias_values = [metrics.get("distribucion_dia", {}).get(d, 0) for d in dias_labels]
    
    # Palabras clave
    palabras_html = ""
    for p in metrics.get("palabras_clave", []):
        palabras_html += f'<span class="badge">{html.escape(p["palabra"])} ({p["conteo"]})</span>'
    
    if not palabras_html:
        palabras_html = '<div class="no-data">No hay palabras clave</div>'
    
    # Fuentes más usadas
    fuentes_html = ""
    for f in metrics.get("fuentes_top", []):
        fuentes_html += f'<span class="badge">{html.escape(f["fuente"])} ({f["conteo"]})</span>'
    
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
        /* Paleta de colores del chatbot */
        :root {{
            --azul-principal: #2c3e50;
            --azul-secundario: #3498db;
            --verdeclaro: #2ecc71;
            --rojosoft: #e74c3c;
            --grisclaro: #ecf0f1;
            --gristexto: #7f8c8d;
            --blanco: #ffffff;
            --sombra: rgba(0,0,0,0.1);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--grisclaro); padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--azul-principal); margin-bottom: 8px; font-size: 1.8rem; font-weight: 700; }}
        .subtitle {{ color: var(--gristexto); margin-bottom: 20px; font-size: 14px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; flex-wrap: wrap; gap: 16px; }}
        .btn {{ background: var(--azul-secundario); color: var(--blanco); padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; transition: background 0.2s; }}
        .btn:hover {{ background: var(--azul-principal); }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: var(--blanco); border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px var(--sombra); }}
        .card-label {{ font-size: 11px; color: var(--gristexto); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
        .card-value {{ font-size: 28px; font-weight: 700; color: var(--azul-principal); }}
        .card-value.positivo {{ color: var(--verdeclaro); }}
        .card-value.negativo {{ color: var(--rojosoft); }}
        .card-sub {{ font-size: 12px; color: var(--gristexto); margin-top: 4px; }}
        
        .chart-container {{ background: var(--blanco); border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px var(--sombra); margin-bottom: 24px; }}
        .chart-title {{ font-size: 16px; font-weight: 600; color: var(--azul-principal); margin-bottom: 16px; }}
        
        .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .badge {{ background: var(--azul-secundario); color: var(--blanco); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; }}
        
        .sources-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--blanco); border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px var(--sombra); }}
        th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--grisclaro); }}
        th {{ background: var(--azul-principal); color: var(--blanco); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ color: var(--azul-principal); }}
        tr:nth-child(even) {{ background: var(--grisclaro); }}
        tr:last-child td {{ border-bottom: none; }}
        
        .no-data {{ color: var(--gristexto); font-style: italic; text-align: center; padding: 20px; }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            h1 {{ font-size: 1.5rem; }}
            .header {{ flex-direction: column; align-items: flex-start; }}
            .btn {{ width: 100%; text-align: center; }}
            .grid {{ grid-template-columns: 1fr 1fr; }}
            table {{ font-size: 12px; }}
            th, td {{ padding: 8px 6px; }}
        }}
        @media (max-width: 480px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .sources-grid {{ grid-template-columns: 1fr; }}
        }}
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
                <div class="card-sub">Conversaciones</div>
            </div>
            <div class="card">
                <div class="card-label">Tiempo Promedio</div>
                <div class="card-value">{metrics.get("tiempo_promedio_ms", 0):.0f}ms</div>
                <div class="card-sub">Media de respuesta</div>
            </div>
            <div class="card">
                <div class="card-label">Tiempo P50</div>
                <div class="card-value">{metrics.get("tiempo_p50_ms", 0):.0f}ms</div>
                <div class="card-sub">Mediana</div>
            </div>
            <div class="card">
                <div class="card-label">Tiempo P95</div>
                <div class="card-value">{metrics.get("tiempo_p95_ms", 0):.0f}ms</div>
                <div class="card-sub">Percentil 95</div>
            </div>
            <div class="card">
                <div class="card-label">Tasa RAG</div>
                <div class="card-value positivo">{metrics.get("tasa_rag", 0):.1f}%</div>
                <div class="card-sub">Con RAG</div>
            </div>
            <div class="card">
                <div class="card-label">Sin Info</div>
                <div class="card-value">{metrics.get("tasa_no_encontrado", 0):.1f}%</div>
                <div class="card-sub">No disponible</div>
            </div>
            <div class="card">
                <div class="card-label">Confianza</div>
                <div class="card-value">{metrics.get("confianza_promedio", 0)*100:.1f}%</div>
                <div class="card-sub">Media del sistema</div>
            </div>
            <div class="card">
                <div class="card-label">Usuarios</div>
                <div class="card-value">{metrics.get("usuarios_unicos", 0):,}</div>
                <div class="card-sub">Sesiones únicas</div>
            </div>
        </div>
        
        <div class="sources-grid">
            <div class="chart-container">
                <div class="chart-title">Fuentes Más Usadas</div>
                <div class="badges">
                    {fuentes_html}
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Palabras Clave (Top 5)</div>
                <div class="badges">
                    {palabras_html}
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Distribución por Hora (24h)</div>
            <canvas id="chartHora"></canvas>
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
                <tbody>
                    {historial_html}
                </tbody>
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
        
        async function refresh() {{
            const btn = document.querySelector('.btn');
            btn.textContent = 'Actualizando...';
            await fetch('/user-dashboard/refresh');
            location.reload();
        }}
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
    html = generate_dashboard_html(metrics, interactions)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"[OK] Dashboard generado: {output_path}")
    print(f"     Interacciones procesadas: {metrics['total_interacciones']}")
    return output_path


if __name__ == "__main__":
    generate_user_dashboard()