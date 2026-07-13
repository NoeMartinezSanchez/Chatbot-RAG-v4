"""
Generate User Dashboard - Analiza interacciones reales de usuarios con el chatbot.
"""
import json
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


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


def get_token_stats() -> Dict[str, Any]:
    """Lee el consumo de tokens desde token_usage.json y token_usage_per_query.jsonl."""
    stats = {
        "tokens_hoy": 0,
        "limite": 100000,
        "porcentaje": 0,
        "promedio_tokens": 0,
        "total_consultas": 0,
    }

    token_file = "token_usage.json"
    if os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                data = json.load(f)
                today = datetime.now().strftime("%Y-%m-%d")
                if data.get("date") == today:
                    stats["tokens_hoy"] = data.get("tokens", 0)
        except:
            pass

    stats["porcentaje"] = round((stats["tokens_hoy"] / stats["limite"]) * 100, 1)

    # Calcular promedio desde token_usage_per_query.jsonl
    per_query_file = "token_usage_per_query.jsonl"
    token_list = []
    if os.path.exists(per_query_file):
        try:
            with open(per_query_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_date = entry.get("timestamp", "")[:10]
                        if entry_date == datetime.now().strftime("%Y-%m-%d"):
                            token_list.append(entry.get("tokens", 0))
                    except:
                        continue
        except:
            pass

    stats["total_consultas"] = len(token_list)
    stats["promedio_tokens"] = round(sum(token_list) / len(token_list)) if token_list else 0

    return stats


def is_useful_response(response_text: str) -> bool:
    useless_patterns = [
        "no encontré información",
        "no encontré información específica",
        "problema procesando tu pregunta",
        "tuve un problema",
        "intenta de nuevo",
        "no sé",
        "no tengo información"
    ]
    response_lower = response_text.lower()
    return not any(pattern in response_lower for pattern in useless_patterns)


def calculate_tokens_por_hora() -> Dict[int, int]:
    """Lee token_usage_per_query.jsonl y agrupa tokens por hora."""
    tokens_por_hora: Dict[int, int] = defaultdict(int)
    file_path = "token_usage_per_query.jsonl"
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ts = entry.get("timestamp", "")
                    if ts:
                        dt = datetime.fromisoformat(ts)
                        tokens_por_hora[dt.hour] += entry.get("tokens", 0)
                except:
                    continue
    except:
        pass
    return dict(tokens_por_hora)


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
            "distribucion_hora": {},
            "tasa_exito": 0,
            "respuestas_utiles": 0,
            "respuestas_no_utiles": 0,
            "tokens_por_hora": {},
            "max_tokens_por_hora": 0,
            "avg_tokens_por_hora": 0
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
    fuentes_top = [{"fuente": f, "conteo": c} for f, c in fuentes_counter.most_common(10)]
    
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
    
    # Tasa de éxito real (solo RAG)
    rag_interactions = [i for i in interactions if i.get("es_rag", False)]
    total_rag = len(rag_interactions)
    respuestas_utiles = sum(1 for i in rag_interactions if is_useful_response(i.get("respuesta", "")))
    respuestas_no_utiles = total_rag - respuestas_utiles
    tasa_exito = (respuestas_utiles / total_rag * 100) if total_rag > 0 else 0
    
    # Tokens por hora
    tokens_por_hora = calculate_tokens_por_hora()
    max_tokens_por_hora = max(tokens_por_hora.values()) if tokens_por_hora else 0
    avg_tokens_por_hora = round(sum(tokens_por_hora.values()) / len(tokens_por_hora)) if tokens_por_hora else 0
    
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
        "distribucion_hora": distribucion_hora,
        "tasa_exito": round(tasa_exito, 1),
        "respuestas_utiles": respuestas_utiles,
        "respuestas_no_utiles": respuestas_no_utiles,
        "tokens_por_hora": dict(tokens_por_hora),
        "max_tokens_por_hora": max_tokens_por_hora,
        "avg_tokens_por_hora": avg_tokens_por_hora
    }


def calculate_sla_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    THRESHOLD_RESPONSE_GREEN = 2000
    THRESHOLD_RESPONSE_YELLOW = 4000
    THRESHOLD_SUCCESS_GREEN = 90.0
    THRESHOLD_SUCCESS_YELLOW = 70.0
    THRESHOLD_NOTFOUND_GREEN = 10.0
    THRESHOLD_NOTFOUND_YELLOW = 20.0
    THRESHOLD_TOKENS_GREEN = 70.0
    THRESHOLD_TOKENS_YELLOW = 85.0

    def status(value, green, yellow):
        if value < green:
            return "green"
        elif value < yellow:
            return "yellow"
        else:
            return "red"

    p95 = metrics.get("tiempo_p95_ms", 0)
    success_rate = metrics.get("tasa_exito", 0)
    not_found = metrics.get("tasa_no_encontrado", 0)
    token_pct = metrics.get("token_porcentaje", 0)

    sla_response = {
        "label": "Tiempo de Respuesta P95",
        "value": f"{p95:.0f}ms",
        "status": status(p95, THRESHOLD_RESPONSE_GREEN, THRESHOLD_RESPONSE_YELLOW),
        "thresholds": f"🟢 < {THRESHOLD_RESPONSE_GREEN}ms | 🟡 < {THRESHOLD_RESPONSE_YELLOW}ms | 🔴 ≥ {THRESHOLD_RESPONSE_YELLOW}ms"
    }
    sla_success = {
        "label": "Tasa de Éxito",
        "value": f"{success_rate:.1f}%",
        "status": status(100 - success_rate, 100 - THRESHOLD_SUCCESS_GREEN, 100 - THRESHOLD_SUCCESS_YELLOW) if success_rate > 0 else "red",
        "thresholds": f"🟢 > {THRESHOLD_SUCCESS_GREEN}% | 🟡 > {THRESHOLD_SUCCESS_YELLOW}% | 🔴 ≤ {THRESHOLD_SUCCESS_YELLOW}%"
    }
    sla_notfound = {
        "label": "Tasa No Encontrado",
        "value": f"{not_found:.1f}%",
        "status": status(not_found, THRESHOLD_NOTFOUND_GREEN, THRESHOLD_NOTFOUND_YELLOW),
        "thresholds": f"🟢 < {THRESHOLD_NOTFOUND_GREEN}% | 🟡 < {THRESHOLD_NOTFOUND_YELLOW}% | 🔴 ≥ {THRESHOLD_NOTFOUND_YELLOW}%"
    }
    sla_tokens = {
        "label": "Uso de Tokens",
        "value": f"{token_pct:.1f}%",
        "status": status(token_pct, THRESHOLD_TOKENS_GREEN, THRESHOLD_TOKENS_YELLOW),
        "thresholds": f"🟢 < {THRESHOLD_TOKENS_GREEN}% | 🟡 < {THRESHOLD_TOKENS_YELLOW}% | 🔴 ≥ {THRESHOLD_TOKENS_YELLOW}%"
    }
    items = [sla_response, sla_success, sla_notfound, sla_tokens]
    status_map = {"green": 3, "yellow": 2, "red": 1}
    total_score = sum(status_map[i["status"]] for i in items)
    max_score = len(items) * 3
    overall_pct = (total_score / max_score) * 100
    if overall_pct >= 80:
        overall_status = "green"
    elif overall_pct >= 50:
        overall_status = "yellow"
    else:
        overall_status = "red"

    return {
        "items": items,
        "overall_status": overall_status,
        "overall_pct": round(overall_pct, 0)
    }


def calculate_roi(metrics: Dict[str, Any]) -> Dict[str, Any]:
    HUMAN_AGENT_COST_PER_HOUR = 15.0
    HUMAN_AVG_HANDLING_MINUTES = 10.0
    WORKING_DAYS_PER_MONTH = 22

    total_queries = metrics.get("total_interacciones", 0)
    bot_avg_ms = metrics.get("tiempo_promedio_ms", 0)
    bot_avg_minutes = bot_avg_ms / 60000.0 if bot_avg_ms > 0 else 0.5

    human_minutes = total_queries * HUMAN_AVG_HANDLING_MINUTES
    bot_minutes = total_queries * bot_avg_minutes
    time_saved_minutes = human_minutes - bot_minutes
    time_saved_hours = time_saved_minutes / 60.0

    cost_human = (human_minutes / 60.0) * HUMAN_AGENT_COST_PER_HOUR
    cost_bot = 0.0
    total_savings = cost_human - cost_bot

    active_days = max(1, (datetime.now() - datetime.strptime("2026-01-01", "%Y-%m-%d")).days)
    monthly_savings = (total_savings / active_days) * WORKING_DAYS_PER_MONTH
    yearly_savings = monthly_savings * 12

    savings_per_query = HUMAN_AVG_HANDLING_MINUTES * (HUMAN_AGENT_COST_PER_HOUR / 60.0)

    return {
        "total_queries": total_queries,
        "bot_avg_minutes": round(bot_avg_minutes, 2),
        "human_avg_minutes": HUMAN_AVG_HANDLING_MINUTES,
        "time_saved_minutes": round(time_saved_minutes, 0),
        "time_saved_hours": round(time_saved_hours, 1),
        "total_savings": round(total_savings, 2),
        "monthly_savings": round(monthly_savings, 2),
        "yearly_savings": round(yearly_savings, 2),
        "savings_per_query": round(savings_per_query, 2),
        "human_cost_rate": HUMAN_AGENT_COST_PER_HOUR
    }


def send_telegram_alert(message: str) -> bool:
    import time
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        logger.debug("TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados")
        return False
    import requests
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    max_retries = 3
    backoff = [2, 4, 8]
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                logger.debug(f"Alerta Telegram enviada (intento {attempt + 1})")
                return True
            else:
                logger.debug(f"Telegram respondió {resp.status_code} (intento {attempt + 1})")
        except Exception as e:
            logger.debug(f"Fallo conexión Telegram (intento {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff[attempt])
    logger.debug("Alerta Telegram no enviada tras reintentos")
    return False


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
    
    # Leer tokens por consulta para la tabla
    token_map = {}
    per_query_file = "token_usage_per_query.jsonl"
    if os.path.exists(per_query_file):
        try:
            with open(per_query_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_date = entry.get("timestamp", "")[:10]
                        if entry_date == datetime.now().strftime("%Y-%m-%d"):
                            token_map[entry.get("timestamp", "")] = entry.get("tokens", 0)
                    except:
                        continue
        except:
            pass
    
    token_keys = sorted(token_map.keys())
    
    # Generar tabla de historial reciente (últimas 10)
    historial_html = ""
    for idx, i in enumerate(reversed(interactions[-10:])):
        ts = formatear_fecha(i.get("timestamp", ""))
        pregunta = html.escape(i.get("pregunta", "-")[:50])
        if len(i.get("pregunta", "")) > 50:
            pregunta += "..."
        respuesta = html.escape(i.get("respuesta", "-")[:60])
        if len(i.get("respuesta", "")) > 60:
            respuesta += "..."
        tiempo = f"{i.get('tiempo_total_ms', 0):.0f}ms" if i.get("tiempo_total_ms") else "-"
        tokens_val = i.get("tokens_used", token_map.get(token_keys[-(idx+1)] if idx < len(token_keys) else "", "-"))
        tokens_cell = f"{tokens_val:,}" if isinstance(tokens_val, int) else "-"
        rag = "Sí" if i.get("es_rag", False) else "No"
        historial_html += f"<tr><td>{ts}</td><td>{pregunta}</td><td>{respuesta}</td><td>{tiempo}</td><td>{tokens_cell}</td><td>{rag}</td></tr>"
    
    if not historial_html:
        historial_html = '<tr><td colspan="6" class="no-data">No hay interacciones registradas</td></tr>'

    # Token stats for cards
    token_stats = get_token_stats()
    tokens_hoy = token_stats["tokens_hoy"]
    limite = token_stats["limite"]
    porcentaje = token_stats["porcentaje"]
    prom_tokens = token_stats["promedio_tokens"]
    
    # Tasa de éxito
    tasa_exito = metrics.get("tasa_exito", 0)
    respuestas_utiles = metrics.get("respuestas_utiles", 0)
    respuestas_no_utiles = metrics.get("respuestas_no_utiles", 0)
    
    # SLA y ROI
    metrics_with_tokens = dict(metrics)
    metrics_with_tokens["token_porcentaje"] = porcentaje
    sla_data = calculate_sla_metrics(metrics_with_tokens)
    roi_data = calculate_roi(metrics)
    
    # Pre-format SLA status icons
    sla_status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    sla_overall_icon = sla_status_icon.get(sla_data["overall_status"], "⚪")
    sla_items_html = ""
    for item in sla_data["items"]:
        icon = sla_status_icon.get(item["status"], "⚪")
        sla_items_html += f"""<div class="sla-item {item["status"]}">
            <div class="sla-indicator {item["status"]}">{icon}</div>
            <div class="sla-info">
                <div class="sla-label">{item["label"]}</div>
                <div class="sla-value">{item["value"]}</div>
            </div>
        </div>"""
    
    # Pre-format ROI cards
    roi_total_savings = roi_data["total_savings"]
    roi_monthly = roi_data["monthly_savings"]
    roi_yearly = roi_data["yearly_savings"]
    roi_time_saved = roi_data["time_saved_hours"]
    roi_per_query = roi_data["savings_per_query"]
    
    # ASCII chart de tokens por hora
    tokens_por_hora = metrics.get("tokens_por_hora", {})
    max_tph = metrics.get("max_tokens_por_hora", 0)
    avg_tph = metrics.get("avg_tokens_por_hora", 0)
    ascii_chart_lines = ["🕐 Consumo de tokens por hora (últimas 24h)", ""]
    for h in range(24):
        tokens_h = tokens_por_hora.get(h, 0)
        if tokens_h > 0:
            bar_len = max(1, int((tokens_h / max_tph) * 20)) if max_tph > 0 else 1
            bar = "█" * bar_len
            ascii_chart_lines.append(f"{h:02d}:00 │ {bar} {tokens_h:,}")
        else:
            ascii_chart_lines.append(f"{h:02d}:00 │ · sin consumo")
    grafico_tokens = "\n".join(ascii_chart_lines)

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
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
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
        
        .progress-bar {{ width: 100%; height: 6px; background: var(--grisclaro); border-radius: 3px; margin-top: 8px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: var(--verdeclaro); border-radius: 3px; transition: width 0.3s ease; }}
        .progress-fill.warning {{ background: #f39c12; }}
        .progress-fill.danger {{ background: var(--rojosoft); }}
        .progress-bar.success .progress-fill.success {{ background: linear-gradient(90deg, #28a745, #20c997); }}
        
        .ascii-chart {{ background: #f8f9fa; padding: 16px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6; overflow-x: auto; white-space: pre; margin: 10px 0; border: 1px solid #e9ecef; }}
        
        /* Pestañas */
        .dashboard-tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; flex-wrap: wrap; }}
        .tab-btn {{ background: none; border: none; padding: 10px 20px; font-size: 14px; cursor: pointer; border-radius: 8px 8px 0 0; transition: all 0.2s; font-weight: 500; color: var(--gristexto); }}
        .tab-btn:hover {{ background: var(--grisclaro); }}
        .tab-btn.active {{ background: var(--azul-secundario); color: var(--blanco); }}
        
        /* Logs */
        .logs-controls {{ display: flex; gap: 10px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
        .logs-controls select, .logs-controls button {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #ddd; font-size: 13px; background: var(--blanco); }}
        .log-count {{ margin-left: auto; font-size: 13px; color: var(--gristexto); }}
        .logs-container {{ max-height: 600px; overflow-y: auto; }}
        .log-entry {{ background: #f8f9fa; border-left: 4px solid #6c757d; padding: 12px; margin-bottom: 8px; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 12px; }}
        .log-entry.error {{ border-left-color: #dc3545; background: #fff5f5; }}
        .log-entry.warning {{ border-left-color: #ffc107; background: #fffbf0; }}
        .log-entry.info {{ border-left-color: #17a2b8; background: #f0f9ff; }}
        .log-header {{ display: flex; gap: 15px; margin-bottom: 6px; font-size: 11px; flex-wrap: wrap; }}
        .log-level {{ font-weight: 700; }}
        .log-time {{ color: var(--gristexto); }}
        .log-module {{ color: #28a745; }}
        .log-message {{ font-size: 12px; white-space: pre-wrap; word-break: break-word; }}
        .log-line {{ font-size: 11px; color: #999; margin-top: 4px; }}
        .log-empty, .log-error, .log-loading {{ text-align: center; padding: 40px; color: var(--gristexto); }}
        .log-error {{ color: #dc3545; }}
        
        /* SLA */
        .sla-overview {{ display: flex; gap: 24px; flex-wrap: wrap; }}
        .sla-overall {{ flex: 1; min-width: 220px; text-align: center; padding: 24px; border-radius: 12px; }}
        .sla-overall.green {{ background: #e8f8e8; border: 2px solid #2ecc71; }}
        .sla-overall.yellow {{ background: #fff8e1; border: 2px solid #f39c12; }}
        .sla-overall.red {{ background: #fde8e8; border: 2px solid #e74c3c; }}
        .sla-overall-icon {{ font-size: 48px; margin-bottom: 8px; }}
        .sla-overall-text {{ font-size: 13px; color: var(--gristexto); margin-bottom: 4px; }}
        .sla-overall-value {{ font-size: 28px; font-weight: 700; }}
        .sla-overall.green .sla-overall-value {{ color: #27ae60; }}
        .sla-overall.yellow .sla-overall-value {{ color: #e67e22; }}
        .sla-overall.red .sla-overall-value {{ color: #c0392b; }}
        .sla-overall-pct {{ font-size: 14px; color: var(--gristexto); margin-top: 4px; }}
        .sla-thresholds-note {{ font-size: 11px; color: #aaa; margin-top: 8px; }}
        .sla-breakdown {{ flex: 2; min-width: 300px; display: flex; flex-direction: column; gap: 12px; }}
        .sla-item {{ display: flex; align-items: center; gap: 16px; padding: 16px; border-radius: 10px; }}
        .sla-item.green {{ background: #e8f8e8; border-left: 4px solid #2ecc71; }}
        .sla-item.yellow {{ background: #fff8e1; border-left: 4px solid #f39c12; }}
        .sla-item.red {{ background: #fde8e8; border-left: 4px solid #e74c3c; }}
        .sla-indicator {{ font-size: 28px; width: 44px; text-align: center; }}
        .sla-info {{ flex: 1; }}
        .sla-label {{ font-size: 12px; color: var(--gristexto); text-transform: uppercase; letter-spacing: 0.3px; }}
        .sla-value {{ font-size: 22px; font-weight: 700; color: var(--azul-principal); }}
        
        /* ROI */
        .roi-card .card-value {{ font-size: 24px; }}
        .roi-methodology {{ padding: 8px 0; line-height: 1.8; color: var(--azul-principal); font-size: 14px; }}
        .roi-methodology ul {{ padding-left: 24px; }}
        .roi-methodology li {{ margin-bottom: 8px; }}
        
        /* PDF */
        @media print {{
            body {{ padding: 0; }}
            .btn, .dashboard-tabs {{ display: none !important; }}
            .tab-content {{ display: block !important; }}
            .tab-content#logs-tab {{ display: none !important; }}
            .tab-content#history-tab {{ display: none !important; }}
            .card, .chart-container {{ break-inside: avoid; box-shadow: none; border: 1px solid #ddd; }}
        }}
        
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
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <button class="btn" onclick="exportPDF()">📄 Exportar Reporte</button>
                <button class="btn" onclick="refresh()">🔄 Actualizar</button>
            </div>
        </div>
        
        <div class="dashboard-tabs">
            <button class="tab-btn active" onclick="showTab('metrics')">📊 Métricas</button>
            <button class="tab-btn" onclick="showTab('sla')">🎯 SLA</button>
            <button class="tab-btn" onclick="showTab('roi')">💰 ROI</button>
            <button class="tab-btn" onclick="showTab('logs')">📋 Logs del Sistema</button>
            <button class="tab-btn" onclick="showTab('history')">📜 Historial</button>
        </div>
        
        <div id="metrics-tab" class="tab-content">
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
            <div class="card">
                <div class="card-label">Tokens Hoy</div>
                <div class="card-value">{tokens_hoy:,} <span style="font-size:14px;color:#7f8c8d;">/ {limite:,}</span></div>
                <div class="card-sub">{porcentaje}% del límite diario</div>
                <div class="progress-bar"><div class="progress-fill" style="width:{porcentaje}%"></div></div>
            </div>
            <div class="card">
                <div class="card-label">Promedio Tokens</div>
                <div class="card-value">{prom_tokens:,}</div>
                <div class="card-sub">por consulta</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-label">📊 Tasa de Éxito</div>
                <div class="card-value positivo">{tasa_exito:.1f}%</div>
                <div class="card-sub">útiles: {respuestas_utiles} | no útiles: {respuestas_no_utiles}</div>
                <div class="progress-bar success"><div class="progress-fill success" style="width:{tasa_exito}%"></div></div>
            </div>
            <div class="card">
                <div class="card-label">✅ Respuestas Útiles</div>
                <div class="card-value">{respuestas_utiles:,}</div>
                <div class="card-sub">con información relevante</div>
            </div>
            <div class="card">
                <div class="card-label">❌ Sin Información</div>
                <div class="card-value">{respuestas_no_utiles:,}</div>
                <div class="card-sub">errores o no encontrado</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">📈 Consumo de Tokens por Hora</div>
            <pre class="ascii-chart">{grafico_tokens}</pre>
            <div style="display:flex;gap:20px;margin-top:8px;flex-wrap:wrap;">
                <span class="card-sub">🔵 Pico máximo: {max_tph:,} tokens/hora</span>
                <span class="card-sub">📊 Promedio: {avg_tph:,} tokens/hora</span>
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
        
        </div>
        
        <div id="sla-tab" class="tab-content" style="display:none;">
        <div class="chart-container">
            <div class="chart-title">🎯 Cumplimiento de SLA</div>
            <div class="sla-overview">
                <div class="sla-overall {sla_data["overall_status"]}">
                    <div class="sla-overall-icon">{sla_overall_icon}</div>
                    <div class="sla-overall-text">Estado General del SLA</div>
                    <div class="sla-overall-value">{sla_data["overall_status"].upper()}</div>
                    <div class="sla-overall-pct">{sla_data["overall_pct"]:.0f}% de cumplimiento</div>
                    <div class="sla-thresholds-note">Verde ≥ 80% · Amarillo ≥ 50% · Rojo &lt; 50%</div>
                </div>
                <div class="sla-breakdown">
                    {sla_items_html}
                </div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">📋 Definición de Umbrales SLA</div>
            <table>
                <thead>
                    <tr><th>Indicador</th><th>🟢 Verde</th><th>🟡 Amarillo</th><th>🔴 Rojo</th></tr>
                </thead>
                <tbody>
                    {''.join(f'<tr><td>{item["label"]}</td><td style="color:#2ecc71;">Verde</td><td style="color:#f39c12;">Amarillo</td><td style="color:#e74c3c;">Rojo</td></tr>' for item in sla_data["items"])}
                </tbody>
            </table>
        </div>
    </div>
    
    <div id="roi-tab" class="tab-content" style="display:none;">
        <div class="grid">
            <div class="card roi-card">
                <div class="card-label">💰 Ahorro Total</div>
                <div class="card-value positivo">${roi_total_savings:,.2f}</div>
                <div class="card-sub">Desde el inicio del servicio</div>
            </div>
            <div class="card roi-card">
                <div class="card-label">📅 Ahorro Mensual Estimado</div>
                <div class="card-value positivo">${roi_monthly:,.2f}</div>
                <div class="card-sub">Basado en {roi_data["total_queries"]:,} consultas procesadas</div>
            </div>
            <div class="card roi-card">
                <div class="card-label">📆 Ahorro Anual Proyectado</div>
                <div class="card-value positivo">${roi_yearly:,.2f}</div>
                <div class="card-sub">Proyección a 12 meses</div>
            </div>
            <div class="card roi-card">
                <div class="card-label">⏱️ Tiempo Ahorrado</div>
                <div class="card-value positivo">{roi_time_saved:,.1f} hrs</div>
                <div class="card-sub">Equivalente a {roi_time_saved / 8:.1f} días laborales</div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">📊 Desglose de Ahorro por Consulta</div>
            <table>
                <thead>
                    <tr><th>Métrica</th><th>Valor</th></tr>
                </thead>
                <tbody>
                    <tr><td>Total de consultas procesadas</td><td><strong>{roi_data["total_queries"]:,}</strong></td></tr>
                    <tr><td>Tiempo promedio del bot por consulta</td><td><strong>{roi_data["bot_avg_minutes"]:.2f} min</strong></td></tr>
                    <tr><td>Tiempo promedio humano estimado</td><td><strong>{roi_data["human_avg_minutes"]:.0f} min</strong></td></tr>
                    <tr><td>Ahorro de tiempo por consulta</td><td><strong>{roi_data["human_avg_minutes"] - roi_data["bot_avg_minutes"]:.2f} min</strong></td></tr>
                    <tr><td>Costo evitado por consulta</td><td><strong>${roi_data["human_cost_rate"] / 60 * roi_data["human_avg_minutes"]:.2f}</strong></td></tr>
                    <tr><td>Costo por hora de agente humano</td><td><strong>${roi_data["human_cost_rate"]:.2f}/hora</strong></td></tr>
                    <tr><td>Ahorro total en tiempo</td><td><strong>{roi_time_saved:,.1f} horas ({roi_time_saved / 8:.1f} días)</strong></td></tr>
                    <tr><td>Ahorro mensual estimado</td><td><strong>${roi_monthly:,.2f}</strong></td></tr>
                    <tr><td>Ahorro anual proyectado</td><td><strong>${roi_yearly:,.2f}</strong></td></tr>
                </tbody>
            </table>
        </div>
        <div class="chart-container">
            <div class="chart-title">💡 Metodología de Cálculo</div>
            <div class="roi-methodology">
                <p>El cálculo de ROI compara el tiempo que tomaría a un agente humano resolver las mismas consultas versus el tiempo real del chatbot:</p>
                <ul>
                    <li><strong>Costo agente humano:</strong> ${roi_data["human_cost_rate"]:.0f} USD/hora (promedio para soporte educativo)</li>
                    <li><strong>Tiempo humano por consulta:</strong> {roi_data["human_avg_minutes"]:.0f} minutos (estimado)</li>
                    <li><strong>Tiempo del chatbot:</strong> {roi_data["bot_avg_minutes"]:.2f} minutos promedio</li>
                    <li><strong>Costo operativo del chatbot:</strong> $0 USD (capa gratuita Groq)</li>
                    <li><strong>Fórmula:</strong> Ahorro = (Tiempo_humano - Tiempo_bot) × Consultas × Costo_hora / 60</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div id="history-tab" class="tab-content" style="display:none;">
        <div class="chart-container">
            <div class="chart-title">Historial Reciente</div>
            <table id="tablaHistorial">
                <thead>
                    <tr><th>Fecha</th><th>Pregunta</th><th>Respuesta</th><th>Tiempo</th><th>Tokens</th><th>RAG</th></tr>
                </thead>
                <tbody>
                    {historial_html}
                </tbody>
            </table>
        </div>
    </div>
    
    <div id="logs-tab" class="tab-content" style="display:none;">
        <div class="logs-controls">
            <select id="log-level-filter">
                <option value="all">Todos los niveles</option>
                <option value="ERROR">❌ Solo Errores</option>
                <option value="WARNING">⚠️ Solo Advertencias</option>
                <option value="INFO">ℹ️ Solo Info</option>
            </select>
            <button onclick="refreshLogs()" class="btn" style="padding:8px 16px;">🔄 Refrescar</button>
            <span id="log-count" class="log-count"></span>
        </div>
        <div class="logs-container" id="logs-container">
            <div class="log-loading">📋 Cargando logs...</div>
        </div>
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
        
        function showTab(name) {{
            document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(name + '-tab').style.display = 'block';
            document.querySelector(`.tab-btn[onclick*="'${{name}}'"]`).classList.add('active');
            if (name === 'logs') refreshLogs();
        }}
        
        function escapeHtml(text) {{
            return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }}
        
        async function refreshLogs() {{
            const level = document.getElementById('log-level-filter').value;
            const url = level === 'all' ? '/api/logs?limit=200' : `/api/logs?limit=200&level=${{level}}`;
            try {{
                const r = await fetch(url);
                const data = await r.json();
                const container = document.getElementById('logs-container');
                document.getElementById('log-count').textContent = data.logs.length ? `Mostrando ${{data.logs.length}} de ${{data.total}} logs` : '';
                if (!data.logs.length) {{
                    container.innerHTML = '<div class="log-empty">No hay logs que coincidan con el filtro</div>';
                    return;
                }}
                container.innerHTML = data.logs.map(log => {{
                    const cls = log.level.toLowerCase();
                    const icon = log.level === 'ERROR' ? '❌' : log.level === 'WARNING' ? '⚠️' : 'ℹ️';
                    const time = new Date(log.timestamp).toLocaleString('es-MX');
                    return `<div class="log-entry ${{cls}}"><div class="log-header"><span class="log-level">${{icon}} ${{log.level}}</span><span class="log-time">${{time}}</span><span class="log-module">📁 ${{log.module}}</span></div><div class="log-message">${{escapeHtml(log.message)}}</div>${{log.line ? `<div class="log-line">📍 Línea: ${{log.line}}</div>` : ''}}</div>`;
                }}).join('');
            }} catch(e) {{
                document.getElementById('logs-container').innerHTML = '<div class="log-error">❌ Error cargando logs. Verifica que /api/logs existe.</div>';
            }}
        }}
        
        let logInterval = null;
        document.addEventListener('visibilitychange', () => {{
            const logsTab = document.getElementById('logs-tab');
            const visible = logsTab && logsTab.style.display !== 'none';
            if (visible && !logInterval) {{
                logInterval = setInterval(refreshLogs, 10000);
            }} else if (!visible && logInterval) {{
                clearInterval(logInterval);
                logInterval = null;
            }}
        }});
        
        async function exportPDF() {{
            const btn = event.target || document.querySelector('.btn');
            const originalText = btn.textContent;
            btn.textContent = '⏳ Generando PDF...';
            btn.disabled = true;
            
            try {{
                const element = document.querySelector('.container');
                const fecha = new Date().toLocaleDateString('es-MX', {{
                    year: 'numeric', month: 'long', day: 'numeric'
                }});
                
                const opt = {{
                    margin:        [10, 10, 10, 10],
                    filename:     `Reporte-Chatbot-RAG-${{fecha.replace(/ /g, '-')}}.pdf`,
                    image:        {{ type: 'jpeg', quality: 0.98 }},
                    html2canvas:  {{ scale: 2, useCORS: true, logging: false }},
                    jsPDF:        {{ unit: 'mm', format: 'a4', orientation: 'portrait' }},
                    pagebreak:    {{ mode: ['avoid-all', 'css', 'legacy'] }}
                }};
                
                // Mostrar todas las pestañas temporalmente para capturar contenido
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'block');
                
                await html2pdf().set(opt).from(element).save();
                
                // Restaurar visibilidad de pestañas
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                const activeTab = document.querySelector('.tab-btn.active');
                if (activeTab) {{
                    const tabName = activeTab.getAttribute('onclick').match(/'([^']+)'/)[1];
                    document.getElementById(tabName + '-tab').style.display = 'block';
                }}
            }} catch (e) {{
                console.error('Error generando PDF:', e);
                alert('Error al generar el PDF. Revisa la consola para más detalles.');
            }} finally {{
                btn.textContent = originalText;
                btn.disabled = false;
            }}
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
    
    token_porcentaje = get_token_stats()["porcentaje"]
    sla_data = calculate_sla_metrics({**metrics, "token_porcentaje": token_porcentaje})
    roi_data = calculate_roi(metrics)
    print(f"     SLA General: {sla_data['overall_status'].upper()} ({sla_data['overall_pct']:.0f}%)")
    print(f"     Ahorro Total: ${roi_data['total_savings']:,.2f} | Mensual: ${roi_data['monthly_savings']:,.2f}")
    
    # Alertas críticas vía Telegram
    if sla_data["overall_status"] == "red":
        msg = (
            f"🚨 *ALERTA SLA - Dashboard {datetime.now().strftime('%d/%m/%Y')}*\n"
            f"Estado general del servicio: 🔴 CRÍTICO\n"
            f"Cumplimiento: {sla_data['overall_pct']:.0f}%\n\n"
        )
        for item in sla_data["items"]:
            icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(item["status"], "⚪")
            msg += f"{icon} {item['label']}: {item['value']}\n"
        msg += f"\n📊 Dashboard: {output_path}"
        send_telegram_alert(msg)
    
    if metrics.get("tasa_no_encontrado", 0) > 25:
        send_telegram_alert(
            f"⚠️ *Alerta de Calidad - {datetime.now().strftime('%d/%m/%Y')}*\n"
            f"Tasa de 'No encontrado' elevada: {metrics['tasa_no_encontrado']:.1f}%\n"
            f"Por encima del umbral del 25%"
        )
    
    return output_path


if __name__ == "__main__":
    generate_user_dashboard()