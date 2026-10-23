"""Genera un reporte HTML comparativo 'antes vs después' de dos análisis.

Toma dos CSV ya evaluados (``*_analizado.csv`` con columna ``evaluacion``) y
produce un HTML autocontenido con el mismo estilo que los reportes previos:
KPIs, gráficas comparativas, cambios aplicados y una tabla por pregunta con la
etiqueta ``mejoró / igual / empeoró``.

Uso:
    python scripts/comparar_antes_despues.py \
        --antes Reportes/Preguntas_equipo/consulta-conversations-2026-09-29_analizado.csv \
        --despues Reportes/Preguntas_equipo/consulta-conversations-2026-09-29_post_analizado.csv \
        --salida Reportes/Preguntas_equipo/comparativo_antes_despues_2026-09-29.html
"""
import argparse
import base64
import html
import io
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("comparar_antes_despues")

COLORES = {"bien": "#28a745", "a_medias": "#f0ad4e", "mal": "#dc3545"}
RANGO = {"bien": 2, "a_medias": 1, "mal": 0}
AZUL = "#2c3e50"


def img_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cargar(ruta: str, sufijo: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["_orden"] = df.groupby("pregunta").cumcount()
    df["_clave"] = df["pregunta"].astype(str) + "||" + df["_orden"].astype(str)
    df = df.rename(columns={
        "evaluacion": f"evaluacion_{sufijo}",
        "respuesta": f"respuesta_{sufijo}",
        "justificacion": f"justificacion_{sufijo}",
        "tokens": f"tokens_{sufijo}",
    })
    df[f"ms_{sufijo}"] = df["tiempo"].astype(str).str.replace("ms", "", regex=False).astype(float)
    return df


def metricas(df, sufijo):
    ev = df[f"evaluacion_{sufijo}"].value_counts().to_dict()
    n = len(df)
    return {
        "n": n,
        "bien": ev.get("bien", 0),
        "a_medias": ev.get("a_medias", 0),
        "mal": ev.get("mal", 0),
        "bien_pct": round(ev.get("bien", 0) / n * 100, 1) if n else 0,
        "mal_pct": round(ev.get("mal", 0) / n * 100, 1) if n else 0,
        "ms_mediana": df[f"ms_{sufijo}"].median(),
        "tokens_media": df[f"tokens_{sufijo}"].mean(),
    }


def graficas(m_antes, m_despues):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cats = ["bien", "a_medias", "mal"]
    pos = range(len(cats))
    ax.bar([p - 0.2 for p in pos], [m_antes[c] for c in cats], width=0.4, label="Antes", color="#95a5a6")
    ax.bar([p + 0.2 for p in pos], [m_despues[c] for c in cats], width=0.4, label="Después", color="#3498db")
    for i, c in enumerate(cats):
        ax.text(i - 0.2, m_antes[c] + 0.2, str(m_antes[c]), ha="center", fontsize=8)
        ax.text(i + 0.2, m_despues[c] + 0.2, str(m_despues[c]), ha="center", fontsize=8)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(["Bien", "A medias", "Mal"])
    ax.set_ylabel("Preguntas")
    ax.set_title("Evaluación: antes vs después", color=AZUL, fontweight="bold")
    ax.legend(frameon=False)
    g1 = img_b64(fig)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar([0 - 0.2, 1 - 0.2], [m_antes["ms_mediana"], m_antes["tokens_media"]], width=0.4, label="Antes", color="#95a5a6")
    ax.bar([0 + 0.2, 1 + 0.2], [m_despues["ms_mediana"], m_despues["tokens_media"]], width=0.4, label="Después", color="#3498db")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Latencia mediana (ms)", "Tokens medios"])
    ax.set_title("Latencia y tokens", color=AZUL, fontweight="bold")
    ax.legend(frameon=False)
    g2 = img_b64(fig)
    return g1, g2


def render(df, cambios):
    filas = []
    n_mejoro = n_igual = n_empeoro = 0
    for _, r in df.iterrows():
        a, d = r["evaluacion_antes"], r["evaluacion_despues"]
        delta = RANGO.get(d, 0) - RANGO.get(a, 0)
        if delta > 0:
            cambio, col = "mejoró", "#28a745"; n_mejoro += 1
        elif delta < 0:
            cambio, col = "empeoró", "#dc3545"; n_empeoro += 1
        else:
            cambio, col = "igual", "#7f8c8d"; n_igual += 1
        filas.append(f"""
        <div class="fila">
          <span class="badge" style="background:{COLORES.get(a,'#95a5a6')}">{a}</span>
          <span class="flecha">→</span>
          <span class="badge" style="background:{COLORES.get(d,'#95a5a6')}">{d}</span>
          <span class="badge" style="background:{col}">{cambio}</span>
          <details class="detalle-pregunta">
            <summary><strong>{html.escape(str(r['pregunta']))}</strong></summary>
            <div class="respuesta"><h4>Antes</h4><p>{html.escape(str(r['respuesta_antes']))}</p>
            <p class="muted">{html.escape(str(r.get('justificacion_antes','')))}</p></div>
            <div class="respuesta"><h4>Después</h4><p>{html.escape(str(r['respuesta_despues']))}</p>
            <p class="muted">{html.escape(str(r.get('justificacion_despues','')))}</p></div>
          </details>
        </div>""")
    return "\n".join(filas), n_mejoro, n_igual, n_empeoro


def construir_html(df, m_antes, m_despues, cambios, contexto):
    g1, g2 = graficas(m_antes, m_despues)
    tabla, n_mejoro, n_igual, n_empeoro = render(df, cambios)

    filas_cambios = ""
    for c in cambios.get("cambios", []):
        filas_cambios += f"""
        <div class="card-dia">
          <h4>{html.escape(c['titulo'])} <span class="n">[{html.escape(c['id'])}]</span></h4>
          <p class="muted"><b>Antes:</b> {html.escape(c['antes'])}</p>
          <p class="muted"><b>Después:</b> {html.escape(c['despues'])}</p>
        </div>"""
    backlog = "".join(f"<li>{html.escape(b)}</li>" for b in cambios.get("backlog", []))

    delta_bien = m_despues["bien"] - m_antes["bien"]
    delta_mal = m_despues["mal"] - m_antes["mal"]

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(contexto['titulo'])}</title>
<style>
  :root {{ --azul-principal:#2c3e50; --azul-secundario:#3498db; --verdeclaro:#2ecc71; --rojosoft:#e74c3c; --gris:#7f8c8d; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:var(--azul-principal); }}
  .contenedor {{ max-width:1220px; margin:0 auto; padding:24px 20px 60px; }}
  h1 {{ font-size:26px; }}
  .subtitulo {{ color:var(--gris); margin-top:6px; font-size:14px; }}
  .chip {{ display:inline-block; background:#ecf0f1; border-radius:20px; padding:4px 12px; font-size:12px; margin:12px 8px 0 0; }}
  .seccion {{ background:#fff; border-radius:12px; padding:22px 24px; margin-top:22px; box-shadow:0 2px 10px rgba(0,0,0,.07); }}
  .seccion h2 {{ font-size:19px; margin-bottom:14px; border-left:4px solid var(--azul-secundario); padding-left:10px; }}
  .metrics-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:8px; }}
  .metric-card {{ background:#fff; padding:18px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.07); text-align:center; }}
  .metric-value {{ font-size:28px; font-weight:800; }}
  .metric-label {{ color:var(--gris); font-size:12px; margin-top:4px; text-transform:uppercase; }}
  .success .metric-value {{ color:var(--verdeclaro); }}
  .error .metric-value {{ color:var(--rojosoft); }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .grid-2 img {{ width:100%; border-radius:10px; }}
  .fila {{ display:flex; align-items:baseline; gap:8px; padding:10px 6px; border-bottom:1px solid #f0f3f6; }}
  .badge {{ color:#fff; padding:3px 11px; border-radius:16px; font-size:12px; font-weight:700; }}
  .flecha {{ color:var(--gris); }}
  details.detalle-pregunta {{ flex:1; min-width:0; }}
  details summary {{ cursor:pointer; font-size:13.5px; }}
  .respuesta {{ background:#f8fafc; border-left:3px solid var(--azul-secundario); border-radius:4px; padding:12px; margin-top:10px; }}
  .respuesta h4 {{ font-size:12px; text-transform:uppercase; color:var(--gris); margin-bottom:6px; }}
  .respuesta p {{ white-space:pre-wrap; line-height:1.55; font-size:13px; }}
  .muted {{ color:var(--gris); font-size:12.5px; }}
  .card-dia {{ border:1px solid #ecf0f1; border-radius:10px; padding:14px; margin-bottom:12px; }}
  footer {{ margin-top:34px; color:var(--gris); font-size:12.5px; text-align:center; }}
  @media (max-width:760px) {{ .metrics-grid {{ grid-template-columns:repeat(2,1fr); }} .grid-2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="contenedor">
  <h1>🔁 {html.escape(contexto['titulo'])}</h1>
  <p class="subtitulo">{html.escape(contexto['subtitulo'])}</p>
  <span class="chip">🗓️ {html.escape(contexto['fechas'])}</span>
  <span class="chip">🧪 {len(df)} preguntas comparadas</span>

  <div class="seccion">
    <h2>Resumen</h2>
    <div class="metrics-grid">
      <div class="metric-card success"><div class="metric-value">{m_despues['bien']} ({m_despues['bien_pct']}%)</div><div class="metric-label">Bien (después)</div></div>
      <div class="metric-card"><div class="metric-value">{m_antes['bien']} ({m_antes['bien_pct']}%)</div><div class="metric-label">Bien (antes)</div></div>
      <div class="metric-card {'success' if delta_bien>0 else 'error' if delta_bien<0 else ''}"><div class="metric-value">{'+' if delta_bien>=0 else ''}{delta_bien}</div><div class="metric-label">Δ Bien</div></div>
      <div class="metric-card {'success' if delta_mal<0 else 'error' if delta_mal>0 else ''}"><div class="metric-value">{'+' if delta_mal>=0 else ''}{delta_mal}</div><div class="metric-label">Δ Mal</div></div>
    </div>
    <div class="metrics-grid" style="margin-top:16px;">
      <div class="metric-card"><div class="metric-value">{n_mejoro}</div><div class="metric-label">Mejoraron</div></div>
      <div class="metric-card"><div class="metric-value">{n_igual}</div><div class="metric-label">Igual</div></div>
      <div class="metric-card"><div class="metric-value">{n_empeoro}</div><div class="metric-label">Empeoraron</div></div>
      <div class="metric-card"><div class="metric-value">{m_despues['ms_mediana']:.0f} ms</div><div class="metric-label">Latencia mediana (después)</div></div>
    </div>
  </div>

  <div class="seccion">
    <h2>Comparativa visual</h2>
    <div class="grid-2">
      <img src="data:image/png;base64,{g1}" alt="Evaluación antes vs después">
      <img src="data:image/png;base64,{g2}" alt="Latencia y tokens">
    </div>
  </div>

  <div class="seccion">
    <h2>Cambios aplicados</h2>
    {filas_cambios}
    <p class="muted" style="margin-top:10px;"><b>Backlog:</b></p>
    <ul class="muted">{backlog}</ul>
  </div>

  <div class="seccion">
    <h2>Detalle por pregunta ({len(df)})</h2>
    {tabla}
  </div>

  <footer>Generado por <code>scripts/comparar_antes_despues.py</code> · Evaluación manual/asistida · Las etiquetas reflejan la rúbrica bien/a_medias/mal.</footer>
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Reporte comparativo antes vs después.")
    parser.add_argument("--antes", required=True)
    parser.add_argument("--despues", required=True)
    parser.add_argument("--salida", default="Reportes/Preguntas_equipo/comparativo_antes_despues_2026-09-29.html")
    parser.add_argument("--cambios", default="Reportes/Preguntas_equipo/cambios_aplicados_2026-09-29.json")
    parser.add_argument("--titulo", default="Comparativo antes vs después — Preguntas del equipo (25-29 sep 2026)")
    args = parser.parse_args()

    df_a = cargar(args.antes, "antes")
    df_d = cargar(args.despues, "despues")
    df = df_a.merge(
        df_d[["_clave", "evaluacion_despues", "respuesta_despues", "justificacion_despues", "tokens_despues", "ms_despues"]],
        on="_clave", how="inner",
    )
    if len(df) == 0:
        logger.error("No se pudieron emparejar preguntas entre ambos CSV.")
        return 1
    logger.info("Preguntas emparejadas: %d", len(df))

    m_antes = metricas(df, "antes")
    m_despues = metricas(df, "despues")

    cambios = {}
    ruta_cambios = Path(args.cambios)
    if ruta_cambios.exists():
        cambios = json.loads(ruta_cambios.read_text(encoding="utf-8"))

    contexto = {
        "titulo": args.titulo,
        "subtitulo": "Comparación de la evaluación de las mismas preguntas antes y después de las mejoras.",
        "fechas": "25/09/2026 - 29/09/2026",
    }
    html_out = construir_html(df, m_antes, m_despues, cambios, contexto)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.salida).write_text(html_out, encoding="utf-8")
    logger.info("Reporte comparativo guardado: %s", args.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
