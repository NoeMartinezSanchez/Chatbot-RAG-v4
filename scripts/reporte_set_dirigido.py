"""Genera el HTML de resultados del set dirigido A+B+C (mejoras de octubre).

Lee:
  - Reportes/Mejoras_Octubre/set_dirigido_abc.csv        (conjunto, categoria, evaluacion_previa)
  - Reportes/Mejoras_Octubre/set_dirigido_abc_post.csv   (respuesta del replay)

Y produce un HTML autocontenido (mismo estilo que los reportes previos) con
KPIs, gráficas por conjunto y la tabla por pregunta.

Uso:
    python scripts/reporte_set_dirigido.py
"""
import base64
import html
import io
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("reporte_set_dirigido")

BASE = Path("Reportes/Mejoras_Octubre")
PLAN = BASE / "set_dirigido_abc.csv"
POST = BASE / "set_dirigido_abc_post.csv"
SALIDA = BASE / "reporte_set_dirigido_abc.html"

COLORES = {"bien": "#28a745", "a_medias": "#f0ad4e", "mal": "#dc3545"}
RANGO = {"bien": 2, "a_medias": 1, "mal": 0}


def img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- Clasificación manual (revisión de las 96 respuestas) ---
MAL = {5, 47, 54}
MEDIAS_A = {2, 4, 6, 11, 12, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53}
MEDIAS_C = {66}
MEDIAS_B = {83}

NOTAS = {
    0: "Mejora: antes 'a medias'; ahora define correctamente 'servicio educativo federal'.",
    5: "La base indica preguntar al asesor/tutor; siguió con el mensaje de fuera de dominio.",
    7: "Mejora: antes 'mal'; ahora responde la duración (2 años y 6 meses).",
    10: "Mejora: antes 'mal'; ahora confirma que no se pueden cursar módulos simultáneos.",
    13: "Mejora: antes 'mal'; ahora explica cómo revisar el estatus del registro.",
    14: "Mejora: antes 'mal'; ahora indica dónde registrarse.",
    17: "Mejora: antes 'mal' (bloqueo de seguridad); ahora responde con normalidad.",
    40: "Mejora: identifica tema de 'otra área' (bachillerato, no carreras) y canaliza.",
    41: "Mejora: antes 'a medias'; ahora explica la validez nacional del certificado.",
    47: "Fallo de recuperación: 'claves/llave de acceso' está en la base pero no se recuperó.",
    54: "Fallo de recuperación: la recuperación de contraseña está en la base.",
    66: "Regresión: el glosario interceptó una pregunta legítima ('validar que estudio en PL') y devolvió la definición de la sigla.",
    83: "Ajuste: 'muchas gracias' recibió la respuesta de saludo; conviene un '¡Con gusto!'. Menor.",
    78: "Aceptable: dio la definición de la sigla (PL) para la variante PL-SEP.",
}


def clasificar(i, conjunto):
    if i in MAL:
        return "mal"
    if conjunto == "A_foco" and i in MEDIAS_A:
        return "a_medias"
    if conjunto == "C_regresion" and i in MEDIAS_C:
        return "a_medias"
    if conjunto == "B_nuevas" and i in MEDIAS_B:
        return "a_medias"
    return "bien"


def justificar(i, ev, conjunto):
    if i in NOTAS:
        return NOTAS[i]
    if ev == "mal":
        return "Fallo de recuperación: la información está en la base de conocimientos pero no se recuperó."
    if ev == "a_medias":
        if conjunto == "A_foco":
            return "Respuesta parcial o fuera de dominio para un tema que debería cubrirse mejor."
        return "Respuesta parcial o genérica."
    return "La respuesta es correcta y completa."


def grafica_conjunto(por_conjunto):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    conjuntos = ["A_foco", "B_nuevas", "C_regresion"]
    etiquetas = ["A · foco", "B · nuevas", "C · regresión"]
    pos = range(len(conjuntos))
    ancho = 0.25
    for k, ev in enumerate(["bien", "a_medias", "mal"]):
        vals = [por_conjunto[c].get(ev, 0) for c in conjuntos]
        ax.bar([p + ancho * (k - 1) for p in pos], vals, width=ancho, label=ev, color=COLORES[ev])
        for i, v in enumerate(vals):
            if v:
                ax.text(i + ancho * (k - 1), v + 0.15, str(v), ha="center", fontsize=8)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Preguntas")
    ax.set_title("Evaluación por conjunto", color="#2c3e50", fontweight="bold")
    ax.legend(frameon=False)
    return img_b64(fig)


def grafica_A(antes, despues):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    evs = ["bien", "a_medias", "mal"]
    pos = range(len(evs))
    ax.bar([p - 0.2 for p in pos], [antes.get(e, 0) for e in evs], width=0.4, label="Antes", color="#95a5a6")
    ax.bar([p + 0.2 for p in pos], [despues.get(e, 0) for e in evs], width=0.4, label="Después", color="#3498db")
    for i, e in enumerate(evs):
        ax.text(i - 0.2, antes.get(e, 0) + 0.2, str(antes.get(e, 0)), ha="center", fontsize=8)
        ax.text(i + 0.2, despues.get(e, 0) + 0.2, str(despues.get(e, 0)), ha="center", fontsize=8)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(["Bien", "A medias", "Mal"])
    ax.set_ylabel("Preguntas")
    ax.set_title("Set A (foco): antes vs después", color="#2c3e50", fontweight="bold")
    ax.legend(frameon=False)
    return img_b64(fig)


def grafica_B(por_categoria):
    cats = sorted(por_categoria, key=lambda c: -por_categoria[c]["n"])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bienes = [por_categoria[c]["bien"] for c in cats]
    totales = [por_categoria[c]["n"] for c in cats]
    pos = range(len(cats))
    ax.barh(list(pos), totales, color="#ecf0f1")
    ax.barh(list(pos), bienes, color="#28a745")
    ax.set_yticks(list(pos))
    ax.set_yticklabels(cats, fontsize=9)
    ax.invert_yaxis()
    for i, c in enumerate(cats):
        ax.text(por_categoria[c]["n"] + 0.1, i, f"{por_categoria[c]['bien']}/{por_categoria[c]['n']}", va="center", fontsize=8)
    ax.set_xlabel("Preguntas (verde = correctas)")
    ax.set_title("Set B (nuevas): aciertos por categoría", color="#2c3e50", fontweight="bold")
    return img_b64(fig)


def main():
    plan = pd.read_csv(PLAN)
    post = pd.read_csv(POST, encoding="utf-8-sig")
    if len(plan) != len(post):
        logger.error("El plan (%d) y el post (%d) no coinciden en filas.", len(plan), len(post))
        return 1

    df = pd.DataFrame({
        "conjunto": plan["conjunto"],
        "categoria": plan["categoria"],
        "pregunta": post["pregunta"],
        "respuesta": post["respuesta"],
        "previa": plan["evaluacion_previa"].fillna("").astype(str),
    })
    df["evaluacion"] = [clasificar(i, df["conjunto"].iloc[i]) for i in range(len(df))]
    df["justificacion"] = [justificar(i, df["evaluacion"].iloc[i], df["conjunto"].iloc[i]) for i in range(len(df))]

    # Métricas
    total = len(df)
    ev = Counter(df["evaluacion"])
    por_conjunto = {c: Counter(df[df["conjunto"] == c]["evaluacion"]) for c in df["conjunto"].unique()}

    # Set A antes/después
    A = df[df["conjunto"] == "A_foco"]
    antes = Counter(A["previa"].replace("", "a_medias"))
    despues = Counter(A["evaluacion"])
    A["dif"] = A["evaluacion"].map(RANGO) - A["previa"].map(RANGO).fillna(1)
    a_mejor = int((A["dif"] > 0).sum()); a_igual = int((A["dif"] == 0).sum()); a_empeor = int((A["dif"] < 0).sum())

    # Set B por categoría
    B = df[df["conjunto"] == "B_nuevas"]
    por_categoria = defaultdict(lambda: {"n": 0, "bien": 0})
    for _, r in B.iterrows():
        por_categoria[r["categoria"]]["n"] += 1
        if r["evaluacion"] == "bien":
            por_categoria[r["categoria"]]["bien"] += 1

    # Set C regresiones
    C = df[df["conjunto"] == "C_regresion"].copy()
    C["dif"] = C["evaluacion"].map(RANGO) - C["previa"].map(RANGO)
    c_regres = int((C["dif"] < 0).sum())

    g_conj = grafica_conjunto(por_conjunto)
    g_a = grafica_A(antes, despues)
    g_b = grafica_B(por_categoria)

    # Tabla por pregunta
    filas = []
    for _, r in df.iterrows():
        color = COLORES.get(r["evaluacion"], "#95a5a6")
        prev = f'<span class="badge" style="background:#95a5a6">{r["previa"]}</span> → ' if r["previa"] else ""
        rev = " si" if str(r.get("revisar", "")) == "si" else ""
        filas.append(f"""
        <div class="fila">
          <span class="col-conj">{html.escape(r['conjunto'])}</span>
          <span class="col-cat">{html.escape(str(r['categoria']))}</span>
          {prev}<span class="badge" style="background:{color}">{r['evaluacion']}</span>
          <details class="detalle-pregunta">
            <summary><strong>{html.escape(str(r['pregunta']))}</strong></summary>
            <div class="respuesta"><h4>Respuesta</h4><p>{html.escape(str(r['respuesta']))}</p></div>
            <div class="justificacion"><h4>Justificación</h4><p>{html.escape(r['justificacion'])}</p></div>
          </details>
        </div>""")
    tabla = "\n".join(filas)

    html_out = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resultados del set dirigido A+B+C — Mejoras Octubre</title>
<style>
  :root {{ --azul:#2c3e50; --azul2:#3498db; --verde:#2ecc71; --rojo:#e74c3c; --gris:#7f8c8d; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:var(--azul); }}
  .cont {{ max-width:1200px; margin:0 auto; padding:24px 20px 60px; }}
  h1 {{ font-size:25px; }} .sub {{ color:var(--gris); margin-top:6px; font-size:14px; }}
  .chip {{ display:inline-block; background:#ecf0f1; border-radius:20px; padding:4px 12px; font-size:12px; margin:12px 8px 0 0; }}
  .sec {{ background:#fff; border-radius:12px; padding:22px 24px; margin-top:22px; box-shadow:0 2px 10px rgba(0,0,0,.07); }}
  .sec h2 {{ font-size:19px; margin-bottom:14px; border-left:4px solid var(--azul2); padding-left:10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:8px; }}
  .card {{ background:#fff; padding:18px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.07); text-align:center; }}
  .val {{ font-size:28px; font-weight:800; }} .lab {{ color:var(--gris); font-size:12px; margin-top:4px; text-transform:uppercase; }}
  .ok .val {{ color:var(--verde); }} .warn .val {{ color:#f0ad4e; }} .err .val {{ color:var(--rojo); }}
  .g2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }} .g2 img, .g3 img {{ width:100%; border-radius:10px; }}
  .fila {{ display:flex; align-items:baseline; gap:8px; padding:10px 6px; border-bottom:1px solid #f0f3f6; }}
  .col-conj {{ width:96px; font-size:11px; color:var(--azul2); font-weight:700; flex-shrink:0; }}
  .col-cat {{ width:120px; font-size:11px; color:var(--gris); flex-shrink:0; }}
  .badge {{ color:#fff; padding:3px 10px; border-radius:14px; font-size:12px; font-weight:700; }}
  details {{ flex:1; min-width:0; }} summary {{ cursor:pointer; font-size:13.5px; }}
  .respuesta,.justificacion {{ background:#f8fafc; border-left:3px solid var(--azul2); border-radius:4px; padding:10px; margin-top:8px; }}
  .respuesta h4,.justificacion h4 {{ font-size:11px; text-transform:uppercase; color:var(--gris); margin-bottom:4px; }}
  .respuesta p {{ white-space:pre-wrap; font-size:13px; line-height:1.5; }}
  .muted {{ color:var(--gris); font-size:12.5px; }}
  footer {{ margin-top:34px; color:var(--gris); font-size:12.5px; text-align:center; }}
  @media(max-width:760px){{ .grid{{grid-template-columns:repeat(2,1fr);}} .g2{{grid-template-columns:1fr;}} }}
</style></head>
<body><div class="cont">
  <h1>🎯 Resultados del set dirigido A+B+C — Mejoras de Octubre</h1>
  <p class="sub">Evaluación de {total} preguntas: A) foco (antes a medias/mal), B) funciones nuevas de hoy, C) regresión. Modelo gpt-oss-120b.</p>
  <span class="chip">🗓️ 24/10/2026</span><span class="chip">🧪 {total} preguntas</span>
  <span class="chip">A: {len(A)} · B: {len(B)} · C: {len(C)}</span>

  <div class="sec"><h2>Resumen</h2>
    <div class="grid">
      <div class="card"><div class="val">{total}</div><div class="lab">Preguntas</div></div>
      <div class="card ok"><div class="val">{ev.get('bien',0)}</div><div class="lab">Bien</div></div>
      <div class="card warn"><div class="val">{ev.get('a_medias',0)}</div><div class="lab">A medias</div></div>
      <div class="card err"><div class="val">{ev.get('mal',0)}</div><div class="lab">Mal</div></div>
    </div>
    <div class="grid" style="margin-top:16px;">
      <div class="card"><div class="val">{a_mejor}</div><div class="lab">A: mejoraron</div></div>
      <div class="card"><div class="val">{a_igual}</div><div class="lab">A: igual</div></div>
      <div class="card"><div class="val">{a_empeor}</div><div class="lab">A: empeoraron</div></div>
      <div class="card err"><div class="val">{c_regres}</div><div class="lab">C: regresiones</div></div>
    </div>
  </div>

  <div class="sec"><h2>Evaluación por conjunto</h2><div class="g2">
    <img src="data:image/png;base64,{g_conj}" alt="Por conjunto">
    <img src="data:image/png;base64,{g_b}" alt="Set B por categoría">
  </div></div>

  <div class="sec"><h2>Set A (foco): antes vs después</h2>
    <div class="g2">
      <img src="data:image/png;base64,{g_a}" alt="A antes/después">
      <div style="line-height:1.7;font-size:14px;">
        <p>De las <b>{len(A)}</b> preguntas que antes quedaron a medias o mal:
        <b>{a_mejor}</b> mejoraron, <b>{a_igual}</b> siguen igual y <b>{a_empeor}</b> empeoraron.</p>
        <p class="muted">Se confirman mejoras en recuperación (dónde registrarse, estatus del registro, módulos simultáneos, duración) y la desaparición del bloqueo de seguridad falso positivo.</p>
      </div>
    </div>
  </div>

  <div class="sec"><h2>Set C (regresión)</h2>
    <p>De {len(C)} preguntas que antes fueron "bien", se detectaron <b>{c_regres}</b> regresión(es).</p>
    <p class="muted">Regresión detectada: el glosario interceptó "como valido que estudie en pl" (debería ir al RAG). Pendiente de ajustar el disparo del glosario.</p>
  </div>

  <div class="sec"><h2>Detalle por pregunta ({total})</h2>
    {tabla}
  </div>

  <footer>Generado por <code>scripts/reporte_set_dirigido.py</code> · Evaluación manual · Modelo gpt-oss-120b.</footer>
</div></body></html>"""

    SALIDA.write_text(html_out, encoding="utf-8")
    logger.info("HTML guardado: %s", SALIDA)
    print(f"Global: {dict(ev)}")
    print(f"A: mejoraron {a_mejor} | igual {a_igual} | empeoraron {a_empeor}")
    print(f"C regresiones: {c_regres}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
