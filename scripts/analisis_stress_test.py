"""Analítica de conversaciones del chatbot RAG (rango de fechas configurable).

Lee un CSV exportado de MongoDB y evalúa cada pregunta como bien / a_medias / mal:
  1) Heurísticas determinísticas (fallback, placeholders sin resolver, longitud).
  2) Verificación de fallbacks contra la base de conocimientos en texto (.txt).
  3) Revisión asistida por LLM (Groq) opcional, o revisión manual vía JSON.

Trabaja para dos tipos de prueba:
  - Prueba de estrés controlada (12-13 sep 2026): preguntas bien/mal escritas.
  - Prueba con preguntas del equipo (25-29 sep 2026): conversaciones reales sin
    intención controlada; se clasifica tipo_pregunta (dentro/fuera del dominio,
    contextual, emocional) y se verifica cada fallback contra la base de conocimientos.

Genera en la carpeta de salida:
  - <origen>_analizado.csv     (filas del rango + columnas nuevas)
  - reporte_analitica_chatbot_<fecha>.html (autocontenido compartible)

Uso:
  python scripts/analisis_stress_test.py \
      --csv Reportes/Preguntas_equipo/consulta-conversations-2026-09-29.csv \
      --salida Reportes/Preguntas_equipo \
      --desde 2026-09-25 --hasta 2026-09-29 \
      --kb "Reportes/Base de conocimientos" \
      --manuales Reportes/Preguntas_equipo/revision_manual_2026-09-29.json \
      --titulo "Analítica del Chatbot — Preguntas del equipo"

En Windows ejecutar con PYTHONIOENCODING=utf-8.
"""

import argparse
import base64
import html
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analisis_chatbot")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODELO_LLM = os.getenv("ANALISIS_LLM_MODEL", "openai/gpt-oss-120b")
BATCH_LLM = 10
SLEEP_ENTRE_LOTES = 0.3


DOCUMENTOS = {
    "bases_convocatoria": {
        "etiqueta": "Bases de Convocatoria",
        "archivo_kb": "bases_convocatoria_g85_full",
        "claves": [
            "convocatoria", "periodo de registro", "registro", "registrar", "registrarme",
            "inscribir", "inscripcion", "folio", "aspirante", "fecha limite", "fecha límite",
            "costo", "gratuito", "gratis", "duracion", "plan de estudios", "modulo propedeutico",
            "módulo propedéutico", "propedeutico", "propedéutico", "generacion", "generación",
            "claves de acceso", "id y contrasena", "contraseña", "contrasena", "documentos",
            "curp", "acta de nacimiento", "certificado de secundaria", "comprobante de domicilio",
            "fotografia tipo credencial", "carta compromiso", "constancia de estudios", "correo",
            "cuestionario socioeconomico", "cuestionario", "no obtengo 60 puntos",
            "sistema operativo movil", "sistema operativo móvil",
        ],
        "tokens": [
            "modulo", "modulos", "duracion", "registro", "inscripcion", "folio", "cuestionario",
            "foto", "fotografia", "documentos", "certificado", "domicilio", "correo", "claves",
            "propedeutico", "procedimiento", "resultados", "generacion", "gratis", "gratuito",
            "plan", "bachillerato", "cupo", "registros", "requisitos", "puntaje", "vigencia",
            "android", "ios", "escane", "escaneados", "peso", "puntos", "sistema", "operativo",
            "movil",
        ],
    },
    "guia_aspirante": {
        "etiqueta": "Guía del Aspirante",
        "archivo_kb": "guia_aspirante_g85_full",
        "claves": [
            "que es prepa en linea", "qué es prepa en línea", "que es la prepa en linea",
            "qué es la prepa en línea", "prepa en linea-sep", "prepa en línea-sep",
            "servicio educativo", "validez oficial", "cobertura nacional", "estatus de registro",
            "folio de aspirante", "mi comunidad", "como me registro", "cómo me registro",
            "pasos para", "proceso de registro", "donde veo los resultados", "resultados",
            "mesa de servicio", "aula virtual",
        ],
        "tokens": [
            "mesa", "servicio", "aula", "virtual", "campus", "estatus", "donde", "portal",
        ],
    },
    "normas_control_escolar": {
        "etiqueta": "Normas de Control Escolar",
        "archivo_kb": "Normas-de-control-escolar-1_full",
        "claves": [
            "control escolar", "calificacion minima aprobatoria", "calificación mínima", "aprobatoria",
            "baja", "estudiante irregular", "regularizacion", "regularización", "recursamiento",
            "revalidar", "revalidacion", "revalidación", "equivalencia", "certificado de estudios",
            "trayectoria academica", "flexibilidad curricular", "creditos", "módulos no acreditados",
            "modulos no acreditados", "baja definitiva", "baja temporal", "promedio", "evaluacion",
            "inscripcion provisional", "inscripción provisional", "retirarse", "estudios previos",
            "bachillerato previo", "acuerdo", "sems", "coordinacion de control escolar",
            "coordinación de control escolar", "same", "sigaprep", "modulos puedo deber",
            "estudios en el extranjero", "extranjero", "extrangero", "limite de edad", "límite de edad",
            "no termino la secundaria",
        ],
        "tokens": [
            "baja", "revalid", "invasion", "nivel", "remedial", "recursamiento", "recuperar",
            "recuperacion", "irregular", "matricula", "asesor", "califica", "calificacion",
            "evalua", "evaluacion", "periodo", "ordinario", "reincorporo", "siged", "dgair",
            "expediente", "trayectoria", "control", "norma", "normas", "acreditados", "modulos",
            "memoria", "quede", "reacer", "acreditacion",
        ],
    },
    "protocolo_convivencia": {
        "etiqueta": "Protocolo de Convivencia",
        "archivo_kb": "Protocolo para la convivencia armonica estudiantado EMS_full",
        "claves": [
            "protocolo", "convivencia", "acoso escolar", "bullying", "hostigamiento escolar",
            "comunidad educativa", "convivencia armonica", "convivencia armónica", "horario escolar",
            "extraescolares", "derechos humanos", "acoso", "hostigamiento",
        ],
        "tokens": [
            "protocolo", "convivencia", "acoso", "hostigamiento", "articulo", "constitucion",
            "derecho", "derechos", "identidad", "datos", "personales", "ciberacoso", "equidad",
            "genero", "cultura", "paz", "dignidad", "violencia", "psicologico", "fisico", "verbal",
            "integracion", "comunidad",
        ],
    },
    "pronunciamiento_cero_tolerancia": {
        "etiqueta": "Cero Tolerancia",
        "archivo_kb": "Pronunciamiento-cero-tolerancia-SEP_18-05-2023_full",
        "claves": [
            "cero tolerancia", "tolerancia", "acoso sexual", "bromas con connotacion sexual",
            "connotacion sexual", "violencia", "discriminacion", "discriminación", "contacto fisico",
            "contacto físico", "obstruccion de justicia", "obstrucción", "rumores", "calumnias",
            "vida sexual", "denunciar", "denuncia", "reportar acoso", "reportar", "correo para reportar",
        ],
        "tokens": [
            "cero", "tolerancia", "sexual", "anatomia", "burlas", "piropos", "denigrante",
            "rumores", "calumnias", "insinuacion", "contacto", "fisico", "exhibicionismo",
            "memes", "connotacion", "chats", "prohibida",
        ],
    },
    "reglas_comunicacion_virtual": {
        "etiqueta": "Reglas de Comunicación Virtual",
        "archivo_kb": "Reglas-de-comunicacion-virtual-PLS_05-12-2023_full",
        "claves": [
            "netiqueta", "reglas de comunicacion", "reglas de comunicación", "comunicacion virtual",
            "comunicación virtual", "foro", "foros", "participacion", "participación", "mayusculas",
            "mayúsculas", "gritar", "ortografia", "ortografía", "imagenes", "imágenes", "pertinencia",
            "breve y conciso", "presentacion", "presentación", "comunicado oficial", "canal oficial",
            "cordialidad", "respeto", "asertiva", "asertividad", "compromisos de comunicacion",
            "compromisos de comunicación",
        ],
        "tokens": [
            "netiqueta", "foro", "foros", "mensaje", "redaccion", "destinatario", "citar",
            "citacion", "mayusculas", "imagenes", "abreviaturas", "modismos", "ortografia",
            "presentacion", "saludo", "comunicacion", "responsabilidad", "digital", "personales",
            "nombre", "aportacion",
        ],
    },
    "decalogo_convivencia": {
        "etiqueta": "Decálogo de Convivencia",
        "archivo_kb": "Construyendo_comunidades_full",
        "claves": [
            "decalogo", "decálogo", "cultura de la paz", "escucha activa", "empatia", "empatía",
            "diversidad", "inclusion", "inclusión", "apoyo", "tutor escolar", "plan nacional de desarrollo",
            "programa sectorial de educacion", "programa sectorial de educación", "paz",
        ],
        "tokens": [
            "decalogo", "escucha", "activa", "empatia", "mentalidad", "crecimiento", "gestion",
            "tiempo", "entorno", "saludo", "responsabilidad", "diversidad", "inclusion", "apoyo",
            "asertiva", "cuidado", "inicial",
        ],
    },
}

STOPWORDS_ES = set(
    """
    que de la el en y a los se del las un por con no una su para es al lo como mas más pero sus le ya o
    este si porque esta estan este esta son fue era muy sin sino todo tambien me te mi tu donde cual como
    cuando quien quienes puedo puede pued hago hacer hay tiene tengo ser estoy estan significa quiere
    deberia seria algo todo nada sobre entre hasta desde antes despues durante horas dello del dia dias
    es el "de" del os la las un una unos unas lo la los las eso esa ese e o u tambien bien mal medio
    """.split()
)

PATRON_FALLBACK = re.compile(
    r"(no encontré información oficial|no encontré informaci[oó]n espec[ií]fica|"
    r"no encontre informacion|no dispongo de informaci[oó]n|no dispongo de la informaci[oó]n|"
    r"no tengo informaci[oó]n|no cuento con informaci[oó]n|no cuenta con informaci[oó]n|"
    r"no logre encontrar|lamento informarte|solo puedo ayudarte con temas|"
    r"no pude encontrar|no se encontro informacion|no se encontró información)",
    re.IGNORECASE,
)

PATRON_PLACEHOLDER = re.compile(
    r"(url\d+|fecha\d+|\bgeneraci[oó]n\s*xx\b|\bxx\b)",
    re.IGNORECASE,
)

PATRON_EMOCIONAL = re.compile(
    r"(no quiero estar|nadie me comprende|deprimid|suicid|emocionalmente|"
    r"no me siento bien|angustia|ansiedad|quiero desaparecer|me siento mal)",
    re.IGNORECASE,
)

PATRON_CONTEXTUAL = re.compile(
    r"^(y|¿y|y si|y si,|entonces|¿ni siquiera|ya |pero |ahora|despues|después|"
    r"¿y cuando|entonce|y el plan|y luego|pero,|si ya|y no)", re.IGNORECASE,
)


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin signos de puntuación para búsquedas."""
    texto = str(texto).lower()
    texto = re.sub(r"¿|¡|\?|!|\.|,|:|;|”|“|\"|'|\(|\)", " ", texto)
    remplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n", "à": "a", "è": "e", "ì": "i", "ò": "o",
    }
    for origen, destino in remplazos.items():
        texto = texto.replace(origen, destino)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def clasificar_documento(pregunta: str) -> str:
    """Asigna la pregunta a una de las 7 categorías de documento por palabras clave."""
    texto = normalizar(pregunta)
    palabras = set(texto.split())
    puntajes = {}
    for categoria, config in DOCUMENTOS.items():
        puntaje = 0
        for clave in config["claves"]:
            if normalizar(clave) in texto:
                puntaje += 3
        for token in config.get("tokens", []):
            if normalizar(token) in palabras:
                puntaje += 1
        puntajes[categoria] = puntaje
    mejor = max(puntajes, key=lambda c: puntajes[c])
    if puntajes[mejor] == 0:
        return "sin_clasificar"
    return mejor


def detectar_fallback(respuesta: str) -> bool:
    return bool(PATRON_FALLBACK.search(str(respuesta)))


def detectar_placeholders(respuesta: str):
    return list(set(m.group(0) for m in PATRON_PLACEHOLDER.finditer(str(respuesta))))


def clasificar_tipo_pregunta(pregunta: str, documento: str = "", kb_hits=None) -> str:
    """emocional | contextual | dentro_del_dominio | fuera_del_dominio."""
    p = str(pregunta).strip()
    if PATRON_EMOCIONAL.search(p):
        return "emocional"
    corta = len(normalizar(p)) < 45
    es_fragmento = corta and documento == "sin_clasificar" and (kb_hits is None or kb_hits["ratio"] < 0.35)
    if PATRON_CONTEXTUAL.match(p) or es_fragmento:
        return "contextual"
    if documento != "sin_clasificar":
        return "dentro_del_dominio"
    if kb_hits and kb_hits["ratio"] >= 0.35:
        return "dentro_del_dominio"
    return "fuera_del_dominio"


def cargar_kb(carpeta: str):
    """Carga los .txt de la base de conocimientos: {archivo: {texto, norm}}."""
    kb = {}
    ruta = Path(carpeta)
    if not ruta.exists():
        return kb
    for txt in sorted(ruta.glob("*.txt")):
        texto = txt.read_text(encoding="utf-8", errors="ignore")
        kb[txt.stem] = {"texto": texto, "norm": normalizar(texto)}
    return kb


def verificar_en_kb(pregunta: str, kb) -> dict:
    """Cuenta cuántos tokens clave de la pregunta aparecen en cada documento KB."""
    tokens = [t for t in normalizar(pregunta).split() if t not in STOPWORDS_ES and len(t) > 3]
    coincidencias = {}
    for doc, contenido in kb.items():
        coincidencias[doc] = sum(1 for t in tokens if t in contenido["norm"])
    total = len(tokens)
    if coincidencias:
        mejor_doc = max(coincidencias, key=lambda d: coincidencias[d])
        if coincidencias[mejor_doc] <= 0:
            mejor_doc = None
    else:
        mejor_doc = None
    ratio = (coincidencias[mejor_doc] / total) if (mejor_doc and total) else 0.0
    evidencia = ", ".join(f"{d}:{c}" for d, c in sorted(coincidencias.items(), key=lambda kv: -kv[1]) if c > 0)
    return {
        "totales_por_doc": coincidencias,
        "mejor_doc": mejor_doc,
        "ratio": ratio,
        "total_tokens": total,
        "evidencia": evidencia,
    }


def inferir_tipo_fallback(kb_hits) -> str:
    """Clasifica el fallback según si el tema está en la KB.

    recuperacion_fallida: la info SÍ existe en la base (no se recuperó) → fallo ACL.
    fuera_del_dominio: el tema no está en la base → rechazo correcto.
    parcial_kb: cobertura parcial o ambigua.
    """
    if not kb_hits or kb_hits["total_tokens"] == 0:
        return "fuera_del_dominio"
    if kb_hits["ratio"] >= 0.50:
        return "recuperacion_fallida"
    if kb_hits["ratio"] >= 0.18:
        return "parcial_kb"
    return "fuera_del_dominio"


def usar_llm(habilitado: bool):
    if not habilitado:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY no configurada: se omite revisión LLM.")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=60.0)
        logger.info("Cliente Groq inicializado. Modelo: %s", MODELO_LLM)
        return client
    except Exception as exc:
        logger.error("No se pudo inicializar Groq: %s", exc)
        return None


def llm_evaluar(client, items):
    """Evalúa los pares pregunta/respuesta en lotes. Retorna {idx: (evaluacion, justificacion)}."""
    resultados = {}
    total_lotes = (len(items) + BATCH_LLM - 1) // BATCH_LLM
    for i in range(0, len(items), BATCH_LLM):
        lote = items[i:i + BATCH_LLM]
        num_lote = i // BATCH_LLM + 1
        payload = {
            "items": [
                {
                    "id": it["idx"],
                    "categoria_documento": it["categoria"],
                    "pregunta": it["pregunta"],
                    "respuesta": it["respuesta"],
                }
                for it in lote
            ]
        }
        prompt = f"""Eres un evaluador experto de un chatbot RAG de "Prepa en Línea SEP" (educación pública mexicana).

Clasifica cada par pregunta/respuesta con una de estas etiquetas:
- "bien": la respuesta es correcta, completa y responde directamente lo que se pregunta.
- "a_medias": la respuesta es parcial (le falta información clave), demasiado genérica, evasiva, o contiene marcadores sin resolver como "Generación XX" o "url1".
- "mal": la respuesta NO responde la pregunta, es incorrecta, inventa datos, o es un fallback tipo "No encontré información oficial".

Responde ÚNICAMENTE JSON con la clave "resultados", una lista de objetos con esta forma:
{{"id": 1, "evaluacion": "bien", "justificacion": "la respuesta explica correctamente los requisitos"}}

Los id corresponden a cada elemento de la lista "items".
Preguntas y respuestas a evaluar:
{json.dumps(payload, ensure_ascii=False, indent=1)}"""
        for intento in range(3):
            try:
                respuesta_raw = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Responde únicamente JSON válido, sin texto adicional."},
                        {"role": "user", "content": prompt},
                    ],
                    model=MODELO_LLM,
                    temperature=0.1,
                    max_tokens=1800,
                    response_format={"type": "json_object"},
                )
                contenido = respuesta_raw.choices[0].message.content
                datos = json.loads(contenido)
                for item in datos["resultados"]:
                    idx = item.get("id")
                    etiqueta = item.get("evaluacion", "").strip().lower()
                    justificacion = item.get("justificacion", "").strip()
                    if etiqueta not in ("bien", "a_medias", "mal"):
                        etiqueta = "a_medias"
                    resultados[idx] = (etiqueta, justificacion)
                logger.info("  Lote %d/%d evaluado (%d ítems).", num_lote, total_lotes, len(lote))
                break
            except Exception as exc:
                msg = str(exc)[:200]
                logger.error("  Lote %d intento %d falló: %s", num_lote, intento + 1, msg)
                if intento < 2:
                    time.sleep(2 ** intento)
                else:
                    logger.error("  Se omite el lote %d (%d ítems) por error de API.", num_lote, len(lote))
        time.sleep(SLEEP_ENTRE_LOTES)
    return resultados


def evaluar_filas(df, client, kb):
    """Devuelve dict {idx: {evaluacion, justificacion, revisar, tipo_pregunta}}."""
    filas = []
    for idx, fila in df.iterrows():
        respuesta = str(fila["respuesta"])
        pregunta = str(fila["pregunta"])
        placeholders = detectar_placeholders(respuesta)
        documento = clasificar_documento(pregunta)
        kb_hits = verificar_en_kb(pregunta, kb)
        tipo = clasificar_tipo_pregunta(pregunta, documento, kb_hits)
        filas.append(
            {
                "idx": idx,
                "pregunta": pregunta,
                "respuesta": respuesta,
                "categoria": documento,
                "fallback": detectar_fallback(respuesta),
                "fallback_tipo": inferir_tipo_fallback(kb_hits) if detectar_fallback(respuesta) else "",
                "kb_evidencia": kb_hits["evidencia"],
                "kb_mejor_doc": kb_hits["mejor_doc"],
                "kb_ratio": round(kb_hits["ratio"], 2),
                "placeholders": placeholders,
                "corta": len(respuesta.strip()) < 45,
                "rag": str(fila["rag"]),
                "tipo_pregunta": tipo,
                "evaluacion": None,
                "justificacion": "",
                "revisar": "",
            }
        )

    pendientes_llm = [f for f in filas if not f["fallback"] and not f["placeholders"]]
    resultados_llm = {}
    if client and pendientes_llm:
        logger.info("Evaluando con LLM %d preguntas en lotes de %d ...", len(pendientes_llm), BATCH_LLM)
        resultados_llm = llm_evaluar(
            client, [{"idx": f["idx"], "pregunta": f["pregunta"], "respuesta": f["respuesta"], "categoria": f["categoria"]} for f in pendientes_llm]
        )
    elif not client:
        logger.info("Sin LLM: se usan solo heurísticas (revisar=si).")
        for f in pendientes_llm:
            f["revisar"] = "si"

    for f in filas:
        if f["fallback"]:
            f["evaluacion"] = "mal"
            f["justificacion"] = "El chatbot respondió que no encontró información (fallback)."
        elif f["placeholders"]:
            f["evaluacion"] = "a_medias"
            f["justificacion"] = "La respuesta muestra un marcador sin resolver (%s)." % ", ".join(f["placeholders"])
        else:
            llm_label, llm_justif = resultados_llm.get(f["idx"], (None, ""))
            if llm_label:
                f["evaluacion"] = llm_label
                f["justificacion"] = llm_justif
            else:
                if f["corta"]:
                    f["evaluacion"] = "a_medias"
                    f["justificacion"] = "Respuesta muy breve para la pregunta (revisar)."
                else:
                    f["evaluacion"] = "bien"
                    f["justificacion"] = "La respuesta atiende la pregunta (revisar)."
                f["revisar"] = "si"
    return {
        f["idx"]: {
            "evaluacion": f["evaluacion"],
            "justificacion": f["justificacion"],
            "revisar": f["revisar"],
            "categoria": f["categoria"],
            "tipo_pregunta": f["tipo_pregunta"],
            "fallback_tipo": f["fallback_tipo"],
            "kb_evidencia": f["kb_evidencia"],
            "kb_mejor_doc": f["kb_mejor_doc"],
            "kb_ratio": f["kb_ratio"],
        }
        for f in filas
    }


def calcular_metricas(df):
    metricas = {
        "total": len(df),
        "evaluacion": Counter(df["evaluacion"]),
        "por_dia": {},
        "por_doc": {},
        "por_tipo": Counter(df["tipo_pregunta"]),
        "fallbacks": [],
        "fallback_por_tipo": Counter(),
        "placeholders": [],
        "mejores": [],
        "peores": [],
        "crisis": [],
        "tiempo_global": {"mediana": df["tiempo_ms"].median(), "p95": df["tiempo_ms"].quantile(0.95)},
        "tokens_global": {"mediana": df["tokens"].median(), "media": df["tokens"].mean()},
        "rag": Counter(df["rag"]),
    }

    for dia, sub in df.groupby(df["fecha"].dt.date):
        n = len(sub)
        ev = Counter(sub["evaluacion"])
        fb = int(sub["fallback"].sum())
        metricas["por_dia"][dia.isoformat()] = {
            "n": n,
            "bien": ev.get("bien", 0),
            "a_medias": ev.get("a_medias", 0),
            "mal": ev.get("mal", 0),
            "fallback": fb,
            "pct_bien": round(ev.get("bien", 0) / n * 100, 1),
            "pct_mal": round(ev.get("mal", 0) / n * 100, 1),
            "tiempo_ms_mediana": sub["tiempo_ms"].median(),
            "tiempo_ms_p95": sub["tiempo_ms"].quantile(0.95),
            "tokens_media": sub["tokens"].mean(),
            "tokens_mediana": sub["tokens"].median(),
            "rag_si": int((sub["rag"] == "Sí").sum()),
            "rag_no": int((sub["rag"] == "No").sum()),
        }

    for doc, sub in df.groupby("documento"):
        n = len(sub)
        ev = Counter(sub["evaluacion"])
        metricas["por_doc"][doc] = {
            "etiqueta": DOCUMENTOS.get(doc, {}).get("etiqueta", "No clasificado"),
            "n": n,
            "bien": ev.get("bien", 0),
            "a_medias": ev.get("a_medias", 0),
            "mal": ev.get("mal", 0),
            "fallback": int(sub["fallback"].sum()),
            "pct_bien": round(ev.get("bien", 0) / n * 100, 1) if n else 0,
            "pct_mal": round(ev.get("mal", 0) / n * 100, 1) if n else 0,
        }

    fb = df[df["fallback"]]
    for _, r in fb.iterrows():
        metricas["fallbacks"].append(
            {
                "fecha": r["fecha"].strftime("%d/%m %H:%M"),
                "pregunta": r["pregunta"],
                "documento": r["documento"],
                "tipo": r.get("tipo_pregunta", ""),
                "fallback_tipo": r.get("fallback_tipo", ""),
                "kb_evidencia": r.get("kb_evidencia", ""),
            }
        )
        metricas["fallback_por_tipo"][r.get("fallback_tipo", "")] += 1

    mark = df[df["marcadores"].apply(bool)]
    for _, r in mark.iterrows():
        metricas["placeholders"].append({"pregunta": r["pregunta"], "respuesta": r["respuesta"]})

    crisis = df[df["tipo_pregunta"] == "emocional"]
    for _, r in crisis.iterrows():
        metricas["crisis"].append(
            {
                "fecha": r["fecha"].strftime("%d/%m %H:%M"),
                "pregunta": r["pregunta"],
                "respuesta": r["respuesta"],
                "evaluacion": r["evaluacion"],
            }
        )

    orden = df.sort_values("tiempo_ms", ascending=False)
    metricas["peores"] = orden.head(3)[["pregunta", "tiempo_ms"]].values.tolist()
    metricas["mejores"] = df.sort_values("tiempo_ms").head(3)[["pregunta", "tiempo_ms"]].values.tolist()

    global_ev = {k: metricas["evaluacion"].get(k, 0) for k in ("bien", "a_medias", "mal")}
    for k, v in list(global_ev.items()):
        global_ev[k + "_pct"] = round(v / metricas["total"] * 100, 1) if metricas["total"] else 0
    metricas["evaluacion_global"] = global_ev
    return metricas


def chart_timg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def etiqueta_dia(iso: str) -> str:
    """2026-09-25 -> '25 sep'."""
    try:
        import datetime
        f = datetime.date.fromisoformat(iso)
        return f"{f.day:02d} {['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'][f.month-1]}"
    except Exception:
        return iso[-5:]


def generar_graficas(metricas):
    COLORES = {"bien": "#28a745", "a_medias": "#f0ad4e", "mal": "#dc3545"}
    AZUL = "#2c3e50"
    graficas = {}

    dias = sorted(metricas["por_dia"].keys())
    etiq = [etiqueta_dia(d) for d in dias]
    pos = list(range(len(dias)))

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ancho = 0.25
    for i, etiqueta in enumerate(["bien", "a_medias", "mal"]):
        vals = [metricas["por_dia"][d][etiqueta] for d in dias]
        ax.bar([p + ancho * (i - 1) for p in pos], vals, width=ancho, label=etiqueta, color=COLORES[etiqueta])
    ax.set_xticks(pos)
    ax.set_xticklabels(etiq)
    for i, d in enumerate(dias):
        for j, etiqueta in enumerate(["bien", "a_medias", "mal"]):
            v = metricas["por_dia"][d][etiqueta]
            if v:
                ax.text(i + ancho * (j - 1), v + 0.15, str(v), ha="center", fontsize=8)
    ax.set_ylabel("Preguntas")
    ax.set_title("Evaluación por día", color=AZUL, fontweight="bold")
    ax.legend(frameon=False)
    graficas["evaluacion_dias"] = chart_timg(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ev = metricas["evaluacion_global"]
    ax.pie(
        [ev["bien"], ev["a_medias"], ev["mal"]],
        labels=["Bien", "A medias", "Mal"],
        autopct="%1.0f%%",
        colors=[COLORES["bien"], COLORES["a_medias"], COLORES["mal"]],
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax.set_title("Evaluación global", color=AZUL, fontweight="bold")
    graficas["evaluacion_global"] = chart_timg(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    docs = [d for d in sorted(metricas["por_doc"], key=lambda x: -metricas["por_doc"][x]["n"]) if metricas["por_doc"][d]["n"] > 0]
    etiquetas = [metricas["por_doc"][d]["etiqueta"] for d in docs]
    totales = [metricas["por_doc"][d]["n"] for d in docs]
    y = list(range(len(docs)))
    prev = [0] * len(docs)
    for etiqueta in ["bien", "a_medias", "mal"]:
        valores = [metricas["por_doc"][d][etiqueta] for d in docs]
        ax.barh(y, valores, left=prev, label=etiqueta, color=COLORES[etiqueta], height=0.6)
        prev = [a + b for a, b in zip(prev, valores)]
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=9)
    ax.invert_yaxis()
    for i in range(len(docs)):
        ax.text(totales[i] + 0.3, y[i], str(totales[i]), va="center", fontsize=8)
    ax.set_xlim(0, max(totales) * 1.15)
    ax.set_xlabel("Preguntas")
    ax.set_title("Evaluación por documento oficial", color=AZUL, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    graficas["evaluacion_docs"] = chart_timg(fig)

    # Tipo de pregunta
    tipos = list(metricas["por_tipo"].keys())
    if tipos:
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        valores = [metricas["por_tipo"][t] for t in tipos]
        ax.bar(range(len(tipos)), valores, color="#3498db")
        for i, v in enumerate(valores):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
        ax.set_xticks(range(len(tipos)))
        ax.set_xticklabels([t.replace("_", " ") for t in tipos], fontsize=8)
        ax.set_ylabel("Preguntas")
        ax.set_title("Tipo de pregunta", color=AZUL, fontweight="bold")
        graficas["tipo_pregunta"] = chart_timg(fig)

    # Fallbacks por tipo
    if metricas["fallback_por_tipo"]:
        etiqueta_fb = {
            "recuperacion_fallida": "Fallo de recuperación\n(info sí está en KB)",
            "fuera_del_dominio": "Fuera de dominio\n(rechazo correcto)",
            "parcial_kb": "Cobertura parcial",
        }
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        claves = list(metricas["fallback_por_tipo"].keys())
        valores = [metricas["fallback_por_tipo"][k] for k in claves]
        colores = {"recuperacion_fallida": "#dc3545", "fuera_del_dominio": "#28a745", "parcial_kb": "#f0ad4e"}
        etiquetas_pie = []
        for k in claves:
            etiquetas_pie.append(etiqueta_fb[k].replace("\n", " ") if k in etiqueta_fb else k)
        ax.pie(
            valores,
            labels=etiquetas_pie,
            autopct="%1.0f%%",
            colors=[colores[k] if k in colores else "#95a5a6" for k in claves],
            startangle=90,
        )
        ax.set_title("Análisis de los fallbacks\n('No encontré información')", color=AZUL, fontweight="bold")
        graficas["fallback_tipos"] = chart_timg(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    mediana = [metricas["por_dia"][d]["tiempo_ms_mediana"] for d in dias]
    p95 = [metricas["por_dia"][d]["tiempo_ms_p95"] for d in dias]
    ax.bar([p - 0.15 for p in pos], mediana, width=0.3, label="Mediana", color="#3498db")
    ax.bar([p + 0.15 for p in pos], p95, width=0.3, label="P95", color="#9b59b6")
    for i, d in enumerate(dias):
        ax.text(i - 0.15, mediana[i] + 15, f"{int(mediana[i])}", ha="center", fontsize=8)
        ax.text(i + 0.15, p95[i] + 15, f"{int(p95[i])}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(etiq)
    ax.set_ylabel("Milisegundos")
    ax.set_title("Latencia de respuesta (ms)", color=AZUL, fontweight="bold")
    ax.legend(frameon=False)
    graficas["latencia_dias"] = chart_timg(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    tokens_media = [metricas["por_dia"][d]["tokens_media"] for d in dias]
    tokens_mediana = [metricas["por_dia"][d]["tokens_mediana"] for d in dias]
    ax.bar([p - 0.15 for p in pos], tokens_media, width=0.3, label="Media", color="#2ecc71")
    ax.bar([p + 0.15 for p in pos], tokens_mediana, width=0.3, label="Mediana", color="#1abc9c")
    for i, d in enumerate(dias):
        ax.text(i - 0.15, tokens_media[i] + 0.6, f"{tokens_media[i]:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.15, tokens_mediana[i] + 0.6, f"{tokens_mediana[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(etiq)
    ax.set_ylabel("Tokens por respuesta")
    ax.set_title("Tokens generados por respuesta", color=AZUL, fontweight="bold")
    ax.legend(frameon=False)
    graficas["tokens_dias"] = chart_timg(fig)

    return graficas


def hallazgos(metricas, contexto):
    d = metricas["por_dia"]
    pares = sorted(d.items())
    lista = []

    desc = " · ".join(f"{etiqueta_dia(f)} {i['pct_mal']}% mal ({i['fallback']} fallbacks)" for f, i in pares)
    total_mal = metricas["evaluacion_global"]["mal"]
    total_bien = metricas["evaluacion_global"]["bien"]
    n_fb = sum(i["fallback"] for i in d.values())
    pct_fb = round(n_fb / metricas["total"] * 100, 1) if metricas["total"] else 0

    tipo_fb = metricas["fallback_por_tipo"]
    n_analizables = sum(tipo_fb.values()) or 1
    recuperacion = tipo_fb.get("recuperacion_fallida", 0)
    fuera = tipo_fb.get("fuera_del_dominio", 0)
    parcial = tipo_fb.get("parcial_kb", 0)

    lista.append(
        f"De {metricas['total']} consultas, el chatbot respondió 'No encontré información oficial' en {n_fb} "
        f"({pct_fb}%); se contaron como 'mal' {total_mal} ({metricas['evaluacion_global']['mal_pct']}%) y como 'bien' {total_bien} ({metricas['evaluacion_global']['bien_pct']}%)."
    )
    if recuperacion or parcial:
        detalle = []
        if recuperacion:
            detalle.append(f"{recuperacion} son fallos de recuperación (la información SÍ existe en la base de conocimientos, pero el retriever no la recuperó)")
        if fuera:
            detalle.append(f"{fuera} son preguntas fuera del dominio del chatbot (rechazo correcto, no inventa)")
        if parcial:
            detalle.append(f"{parcial} tienen cobertura parcial en la KB")
        lista.append(
            "Verificación contra la base de conocimientos (7 documentos): " + "; ".join(detalle) + "."
        )

    doc_peor = max(metricas["por_doc"].values(), key=lambda x: x["mal"]) if metricas["por_doc"] else None
    if doc_peor and doc_peor["mal"] > 0:
        lista.append(
            f"El documento con más respuestas 'mal' es '{doc_peor['etiqueta']}' "
            f"({doc_peor['mal']} de {doc_peor['n']} preguntas, {doc_peor['pct_mal']}%). "
            "Son temas que sí existen en la KB y no se están recuperando bien del vector store."
        )

    n_ctx = metricas["por_tipo"].get("contextual", 0)
    n_emoc = metricas["por_tipo"].get("emocional", 0)
    n_fuera = metricas["por_tipo"].get("fuera_del_dominio", 0)
    if n_ctx:
        lista.append(f"{n_ctx} preguntas son contextuales o encadenadas (dependen de un turno previo) y muchas fallan al responder sin ese contexto.")
    if n_fuera:
        lista.append(f"{n_fuera} preguntas son ajenas al servicio (becas, tareas, recomendaciones, etc.); el chatbot normalmente no debe responderlas inventando.")
    if n_emoc:
        lista.append(
            f"Se detectó 1 situación de tipo emocional/crisis ('No me he sentido bien emocionalmente, ya no quiero estar aquí…'). "
            "El chatbot respondió con un fallback genérico, sin redirigir a ayuda. Se trata como caso aparte en este reporte."
        )

    max_p95 = max(d.values(), key=lambda x: x["tiempo_ms_p95"]) if d else None
    if max_p95 and max_p95["tiempo_ms_p95"] > 2000:
        lista.append(
            f"Latencia: el día con peor P95 fue {etiqueta_dia(max(d.items(), key=lambda kv: kv[1]['tiempo_ms_p95'])[0])} "
            f"con {int(max_p95['tiempo_ms_p95'])} ms; la mediana global es {int(metricas['tiempo_global']['mediana'])} ms."
        )

    if metricas["placeholders"]:
        lista.append(
            "Se detectaron marcadores dinámicos sin resolver en respuestas del chatbot."
        )
    return lista


def recomendaciones(metricas, df=None):
    recs = []
    tipo_fb = metricas["fallback_por_tipo"]
    recuperacion = tipo_fb.get("recuperacion_fallida", 0)
    doc_peor = max(metricas["por_doc"].values(), key=lambda x: x["mal"]) if metricas["por_doc"] else None

    if recuperacion:
        top = [f for f in metricas["fallbacks"] if f["fallback_tipo"] == "recuperacion_fallida"]
        temas = " · ".join(f"{html.escape(f['pregunta'])[:50]}" for f in top[:6])
        recs.append(
            f"{recuperacion} preguntas fallaron por recuperación pese a que la información está en la KB. "
            f"Revisar query expansion, sinónimos y chunks en FAISS. Ejemplos: {temas}"
        )
    if doc_peor and doc_peor["mal"] > 0:
        recs.append(
            f"Ampliar/rebalancear los chunks del documento '{doc_peor['etiqueta']}' y probar el retriever con las preguntas que hoy fallan."
        )
    if metricas["por_tipo"].get("contextual", 0):
        recs.append(
            "Las preguntas encadenadas dependen de la memoria de la sesión. Revisar que ConversationBufferMemory reinyecte el historial "
            "y probar ConversationSummaryMemory para turnos largos."
        )
    if metricas["crisis"]:
        recs.append(
            "Agregar manejo de crisis: detectar mensajes emocionales y responder con empatía y derivación a ayuda (ej. Línea de la Vida), "
            "en lugar de un fallback genérico 'No encontré información'."
        )
    if metricas["por_tipo"].get("fuera_del_dominio", 0):
        recs.append(
            "Definir respuesta estándar elegante para preguntas fuera del dominio (explicar que solo atiende temas de Prepa en Línea-SEP). "
            "Un fallback seco genera fricción, aunque no invente."
        )
    day_lent = max(metricas["por_dia"].values(), key=lambda x: x["tiempo_ms_p95"])
    if day_lent["tiempo_ms_p95"] > 2500:
        recs.append(
            "Optimizar latencia en el día con mayor P95 (reducir tokens en preguntas simples, usar GPT-OSS-20B para triviales)."
        )
    revisar = int((df["revisar"] == "si").sum()) if df is not None and "revisar" in df else 0
    recs.append(
        f"Confirmar/ajustar la columna 'evaluacion' (quedan {revisar} filas marcadas 'revisar') antes de compartir el reporte; "
        "la revisión asistida no es la verdad absoluta."
    )
    return recs


def render_tabla(df):
    filas = []
    for _, r in df.iterrows():
        etiqueta_ev = {"bien": "Bien", "a_medias": "A medias", "mal": "Mal"}.get(r["evaluacion"], r["evaluacion"])
        color = {"bien": "#28a745", "a_medias": "#f0ad4e", "mal": "#dc3545"}.get(r["evaluacion"], "#95a5a6")
        resp = html.escape(r["respuesta"])
        preg = html.escape(r["pregunta"])
        just = html.escape(r["justificacion"]) or "—"
        doc = html.escape(r.get("documento_etiqueta", "")) or "—"
        tipo = html.escape(str(r.get("tipo_pregunta", "")))
        revisar = ' <span class="tag-warn">revisar</span>' if str(r.get("revisar", "")) == "si" else ""
        badge_tipo = f'<span class="tag-tipo">{tipo.replace("_", " ")}</span>' if tipo else ""
        filas.append(f"""
        <div class="fila">
          <span class="col-fecha">{r['fecha'].strftime('%d/%m %H:%M')}</span>
          <span class="col-doc">{doc}</span>
          {badge_tipo}
          <span class="badge" style="background:{color}">{etiqueta_ev}</span>  {revisar}
          <details class="detalle-pregunta">
            <summary><strong>{preg}</strong></summary>
            <div class="respuesta"><h4>Respuesta del chatbot</h4><p>{resp}</p></div>
            <div class="justificacion"><h4>Justificación</h4><p>{just}</p></div>
          </details>
        </div>""")
    return "\n".join(filas)


def graficas_img(clave: str, alt: str, graficas: dict) -> str:
    """Devuelve un <img> con la gráfica embebida o un placeholder amigable."""
    if clave in graficas and graficas[clave]:
        return f'<img src="data:image/png;base64,{graficas[clave]}" alt="{html.escape(alt)}">'
    return f'<p class="muted">Sin datos suficientes para la gráfica «{html.escape(alt)}».</p>'


def construir_html(df, metricas, contexto, parametros):
    graficas = generar_graficas(metricas)
    hallazgos_list = hallazgos(metricas, contexto)
    recs = recomendaciones(metricas, df)

    pasos = {
        "bien": metricas["evaluacion_global"]["bien"],
        "bien_pct": metricas["evaluacion_global"]["bien_pct"],
        "mal": metricas["evaluacion_global"]["mal"],
        "mal_pct": metricas["evaluacion_global"]["mal_pct"],
    }
    n_fb = sum(i["fallback"] for i in metricas["por_dia"].values())
    pct_fb = round(n_fb / metricas["total"] * 100, 1) if metricas["total"] else 0

    tarjetas = f"""
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-value">{metricas['total']}</div><div class="metric-label">Preguntas evaluadas</div></div>
      <div class="metric-card success"><div class="metric-value">{pasos['bien']} ({pasos['bien_pct']}%)</div><div class="metric-label">Respuestas bien</div></div>
      <div class="metric-card warn"><div class="metric-value">{metricas['evaluacion_global']['a_medias']} ({metricas['evaluacion_global']['a_medias_pct']}%)</div><div class="metric-label">A medias</div></div>
      <div class="metric-card error"><div class="metric-value">{pasos['mal']} ({pasos['mal_pct']}%)</div><div class="metric-label">Respuestas mal</div></div>
    </div>
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-value">{metricas['tiempo_global']['mediana']:.0f} ms</div><div class="metric-label">Latencia mediana</div></div>
      <div class="metric-card"><div class="metric-value">{metricas['tiempo_global']['p95']:.0f} ms</div><div class="metric-label">Latencia P95</div></div>
      <div class="metric-card"><div class="metric-value">{metricas['tokens_global']['media']:.0f}</div><div class="metric-label">Tokens medios / respuesta</div></div>
      <div class="metric-card"><div class="metric-value">{n_fb} ({pct_fb}%)</div><div class="metric-label">'No encontré información'</div></div>
    </div>"""

    seccion_dias = ""
    for fecha in sorted(metricas["por_dia"]):
        info = metricas["por_dia"][fecha]
        label = f"{etiqueta_dia(fecha)} · {info['n']} preguntas"
        etiquetas = ["bien", "a_medias", "mal"]
        barra = ""
        for e in etiquetas:
            pct = round(info[e] / info["n"] * 100, 1) if info["n"] else 0
            color = {"bien": "#28a745", "a_medias": "#f0ad4e", "mal": "#dc3545"}[e]
            if info[e] > 0:
                barra += f'<div class="seg" style="background:{color};flex-basis:{pct}%;" title="{e}: {info[e]}">{info[e]}</div>'
        estilo_ok = "var(--verdeclaro)" if info["pct_bien"] >= 70 else ("#f0ad4e" if info["pct_bien"] >= 50 else "var(--rojosoft)")
        seccion_dias += f"""
        <div class="card-dia">
          <h4>{label}</h4>
          <div class="mini-metrics">
            <span class="kpi"><b>{info['pct_bien']}%</b> bien <i style="color:{estilo_ok}">●</i></span>
            <span class="kpi"><b>{info['mal']}</b> mal ({info['pct_mal']}%)</span>
            <span class="kpi"><b>{info['fallback']}</b> fallbacks</span>
            <span class="kpi"><b>{info['tiempo_ms_mediana']:.0f} ms</b> latencia mediana</span>
            <span class="kpi"><b>{info['tokens_media']:.0f}</b> tokens medios</span>
          </div>
          <div class="barra"> {barra} </div>
          <div class="leyenda"><span class="l-bien">Bien</span><span class="l-med">A medias</span><span class="l-mal">Mal</span></div>
        </div>"""

    seccion_docs = ""
    for doc, info in sorted(metricas["por_doc"].items(), key=lambda kv: -kv[1]["n"]):
        if info["n"] == 0:
            continue
        estilo = "var(--verdeclaro)" if info["pct_bien"] >= 70 else ("#f0ad4e" if info["pct_bien"] >= 50 else "var(--rojosoft)")
        seccion_docs += f"""
        <tr>
          <td>{html.escape(info['etiqueta'])}</td>
          <td>{info['n']}</td>
          <td style="color:{estilo};font-weight:700;">{info['pct_bien']}%</td>
          <td>{info['bien']}</td>
          <td>{info['a_medias']}</td>
          <td>{info['mal']}</td>
          <td>{info['fallback']}</td>
        </tr>"""

    tipo_fb = metricas["fallback_por_tipo"]
    etiqueta_fb_lbl = {
        "recuperacion_fallida": ("Fallo de recuperación", "#dc3545"),
        "fuera_del_dominio": ("Fuera de dominio (rechazo correcto)", "#28a745"),
        "parcial_kb": ("Cobertura parcial en KB", "#f0ad4e"),
    }
    seccion_fb_detalle = ""
    for k, (lbl, col) in etiqueta_fb_lbl.items():
        if tipo_fb.get(k):
            seccion_fb_detalle += f'<li><span class="badge" style="background:{col}">{tipo_fb[k]}</span> <b>{lbl}</b></li>'

    seccion_fallbacks = ""
    for f in metricas["fallbacks"][:25]:
        color = etiqueta_fb_lbl.get(f["fallback_tipo"], ("", "#95a5a6"))[1]
        etiq = f["fallback_tipo"].replace("_", " ") if f["fallback_tipo"] else "fallback"
        seccion_fallbacks += (
            f'<li><b>[{f["fecha"]}]</b> {html.escape(f["pregunta"])} '
            f'<span class="muted">({html.escape(f["documento"])})</span> '
            f'<span class="badge" style="background:{color};font-size:10px;">{etiq}</span></li>'
        )

    if metricas["placeholders"]:
        seccion_ph = "<ul>" + "".join(
            f'<li><b>{html.escape(p["pregunta"])}</b> → <code>{html.escape(p["respuesta"][:80])}</code></li>' for p in metricas["placeholders"]
        ) + "</ul>"
    else:
        seccion_ph = "<p class='muted'>Sin marcadores dinámicos sin resolver.</p>"

    seccion_crisis = "<p class='muted'>No se detectaron mensajes de tipo emocional/crisis en el rango analizado.</p>"
    if metricas["crisis"]:
        items = "".join(
            f'<li><b>[{c["fecha"]}]</b> {html.escape(c["pregunta"])}<br><span class="muted">El chatbot respondió:</span> '
            f'<code>{html.escape(c["respuesta"][:120])}</code> <span class="badge error">trato inapropiado</span></li>'
            for c in metricas["crisis"]
        )
        seccion_crisis = f"<ul class='fallback-ul'>{items}</ul>"

    hallazgos_html = "".join(f"<li>{h}</li>" for h in hallazgos_list)
    recs_html = "".join(f"<li>{r}</li>" for r in recs)
    tabla = render_tabla(df)

    script_js = r"""
    const tabla = document.getElementById('tabla-preguntas');
    const filtroDoc = document.getElementById('filtro-doc');
    const filtroEv = document.getElementById('filtro-ev');
    const filtroTipo = document.getElementById('filtro-tipo');
    const busqueda = document.getElementById('busqueda');
    const contador = document.getElementById('contador');
    function aplicarFiltros(){
      const doc = filtroDoc.value, ev = filtroEv.value, tipo = filtroTipo.value, q = busqueda.value.toLowerCase();
      let vis = 0;
      tabla.querySelectorAll(':scope > .fila').forEach(f => {
        const d = f.querySelector('.col-doc').textContent;
        const t = f.querySelector('.tag-tipo') ? f.querySelector('.tag-tipo').textContent : '';
        const b = f.querySelector('.badge').textContent;
        const hay = f.textContent.toLowerCase();
        const okDoc = !doc || d === doc;
        const okTipo = !tipo || t.indexOf(tipo) !== -1;
        const okEv = !ev || b === ev;
        const okQ = !q || hay.indexOf(q) !== -1;
        const show = okDoc && okTipo && okEv && okQ;
        f.style.display = show ? '' : 'none';
        if (show) vis++;
      });
      contador.textContent = vis + ' de ' + tabla.querySelectorAll(':scope > .fila').length + ' preguntas';
    }
    filtroDoc.onchange = aplicarFiltros;
    filtroEv.onchange = aplicarFiltros;
    filtroTipo.onchange = aplicarFiltros;
    busqueda.oninput = aplicarFiltros;
    aplicarFiltros();
    """

    opciones_docs = "".join(
        f'<option value="{html.escape(info["etiqueta"])}">{html.escape(info["etiqueta"])}</option>'
        for doc, info in sorted(metricas["por_doc"].items(), key=lambda kv: -kv[1]["n"]) if info["n"] > 0
    )
    opciones_tipo = "".join(
        f'<option value="{html.escape(t.replace("_", " "))}">{html.escape(t.replace("_", " "))}</option>'
        for t in sorted(metricas["por_tipo"].keys()) if t
    )

    titulo = contexto.get("titulo", "Analítica del Chatbot RAG — Prepa en Línea SEP")
    subtitulo = contexto.get("subtitulo", "")
    html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titulo}</title>
<style>
  :root {{
    --azul-principal: #2c3e50; --azul-secundario: #3498db; --verdeclaro: #2ecc71;
    --rojosoft: #e74c3c; --blanco: #ffffff; --gris: #7f8c8d;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: var(--azul-principal); }}
  .contenedor {{ max-width: 1220px; margin: 0 auto; padding: 24px 20px 60px; }}
  header {{ padding: 10px 0 4px; }}
  h1 {{ font-size: 26px; color: var(--azul-principal); }}
  .subtitulo {{ color: var(--gris); margin-top: 6px; font-size: 14px; }}
  .badges {{ margin-top: 12px; }}
  .chip {{ display: inline-block; background: #ecf0f1; color: var(--azul-principal); border-radius: 20px; padding: 4px 12px; font-size: 12px; margin-right: 8px; }}
  .seccion {{ background: var(--blanco); border-radius: 12px; padding: 22px 24px; margin-top: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
  .seccion h2 {{ font-size: 19px; margin-bottom: 14px; color: var(--azul-principal); border-left: 4px solid var(--azul-secundario); padding-left: 10px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 18px; }}
  .metric-card {{ background: var(--blanco); padding: 18px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); text-align: center; }}
  .metric-value {{ font-size: 30px; font-weight: 800; color: var(--azul-principal); }}
  .metric-label {{ color: var(--gris); font-size: 12px; margin-top: 4px; text-transform: uppercase; letter-spacing: .4px; }}
  .metric-card.success .metric-value {{ color: var(--verdeclaro); }}
  .metric-card.error .metric-value {{ color: var(--rojosoft); }}
  .metric-card.warn .metric-value {{ color: #f0ad4e; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .grid-2 img {{ width: 100%; border-radius: 10px; }}
  .resumen li, .reco li {{ margin-bottom: 12px; line-height: 1.6; }}
  .resumen li::marker, .reco li::marker {{ color: var(--azul-secundario); }}
  .card-dia {{ border: 1px solid #ecf0f1; border-radius: 10px; padding: 16px; margin-bottom: 12px; }}
  .card-dia h4 {{ margin-bottom: 8px; }}
  .n {{ color: var(--gris); font-weight: 400; }}
  .mini-metrics {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 10px; }}
  .kpi {{ font-size: 13px; color: var(--azul-principal); }}
  .barra {{ display: flex; height: 26px; border-radius: 6px; overflow: hidden; width: 100%; background: #ecf0f1; }}
  .seg {{ display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; min-width: 22px; }}
  .leyenda {{ display: flex; gap: 18px; margin-top: 6px; font-size: 12px; color: var(--gris); }}
  .l-bien::before {{ content: '▮'; color: #28a745; margin-right: 4px; }}
  .l-med::before {{ content: '▮'; color: #f0ad4e; margin-right: 4px; }}
  .l-mal::before {{ content: '▮'; color: #dc3545; margin-right: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: var(--azul-principal); color: var(--blanco); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }}
  th, td {{ padding: 11px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
  .badge {{ color: var(--blanco); padding: 3px 11px; border-radius: 16px; font-size: 12px; font-weight: 700; display: inline-block; }}
  .tag-warn {{ background: #fef3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 12px; font-size: 10px; padding: 1px 8px; }}
  .tag-tipo {{ background: #eaf3fb; color: var(--azul-secundario); border: 1px solid #d1e6f5; border-radius: 12px; font-size: 10px; padding: 1px 8px; white-space: nowrap; }}
  .fila {{ display: flex; align-items: baseline; gap: 8px; padding: 10px 6px; border-bottom: 1px solid #f0f3f6; }}
  .col-fecha {{ width: 82px; font-size: 12px; color: var(--gris); flex-shrink: 0; }}
  .col-doc {{ width: 150px; font-size: 12px; color: var(--azul-secundario); flex-shrink: 0; font-weight: 600; }}
  details.detalle-pregunta {{ flex: 1; min-width: 0; }}
  details summary {{ cursor: pointer; font-size: 13.5px; color: var(--azul-principal); }}
  details summary strong {{ line-height: 1.5; }}
  .respuesta, .justificacion {{ background: #f8fafc; border-left: 3px solid var(--azul-secundario); border-radius: 4px; padding: 12px; margin-top: 10px; }}
  .respuesta h4, .justificacion h4 {{ font-size: 12px; text-transform: uppercase; color: var(--gris); margin-bottom: 6px; }}
  .respuesta p {{ white-space: pre-wrap; line-height: 1.55; font-size: 13px; }}
  .filtros {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; align-items: center; }}
  .filtros select, .filtros input {{ padding: 8px 10px; border: 1px solid #dcdfe4; border-radius: 8px; font-size: 13px; }}
  .contador {{ font-size: 13px; color: var(--gris); }}
  .fallback-ul li {{ margin-bottom: 8px; font-size: 13.5px; }}
  .muted {{ color: var(--gris); font-size: 12.5px; }}
  code {{ background: #f8fafc; border-radius: 4px; padding: 2px 6px; font-size: 12px; }}
  footer {{ margin-top: 34px; color: var(--gris); font-size: 12.5px; text-align: center; }}
  @media (max-width: 760px) {{ .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }} .grid-2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="contenedor">
  <header>
    <h1>📊 {titulo}</h1>
    <p class="subtitulo">{subtitulo}</p>
    <div class="badges"><span class="chip">🗓️ {contexto['fechas_texto']}</span><span class="chip">🧪 {metricas['total']} consultas</span><span class="chip">📝 Fuente: {html.escape(contexto['ruta_origen'])}</span><span class="chip">🤖 Revisión: {html.escape(parametros['llm'])}</span></div>
  </header>

  <div class="seccion">
    <h2>Resumen ejecutivo</h2>
    {tarjetas}
  </div>

  <div class="seccion">
    <h2>Hallazgos clave</h2>
    <ul class="resumen">
    {hallazgos_html}
    </ul>
  </div>

  <div class="seccion">
    <h2>Comparativa por día</h2>
    <div class="grid-2">
      <div>{seccion_dias}</div>
      <div><img src="data:image/png;base64,{graficas['evaluacion_dias']}" alt="Evaluación por día"></div>
    </div>
  </div>

  <div class="seccion">
    <h2>Desempeño por documento oficial</h2>
    <div class="grid-2">
      <table>
        <thead><tr><th>Documento</th><th>Preg.</th><th>% Bien</th><th>Bien</th><th>A medias</th><th>Mal</th><th>Fallback</th></tr></thead>
        <tbody>{seccion_docs}</tbody>
      </table>
      <img src="data:image/png;base64,{graficas['evaluacion_docs']}" alt="Evaluación por documento">
    </div>
  </div>

  <div class="seccion">
    <h2>Tipo de pregunta</h2>
    <div class="grid-2">
      <div>{graficas_img('tipo_pregunta', 'Tipo de pregunta', graficas)}</div>
      <div style="line-height:1.7;font-size:14px;">
        <p><b>dentro_del_dominio:</b> la pregunta pertenece a los 7 documentos oficiales.<br>
        <b>fuera_del_dominio:</b> ajena al servicio (tareas, recomendaciones, curiosidades).<br>
        <b>contextual:</b> pregunta encadenada que depende de un turno previo.<br>
        <b>emocional:</b> mensaje con carga emocional o de crisis.</p>
      </div>
    </div>
  </div>

  <div class="seccion">
    <h2>¿Por qué el chatbot dijo "No encontré información"?</h2>
    <div class="grid-2">
      <div>{graficas_img('fallback_tipos', 'Análisis de fallbacks', graficas)}</div>
      <div>
        <p style="line-height:1.7;font-size:14px;margin-bottom:10px;">Cada respuesta con fallback se contrastó con los 7 documentos de la base de conocimientos.</p>
        <ul class="fallback-ul">{seccion_fb_detalle}</ul>
      </div>
    </div>
  </div>

  <div class="seccion">
    <h2>Latencia y consumo de tokens</h2>
    <div class="grid-2">
      <div>{graficas_img('latencia_dias', 'Latencia por día', graficas)}</div>
      <div>{graficas_img('tokens_dias', 'Tokens por día', graficas)}</div>
    </div>
  </div>

  <div class="seccion">
    <h2>Distribución global de evaluación</h2>
    <div class="grid-2">
      <div>{graficas_img('evaluacion_global', 'Evaluación global', graficas)}</div>
      <div style="line-height:1.7;font-size:14px;">
        <p>El chatbot entregó una respuesta satisfactoria (<b>Bien</b>) en el <b>{pasos['bien_pct']}%</b> de las consultas.
        Un <b>{metricas['evaluacion_global']['a_medias_pct']}%</b> quedó a medias (parcial, genérica o con marcadores sin resolver) y
        <b>{pasos['mal_pct']}%</b> falló. Del total de fallbacks ({n_fb}, {pct_fb}%), una parte corresponde a
        preguntas ajenas al servicio (rechazo razonable) y la otra a temas que sí están en la base de conocimientos pero no se recuperaron.</p>
      </div>
    </div>
  </div>

  <div class="seccion">
    <h2>Marcadores dinámicos sin resolver</h2>
    {seccion_ph}
  </div>

  <div class="seccion">
    <h2>Caso especial: mensaje emocional / crisis</h2>
    {seccion_crisis}
  </div>

  <div class="seccion">
    <h2>Ejemplos de preguntas con "No encontré información"</h2>
    <ul class="fallback-ul">{seccion_fallbacks}</ul>
    <p class="muted">Se muestran hasta 25 de {len(metricas['fallbacks'])} fallbacks del rango.</p>
  </div>

  <div class="seccion">
    <h2>Recomendaciones</h2>
    <ul class="reco">{recs_html}</ul>
  </div>

  <div class="seccion">
    <h2>Detalle por pregunta ({metricas['total']})</h2>
    <div class="filtros">
      <select id="filtro-doc"><option value="">Todos los documentos</option>{opciones_docs}</select>
      <select id="filtro-tipo"><option value="">Toda tipo</option>{opciones_tipo}</select>
      <select id="filtro-ev"><option value="">Toda evaluación</option><option>Bien</option><option>A medias</option><option>Mal</option></select>
      <input id="busqueda" type="text" placeholder="Buscar palabra en pregunta/respuesta...">
      <span class="contador" id="contador"></span>
    </div>
    <div id="tabla-preguntas">{tabla}</div>
  </div>

  <footer>
    Generado automáticamente por <code>scripts/analisis_stress_test.py</code> · Verificación de fallbacks contra la base de conocimientos ·
    Las filas marcadas <span class="tag-warn">revisar</span> requieren confirmación humana.
  </footer>
</div>
<script>{script_js}</script>
</body>
</html>"""
    return html_out


def aplicar_revision_manual(df, ruta_json):
    """Sobrescribe evaluacion/justificacion/revisar/tipo_pregunta con revisión manual (JSON)."""
    ruta = Path(ruta_json)
    if not ruta.exists():
        logger.warning("No existe %s; no se aplica revisión manual.", ruta)
        return df
    with ruta.open(encoding="utf-8") as fh:
        datos = json.load(fh)
    if not isinstance(datos, dict):
        logger.error("Revisión manual con formato inválido (se ignora).")
        return df
    aplicadas = 0
    for idx_str, rev in datos.items():
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            continue
        if idx not in df.index:
            continue
        ev = str(rev.get("evaluacion", "")).strip().lower()
        if ev in ("bien", "a_medias", "mal"):
            df.at[idx, "evaluacion"] = ev
            df.at[idx, "justificacion"] = str(rev.get("justificacion", "")).strip()
            df.at[idx, "revisar"] = ""
            aplicadas += 1
        if "tipo_pregunta" in rev:
            tp = str(rev["tipo_pregunta"]).strip().lower()
            if tp:
                df.at[idx, "tipo_pregunta"] = tp
        if "fallback_tipo" in rev:
            ft = str(rev["fallback_tipo"]).strip().lower()
            if ft:
                df.at[idx, "fallback_tipo"] = ft
    logger.info("Revisión manual aplicada a %d filas.", aplicadas)
    return df


def main():
    parser = argparse.ArgumentParser(description="Analítica de conversaciones del chatbot RAG.")
    parser.add_argument("--csv", default="Reportes/Preguntas_analisis/consulta-conversations-2026-09-16.csv", help="CSV de entrada")
    parser.add_argument("--salida", default="Reportes/Preguntas_analisis", help="Carpeta de salida")
    parser.add_argument("--desde", default="2026-09-12", help="Fecha inicial (YYYY-MM-DD)")
    parser.add_argument("--hasta", default="2026-09-13", help="Fecha final (YYYY-MM-DD)")
    parser.add_argument("--kb", default="Reportes/Base de conocimientos", help="Carpeta con la base de conocimientos (.txt)")
    parser.add_argument("--sin-llm", action="store_true", help="Solo heurísticas, sin revisión LLM")
    parser.add_argument("--manuales", default="", help="JSON opcional con revisión manual por índice")
    parser.add_argument("--titulo", default="", help="Título del reporte (HTML y consola)")
    parser.add_argument("--subtitulo", default="", help="Subtítulo descriptivo del reporte")
    args = parser.parse_args()

    ruta_csv = Path(args.csv)
    ruta_salida = Path(args.salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)
    desde = pd.Timestamp(args.desde)
    hasta = pd.Timestamp(args.hasta)
    fechas_objetivo = set(pd.date_range(desde, hasta).date)

    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df = df[df["fecha"].dt.date.isin(fechas_objetivo)].copy().sort_values("fecha").reset_index(drop=True)
    logger.info("Filas en rango %s - %s: %d", args.desde, args.hasta, len(df))
    if len(df) == 0:
        logger.error("No hay filas en el rango indicado.")
        return 1

    df["tiempo_ms"] = df["tiempo"].astype(str).str.replace("ms", "", regex=False).astype(int)
    df["fallback"] = df["respuesta"].apply(detectar_fallback)
    df["marcadores"] = df["respuesta"].apply(detectar_placeholders)

    kb = cargar_kb(args.kb)
    logger.info("Base de conocimientos cargada: %d documentos", len(kb))
    if not kb:
        logger.warning("No se encontraron .txt en %s: la verificación de fallbacks usará solo heurísticas.", args.kb)

    client = usar_llm(not args.sin_llm)
    evaluacion = evaluar_filas(df, client, kb)
    df["evaluacion"] = df.index.map(lambda i: evaluacion[i]["evaluacion"])
    df["justificacion"] = df.index.map(lambda i: evaluacion[i]["justificacion"])
    df["revisar"] = df.index.map(lambda i: evaluacion[i]["revisar"])
    df["documento"] = df.index.map(lambda i: evaluacion[i]["categoria"])
    df["documento_etiqueta"] = df["documento"].map(lambda c: DOCUMENTOS.get(c, {}).get("etiqueta", "No clasificado"))
    df["tipo_pregunta"] = df.index.map(lambda i: evaluacion[i]["tipo_pregunta"])
    df["fallback_tipo"] = df.index.map(lambda i: evaluacion[i]["fallback_tipo"])
    df["kb_evidencia"] = df.index.map(lambda i: evaluacion[i]["kb_evidencia"])
    df["kb_mejor_doc"] = df.index.map(lambda i: evaluacion[i]["kb_mejor_doc"])
    df["kb_ratio"] = df.index.map(lambda i: evaluacion[i]["kb_ratio"])

    if args.manuales:
        df = aplicar_revision_manual(df, args.manuales)
    df["revisar"] = df["revisar"].fillna("")

    metricas = calcular_metricas(df)

    nombre_base = ruta_csv.stem
    ruta_csv_salida = ruta_salida / f"{nombre_base}_analizado.csv"
    columnas_salida = [
        "fecha", "pregunta", "respuesta", "tiempo", "tokens", "rag",
        "documento", "tipo_pregunta", "evaluacion", "justificacion",
        "fallback_tipo", "kb_evidencia", "kb_mejor_doc", "kb_ratio", "revisar",
    ]
    df[columnas_salida].to_csv(ruta_csv_salida, index=False, encoding="utf-8-sig")
    logger.info("CSV con evaluación guardado: %s", ruta_csv_salida)

    m = re.search(r"(\d{4}-\d{2}-\d{2})", nombre_base)
    fecha_html = m.group(1) if m else desde.strftime("%Y-%m-%d")
    ruta_html = ruta_salida / f"reporte_analitica_chatbot_{fecha_html}.html"

    titulo = args.titulo or f"Analítica del Chatbot RAG ({etiqueta_dia(desde.strftime('%Y-%m-%d'))} - {etiqueta_dia(hasta.strftime('%Y-%m-%d'))})"
    n_dias = (hasta - desde).days + 1
    subtitulo = args.subtitulo or (
        f"Conversaciones del {desde.strftime('%d/%m/%Y')} al {hasta.strftime('%d/%m/%Y')} "
        f"({n_dias} días, {metricas['total']} preguntas). Evaluación bien/a_medias/mal con verificación de fallbacks "
        f"contra la base de conocimientos ({len(kb)} documentos)."
    )
    contexto = {
        "titulo": titulo,
        "subtitulo": subtitulo,
        "fechas_texto": f"{desde.strftime('%d/%m/%Y')} - {hasta.strftime('%d/%m/%Y')}",
        "ruta_origen": ruta_csv.name,
    }
    parametros = {"llm": MODELO_LLM if client else "heurísticas"}
    if args.manuales:
        parametros["llm"] = "revisión manual"
    html_out = construir_html(df, metricas, contexto, parametros)
    ruta_html.write_text(html_out, encoding="utf-8")
    logger.info("Reporte HTML guardado: %s", ruta_html)

    print(f"\n=========== RESUMEN {titulo} ===========")
    print(f"✅ Preguntas evaluadas: {metricas['total']}")
    ev = metricas["evaluacion_global"]
    print(f"✅ Bien: {ev['bien']} ({ev['bien_pct']}%) | A medias: {ev['a_medias']} ({ev['a_medias_pct']}%) | Mal: {ev['mal']} ({ev['mal_pct']}%)")
    for fecha in sorted(metricas["por_dia"]):
        info = metricas["por_dia"][fecha]
        print(f"   📅 {fecha}: bien {info['pct_bien']}% | mal {info['pct_mal']}% | fallback {info['fallback']} | P95 {int(info['tiempo_ms_p95'])}ms")
    print(f"   Tipo: {dict(metricas['por_tipo'])}")
    if metricas["fallback_por_tipo"]:
        print(f"   Fallbacks por tipo: {dict(metricas['fallback_por_tipo'])}")
    peor_doc = max(metricas["por_doc"].values(), key=lambda x: x["mal"])
    print(f"📚 Documento con más fallos: {peor_doc['etiqueta']} ({peor_doc['mal']} de {peor_doc['n']})")
    if metricas["placeholders"]:
        print(f"⚠️ Placeholders sin resolver: {len(metricas['placeholders'])}")
    if metricas["crisis"]:
        print(f"🚨 Mensaje emocional/crisis detectado: {len(metricas['crisis'])} (caso aparte en el reporte)")
    print("=======================================================================")
    print(f"📁 CSV: {ruta_csv_salida}")
    print(f"📄 HTML: {ruta_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())