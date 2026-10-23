"""Replay de evaluación: reenvía preguntas históricas al chatbot local.

Lee un CSV exportado de conversaciones (``fecha, pregunta, respuesta, tiempo,
tokens, rag``), toma las preguntas del rango de fechas indicado y las vuelve a
ejecutar contra el sistema RAG **local**, respetando una cuota de
``--por-minuto`` (por defecto 8, para no superar el límite de seguridad de
menos de 9 consultas por minuto).

Salida: un CSV nuevo con el MISMO esquema del original (``fecha, pregunta,
respuesta, tiempo, tokens, rag``) más una columna ``session_id`` de replay.
Ese CSV se analiza después con ``scripts/analisis_stress_test.py``.

Uso típico (en un entorno con el modelo y la API de Groq funcionando):
    python scripts/replay_evaluacion.py \
        --csv Reportes/Preguntas_equipo/consulta-conversations-2026-09-29.csv \
        --salida Reportes/Preguntas_equipo/consulta-conversations-2026-09-29_post.csv \
        --desde 2026-09-25 --hasta 2026-09-29 --por-minuto 8

    python scripts/replay_evaluacion.py --dry-run   # valida sin llamar al modelo
"""
import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Desactivar TensorFlow para transformers ANTES de cualquier import del RAG.
# El TF 2.18 instalado está corrupto y el proyecto usa PyTorch; sin esto,
# sentence-transformers falla al importar.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Permitir importar el paquete raíz (rag, models, ...) al ejecutar desde scripts/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cargar variables de entorno (.env) antes de inicializar el RAG / Groq.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("replay_evaluacion")


def cargar_preguntas(ruta_csv: str, desde: str, hasta: str, limite: int = 0):
    """Carga y filtra las preguntas a repetir, ordenadas por fecha."""
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["fecha"])
    objetivo = set(pd.date_range(pd.Timestamp(desde), pd.Timestamp(hasta)).date)
    df = df[df["fecha"].dt.date.isin(objetivo)].sort_values("fecha").reset_index(drop=True)
    if limite:
        df = df.head(limite)
    return df


def segmentar_sesiones(df, gap_min: int = 30):
    """Agrupa mensajes en sesiones usando el hueco de tiempo entre ellos.

    Sin un ``session_id`` en el CSV, dos mensajes consecutivos se consideran de
    la misma conversación si distan menos de ``gap_min`` minutos.
    """
    sesiones = []
    actual = 0
    prev = None
    for _, r in df.iterrows():
        if prev is not None and (r["fecha"] - prev).total_seconds() > gap_min * 60:
            actual += 1
        sesiones.append(f"replay-{actual:03d}")
        prev = r["fecha"]
    return sesiones


def es_respuesta_error(texto: str) -> bool:
    """Detecta respuestas que en realidad son errores (429/mantenimiento)."""
    t = str(texto or "").lower()
    if not t.strip():
        return True
    marcas = (
        "problema procesando",
        "modo de mantenimiento",
        "error_replay",
        "por favor intenta de nuevo",
    )
    return any(m in t for m in marcas)


async def ejecutar_replay(df, ruta_salida, por_minuto, gap_min, dry_run, guardar_mongodb=False, reusar=False):
    """Ejecuta el replay contra el RAG local y guarda el CSV resultante.

    Con ``reusar=True`` conserva las respuestas válidas ya presentes en
    ``ruta_salida`` y solo re-ejecuta las que quedaron con error (útil para
    reanudar tras agotar la cuota diaria de Groq).
    """
    if dry_run:
        df = df.copy()
        df["respuesta"] = "[dry-run] no se llamó al modelo"
        df["tiempo"] = "0ms"
        df["tokens"] = 0
        df["rag"] = "No"
        df["session_id"] = segmentar_sesiones(df, gap_min)
        columnas = ["fecha", "pregunta", "respuesta", "tiempo", "tokens", "rag", "session_id"]
        out = df[columnas].copy()
        out["fecha"] = out["fecha"].dt.strftime("%d/%m/%Y %H:%M")
        out.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
        logger.info("[dry-run] Plan de replay guardado en %s (%d preguntas)", ruta_salida, len(out))
        return

    # Import perezoso: evita cargar el modelo en modo --dry-run.
    from rag.core import RAGSystem
    from langchain_layer.wrappers import LangChainRAGWrapper

    # Respuestas reutilizables de una ejecución previa.
    reuso = {}
    if reusar and Path(ruta_salida).exists():
        prev = pd.read_csv(ruta_salida, encoding="utf-8-sig").reset_index(drop=True)
        for i, r in prev.iterrows():
            if not es_respuesta_error(r.get("respuesta", "")):
                reuso[i] = r
        logger.info("Reanudando: %d respuestas válidas reutilizadas, %d por ejecutar",
                    len(reuso), len(df) - len(reuso))

    logger.info("Inicializando RAGSystem y wrapper...")
    rag = RAGSystem()
    wrapper = LangChainRAGWrapper(rag, memory_enabled=True, mongodb_enabled=guardar_mongodb)

    sesiones = segmentar_sesiones(df, gap_min)
    espera = 60.0 / max(1, por_minuto)
    filas = []
    total = len(df)

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        idx0 = i - 1
        pregunta = str(r["pregunta"])
        sesion = sesiones[i - 1]

        # Reutilizar respuesta válida previa (sin llamar al modelo).
        if idx0 in reuso and str(reuso[idx0]["pregunta"]) == pregunta:
            fila = reuso[idx0].to_dict()
            fila["session_id"] = sesion
            filas.append(fila)
            # Reinyectar en memoria local para no perder el contexto del turno.
            try:
                wrapper._session_memories[sesion].save_context(
                    {"input": pregunta}, {"output": str(fila.get("respuesta", ""))}
                )
            except Exception:
                pass
            logger.info("♻️ %d/%d reutilizada: %s", i, total, pregunta[:60])
            continue

        t0 = time.time()
        try:
            res = await wrapper.query_with_memory(
                question=pregunta,
                session_id=sesion,
                user_id="replay",
                conversation_id=sesion,
            )
            respuesta = res.get("response", "")
            is_rag = res.get("is_rag_response", False)
            latency_ms = int(res.get("latency_ms", (time.time() - t0) * 1000))
        except Exception as e:
            logger.error("Error en pregunta %d/%d: %s", i, total, e)
            respuesta = "ERROR_REPLAY"
            is_rag = False
            latency_ms = int((time.time() - t0) * 1000)

        filas.append({
            "fecha": r["fecha"].strftime("%d/%m/%Y %H:%M"),
            "pregunta": pregunta,
            "respuesta": respuesta,
            "tiempo": f"{latency_ms}ms",
            "tokens": max(1, len(respuesta) // 4),
            "rag": "Sí" if is_rag else "No",
            "session_id": sesion,
        })
        logger.info("📝 %d/%d [%s] %s", i, total, "RAG" if is_rag else "directo", pregunta[:60])

        # Throttle: respeta la cuota (<9 por minuto).
        if i < total:
            time.sleep(espera)

    pd.DataFrame(filas).to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    logger.info("Replay completado: %s (%d filas)", ruta_salida, len(filas))


def main():
    parser = argparse.ArgumentParser(description="Replay de preguntas contra el chatbot local.")
    parser.add_argument("--csv", default="Reportes/Preguntas_equipo/consulta-conversations-2026-09-29.csv")
    parser.add_argument("--salida", default="Reportes/Preguntas_equipo/consulta-conversations-2026-09-29_post.csv")
    parser.add_argument("--desde", default="2026-09-25")
    parser.add_argument("--hasta", default="2026-09-29")
    parser.add_argument("--por-minuto", type=int, default=8, help="Máximo de preguntas por minuto (cuota)")
    parser.add_argument("--sesion-gap-min", type=int, default=30, help="Minutos para separar sesiones")
    parser.add_argument("--limite", type=int, default=0, help="Procesar solo las primeras N (pruebas)")
    parser.add_argument("--dry-run", action="store_true", help="No llama al modelo; solo escribe el plan")
    parser.add_argument("--guardar-mongodb", action="store_true", help="Persistir el replay en MongoDB (por defecto NO)")
    parser.add_argument("--reusar", action="store_true", help="Reutilizar respuestas válidas del CSV de salida y re-ejecutar solo las fallidas")
    args = parser.parse_args()

    df = cargar_preguntas(args.csv, args.desde, args.hasta, args.limite)
    if len(df) == 0:
        logger.error("No hay preguntas en el rango indicado.")
        return 1
    logger.info("Preguntas a repetir: %d (rango %s - %s)", len(df), args.desde, args.hasta)

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(ejecutar_replay(df, args.salida, args.por_minuto, args.sesion_gap_min, args.dry_run, args.guardar_mongodb, args.reusar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
