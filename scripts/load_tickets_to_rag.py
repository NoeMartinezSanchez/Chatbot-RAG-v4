#!/usr/bin/env python3
"""
Cargador y registrador de tickets de mesa de servicio.

Lee el Excel de preguntas frecuentes que el departamento de mesa de servicio
revisó, asigna a cada fila un ``ticket_id`` DETERMINÍSTICO e idempotente
(mismo Excel → mismos IDs), y genera:

1. **Registro de trazabilidad** ``data/ticket_registry.jsonl`` — un "libro
   mayor" versionado con: ticket_id, source_file generado, categoría, asunto,
   hash de la respuesta y si ya fue cargado al RAG (``en_faiss``).

2. **Chunks RAG listos** ``data/vector_store/tickets_chunks.jsonl`` — en el
   formato que consume ``scripts/load_chunks_to_rag.py`` (por si en el futuro
   se desea reindexar FAISS). Cada chunk lleva el ``ticket_id`` en su
   metadata para trazabilidad por respuesta en el dashboard.

NO modifica FAISS: solo genera/actualiza el registro y los chunks preparados.

Uso:
    python scripts/load_tickets_to_rag.py
    python scripts/load_tickets_to_rag.py --check          # solo inspección
    python scripts/load_tickets_to_rag.py --xlsx <ruta>
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ruta por defecto (carpeta Reportes NO está versionada en git).
DEFAULT_XLSX = os.path.join("Reportes", "Tickets mesa de servicio", "Preguntas_frecuentes_sin_duplicados.xlsx")
REGISTRY_FILE = os.path.join("data", "ticket_registry.jsonl")
CHUNKS_FILE = os.path.join("data", "vector_store", "tickets_chunks.jsonl")

TICKET_ID_PREFIX = "TKT-MESA-"


def _slugify(name: str) -> str:
    """Convierte un nombre de hoja/categoría a slug snake_case.

    Args:
        name: Nombre original (ej: "Aspirante no cuenta con cert").

    Returns:
        Slug en minúsculas con guiones bajos y sin acentos.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.strip().lower().replace(" ", "_")
    return "".join(c for c in text if c.isalnum() or c == "_") or "sin_categoria"


def _response_hash(respuesta: str) -> str:
    """Hash MD5 corto de la respuesta para identificación rápida."""
    return hashlib.md5((respuesta or "").encode("utf-8")).hexdigest()[:10]


def read_tickets(xlsx_path: str) -> List[Dict[str, Any]]:
    """Lee las hojas del Excel de tickets.

    Cada fila se convierte en un dict candidato a ticket con ``categoria``
    (nombre de hoja), ``asunto`` y ``respuesta`` (normalizada si existe).

    Args:
        xlsx_path: Ruta al Excel de tickets.

    Returns:
        Lista de tickets (sin ``ticket_id`` aún).
    """
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl no está instalado") from exc

    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel de tickets no encontrado: {path}")

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    tickets: List[Dict[str, Any]] = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        header = next(rows, None) or ()
        header = [str(h).strip().lower() if h else "" for h in header]

        col_asunto = None
        col_resp_normalizada = None
        col_resp_final = None
        for idx, h in enumerate(header):
            if "asunto" in h:
                col_asunto = idx
            if "normalizada" in h:
                col_resp_normalizada = idx
            elif "respuesta" in h:
                col_resp_final = idx

        if col_asunto is None:
            continue

        for row in rows:
            if not row or not row[col_asunto]:
                continue
            asunto = str(row[col_asunto]).strip()
            respuesta = ""
            if col_resp_normalizada is not None and row[col_resp_normalizada]:
                respuesta = str(row[col_resp_normalizada]).strip()
            elif col_resp_final is not None and row[col_resp_final]:
                respuesta = str(row[col_resp_final]).strip()
            if not respuesta:
                continue
            tickets.append({
                "categoria": ws.title,
                "categoria_slug": _slugify(ws.title),
                "asunto": asunto,
                "respuesta": respuesta,
                "respuesta_hash": _response_hash(respuesta),
            })
    wb.close()
    return tickets


def build_ticket_registry(
    xlsx_path: str = DEFAULT_XLSX,
    registry_file: str = REGISTRY_FILE,
    chunks_file: str = CHUNKS_FILE,
    write: bool = True,
) -> Dict[str, Any]:
    """Genera el registro de tickets y los chunks listos para RAG.

    Los ``ticket_id`` se asignan de forma determinística: se recorre el Excel
    por hoja (orden alfabético) y cada fila recibe ``TKT-MESA-####`` en orden.
    Así, ejecutar dos veces sobre el mismo Excel produce IDs idénticos.

    Args:
        xlsx_path: Ruta al Excel de tickets.
        registry_file: Ruta de salida del registro JSONL.
        chunks_file: Ruta de salida de los chunks RAG JSONL.
        write: Si False, solo devuelve las estadísticas sin escribir archivos.

    Returns:
        Dict con stats: total tickets, por categoría, por estado de carga.
    """
    tickets = read_tickets(xlsx_path)
    tickets.sort(key=lambda t: (t["categoria_slug"], t["asunto"].lower()))

    registry_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    now = datetime.now().isoformat()

    existing = {}
    if Path(registry_file).exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = (row.get("categoria_slug"), row.get("asunto", "").lower(), row.get("respuesta_hash"))
                    existing[key] = row
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer registro previo ({e}); se regenera desde cero")

    by_cat: Dict[str, int] = {}
    for idx, t in enumerate(tickets, start=1):
        ticket_id = f"{TICKET_ID_PREFIX}{idx:04d}"
        source_file = f"tkt_mesa_{idx:04d}.txt"
        prev = existing.get((t["categoria_slug"], t["asunto"].lower(), t["respuesta_hash"]))
        en_faiss = bool(prev and prev.get("en_faiss", False))
        prev_id = prev.get("ticket_id") if prev else None

        registry_rows.append({
            "ticket_id": prev_id or ticket_id,
            "source_file_generated": source_file,
            "categoria": t["categoria"],
            "categoria_slug": t["categoria_slug"],
            "asunto": t["asunto"],
            "respuesta_hash": t["respuesta_hash"],
            "en_faiss": en_faiss,
            "created_at": now,
        })

        chunk_rows.append({
            "text": f"Asunto: {t['asunto']}\n{t['respuesta']}",
            "doc_type": "ticket",
            "source_file": source_file,
            "chunk_id": f"tkt_{idx:04d}",
            "page_range": "1-1",
            "metadata": {
                "ticket_id": prev_id or ticket_id,
                "categoria": t["categoria"],
                "categoria_slug": t["categoria_slug"],
                "asunto": t["asunto"],
                "tipo": "ticket_mesa_servicio",
            },
            "imported_at": now,
            "cargado_en_faiss": en_faiss,
        })
        by_cat[t["categoria"]] = by_cat.get(t["categoria"], 0) + 1

    if write:
        Path(registry_file).parent.mkdir(parents=True, exist_ok=True)
        with open(registry_file, "w", encoding="utf-8") as f:
            for row in registry_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"✅ Registro de tickets guardado: {registry_file}")

        Path(chunks_file).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "w", encoding="utf-8") as f:
            for row in chunk_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"✅ Chunks RAG listos: {chunks_file}")

    return {
        "total_tickets": len(tickets),
        "por_categoria": by_cat,
        "ya_en_faiss": sum(1 for r in registry_rows if r["en_faiss"]),
        "pendientes_de_cargar": sum(1 for r in registry_rows if not r["en_faiss"]),
        "registry_file": registry_file,
        "chunks_file": chunks_file,
    }


def mark_tickets_loaded(
    registry_file: str = REGISTRY_FILE,
    chunks_file: str = CHUNKS_FILE,
) -> Dict[str, Any]:
    """Marca el registro y los chunks como ya cargados en FAISS.

    Se usa DESPUÉS de ejecutar ``load_chunks_to_rag.py`` sobre
    ``tickets_chunks.jsonl`` para reflejar la trazabilidad real.

    Args:
        registry_file: Ruta del registro JSONL.
        chunks_file: Ruta de los chunks JSONL.

    Returns:
        Dict con totales actualizados.
    """
    registry_path = Path(registry_file)
    if not registry_path.exists():
        raise FileNotFoundError(f"No existe el registro: {registry_file}")

    rows: List[Dict[str, Any]] = []
    with open(registry_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["en_faiss"] = True
            rows.append(row)
    with open(registry_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if Path(chunks_file).exists():
        chunk_rows: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                chunk["cargado_en_faiss"] = True
                chunk_rows.append(chunk)
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunk_rows:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"✅ Tickets marcados como cargados en FAISS: {len(rows)}")
    return {"total_tickets": len(rows), "registry_file": registry_file, "chunks_file": chunks_file}


def main() -> None:
    parser = argparse.ArgumentParser(description="Registrar tickets de mesa de servicio")
    parser.add_argument("--xlsx", default=DEFAULT_XLSX, help="Ruta al Excel de tickets")
    parser.add_argument("--check", action="store_true", help="Solo inspección (no escribe archivos)")
    parser.add_argument("--mark-loaded", action="store_true", help="Marcar en_faiss=True (tras una reindexación real)")
    args = parser.parse_args()

    try:
        if args.mark_loaded:
            stats = mark_tickets_loaded()
            print(f"✅ Tickets marcados como cargados: {stats['total_tickets']}")
            return

        if args.check:
            tickets = read_tickets(args.xlsx)
            print(f"📋 Total de tickets encontrados: {len(tickets)}")
            from collections import Counter
            by_cat = Counter(t["categoria"] for t in tickets)
            for cat, n in sorted(by_cat.items()):
                print(f"   • {cat}: {n}")
            print("ℹ️  Modo --check: no se escribió ningún archivo.")
            return

        stats = build_ticket_registry(xlsx_path=args.xlsx, write=True)
        print("=" * 60)
        print("📊 RESULTADO DEL REGISTRO DE TICKETS")
        print("=" * 60)
        print(f"   ✅ Total tickets: {stats['total_tickets']}")
        print(f"   🔄 Ya en FAISS: {stats['ya_en_faiss']}")
        print(f"   ⏳ Pendientes de cargar: {stats['pendientes_de_cargar']}")
        for cat, n in sorted(stats["por_categoria"].items()):
            print(f"      - {cat}: {n}")
        print(f"   📄 Registro: {stats['registry_file']}")
        print(f"   📄 Chunks listos: {stats['chunks_file']}")
        print("ℹ️  Estos chunks NO están en FAISS todavía. Para cargarlos usa:")
        print("   python scripts/load_chunks_to_rag.py --chunks data/vector_store/tickets_chunks.jsonl")
        print("   y luego regresa a marcar el registro con --mark-loaded.")
    except Exception as e:
        logger.error(f"❌ Error en el registro de tickets: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()