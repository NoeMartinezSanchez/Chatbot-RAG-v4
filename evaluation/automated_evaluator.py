"""Automated Evaluator for RAG Chatbot.

Runs evaluation tests against the chatbot and logs results.
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger


# Usar /data para persistencia en HF Spaces (Storage Bucket)
DATA_DIR = Path("/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rutas en /data/
EVALUATION_LOG = DATA_DIR / "evaluation_results.jsonl"
DASHBOARD_PATH = DATA_DIR / "dashboard.html"
SUMMARY_PATH = DATA_DIR / "evaluation_summary.json"

TIMEOUT_SECONDS = 30


def _evaluate_response(response: str, palabras_clave: List[str]) -> tuple[bool, str]:
    """
    Evaluate if response contains the expected keywords.
    
    Args:
        response: The chatbot's response
        palabras_clave: List of keywords to check
    
    Returns:
        (is_correct, method_used)
    """
    response_lower = response.lower()
    palabras_lower = [p.lower() for p in palabras_clave]
    
    metodo1 = all(p in response_lower for p in palabras_lower)
    metodo2 = any(p in response_lower for p in palabras_lower)
    
    if metodo1:
        return True, "metodo1_todas"
    elif metodo2:
        return True, "metodo2_al_menos_una"
    else:
        return False, "ninguno"


def _run_single_test(
    retriever: Any,
    generator: Any,
    test: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single test case."""
    pregunta = test["pregunta"]
    palabras_clave = test.get("palabras_clave", [])
    
    start_time = time.time()
    retrieval_start = start_time
    
    try:
        import numpy as np
        from rag.embeddings import EmbeddingModel
        
        embedder = EmbeddingModel()
        query_embedding = embedder.embed_text(pregunta)
        
        results = retriever.retrieve(pregunta, query_embedding, top_k=5)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
    except Exception as e:
        logger.warning(f"Error in retrieval for test {test.get('id')}: {e}")
        return {
            "test_id": test.get("id", "unknown"),
            "pregunta": pregunta,
            "respuesta_chatbot": "",
            "correcto": False,
            "metodo_usado": "error_retrieval",
            "latency_ms": 0,
            "retrieval_time_ms": 0,
            "generation_time_ms": 0,
            "error": str(e)
        }
    
    contexts = [r.get("content", r.get("text", "")) for r in results]
    context_str = " ".join(contexts)
    
    # Logging detallado de chunks recuperados
    test_id = test.get("id", "unknown")
    logger.info(f"🔍 Chunks recuperados para {test_id}:")
    for i, chunk in enumerate(results):
        chunk_id = chunk.get("chunk_id", chunk.get("doc_index", "N/A"))
        score = chunk.get("similarity", chunk.get("score", chunk.get("reranked_score", 0)))
        fuente = chunk.get("source_file", chunk.get("metadata", {}).get("source_file", "N/A"))
        logger.info(f"   {i+1}. ID:{str(chunk_id)[:12]} - score:{score:.4f} - fuente:{fuente}")
    
    generation_start = time.time()
    
    try:
        response = generator.generate_with_context(
            context=context_str,
            question=pregunta,
        )
        generation_time = (time.time() - generation_start) * 1000
        
    except Exception as e:
        logger.warning(f"Error in generation for test {test.get('id')}: {e}")
        return {
            "test_id": test.get("id", "unknown"),
            "pregunta": pregunta,
            "respuesta_chatbot": "",
            "correcto": False,
            "metodo_usado": "error_generation",
            "latency_ms": (time.time() - start_time) * 1000,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": 0,
            "error": str(e)
        }
    
    correcto, metodo = _evaluate_response(response, palabras_clave)
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "test_id": test.get("id", "unknown"),
        "pregunta": pregunta,
        "respuesta_chatbot": response[:500] if response else "",
        "respuesta_esperada": test.get("respuesta_esperada", ""),
        "palabras_clave": palabras_clave,
        "correcto": correcto,
        "metodo_usado": metodo,
        "dificultad": test.get("dificultad", "unknown"),
        "categoria": test.get("categoria", "unknown"),
        "latency_ms": round(total_time, 2),
        "retrieval_time_ms": round(retrieval_time, 2),
        "generation_time_ms": round(generation_time, 2),
        "chunks_retrieved": len(results)
    }


def _load_tests(test_set_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    try:
        with open(test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tests", [])
    except Exception as e:
        logger.error(f"Error loading test set: {e}")
        return []


def _generate_dashboard(output_path: Path = None):
    """Generate dashboard after evaluation completes."""
    try:
        from evaluation.generate_dashboard import generate_dashboard
        dashboard_file = generate_dashboard(output_path=output_path)
        logger.info(f"✅ Dashboard generado: {dashboard_file}")
        logger.info(f"✅ Existe: {dashboard_file.exists() if dashboard_file else False}")
    except Exception as e:
        logger.warning(f"⚠️ Error generando dashboard: {e}")


def run_automated_evaluation(
    retriever: Any,
    generator: Any,
    test_set_path: str = "evaluation/test_set.json",
    output_path: str = None,
    run_async: bool = True
):
    """
    Run automated evaluation on the chatbot.
    
    Args:
        retriever: OptimizedRetriever instance
        generator: GemmaGenerator instance
        test_set_path: Path to test_set.json
        output_path: Custom output path (optional)
        run_async: If True, runs in background thread
    """
    if output_path is None:
        output_path = EVALUATION_LOG
    
    def _run():
        logger.info("🚀 Starting automated evaluation...")
        
        tests = _load_tests(test_set_path)
        if not tests:
            logger.warning("No tests found in test_set.json")
            return
        
        logger.info(f"Loaded {len(tests)} test cases")
        
        if os.path.exists(output_path):
            os.remove(output_path)
        
        results = []
        for i, test in enumerate(tests):
            logger.info(f"Running test {i+1}/{len(tests)}: {test.get('id', 'unknown')}")
            
            result = _run_single_test(retriever, generator, test)
            results.append(result)
            
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        total = len(results)
        correctas = sum(1 for r in results if r["correcto"])
        incorrectas = total - correctas
        tasa_exito = (correctas / total * 100) if total > 0 else 0
        
        logger.info(f"✅ Evaluation complete: {correctas}/{total} correct ({tasa_exito:.1f}%)")
        
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_tests": total,
            "correctas": correctas,
            "incorrectas": incorrectas,
            "tasa_exito": round(tasa_exito, 2),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 2) if total > 0 else 0,
            "avg_retrieval_ms": round(sum(r["retrieval_time_ms"] for r in results) / total, 2) if total > 0 else 0,
            "avg_generation_ms": round(sum(r["generation_time_ms"] for r in results) / total, 2) if total > 0 else 0,
        }
        
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Resultados guardados en: {DATA_DIR}/")
        logger.info(f"   - Resumen: {SUMMARY_PATH}")
        logger.info(f"   - Dashboard: {DASHBOARD_PATH}")
        logger.info(f"   - Resultados detallados: {EVALUATION_LOG}")
        
        # Imprimir resultados detallados en los logs
        logger.info("=" * 60)
        logger.info("📋 RESULTADOS DETALLADOS DE EVALUACIÓN:")
        logger.info("=" * 60)
        for r in results:
            estado = "✅ CORRECTO" if r.get("correcto") else "❌ INCORRECTO"
            test_id = r.get("test_id", "N/A")
            pregunta = r.get("pregunta", "")[:60]
            respuesta = r.get("respuesta_chatbot", "")[:80]
            logger.info(f"{estado} | {test_id} | {pregunta}...")
            logger.info(f"   → Respuesta: {respuesta}")
        
        logger.info("=" * 60)
        logger.info(f"📊 RESUMEN: {correctas}/{total} correctas ({tasa_exito:.1f}%)")
        logger.info("=" * 60)
        
        _generate_dashboard(output_path=DASHBOARD_PATH)
    
    if run_async:
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("Evaluation started in background")
    else:
        _run()


def run_evaluation_sync(
    retriever: Any,
    generator: Any,
    test_set_path: str = "evaluation/test_set.json"
) -> Dict[str, Any]:
    """
    Run evaluation synchronously and return summary.
    
    Returns:
        Summary dict with metrics
    """
    output_path = EVALUATION_LOG
    
    tests = _load_tests(test_set_path)
    if not tests:
        return {"error": "No tests found"}
    
    results = []
    for test in tests:
        result = _run_single_test(retriever, generator, test)
        results.append(result)
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    total = len(results)
    correctas = sum(1 for r in results if r["correcto"])
    
    _generate_dashboard(output_path=DASHBOARD_PATH)
    
    return {
        "total": total,
        "correctas": correctas,
        "incorrectas": total - correctas,
        "tasa_exito": round(correctas / total * 100, 2) if total > 0 else 0
    }