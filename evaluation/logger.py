import json
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "evaluation.jsonl"
MODEL_NAME = "gemma-2b"


def setup_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        class JsonLinesFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps(record.__dict__)
        
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)
    
    return logger


def log_evaluation_result(
    question: str,
    response: str,
    retrieved_chunks_ids: list,
    ground_truth_chunks_ids: list,
    latency_ms: float
):
    hit_rate = len(set(retrieved_chunks_ids) & set(ground_truth_chunks_ids)) > 0
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "response": response[:500] if response else "",
        "retrieved_chunks_ids": retrieved_chunks_ids,
        "ground_truth_ids": ground_truth_chunks_ids,
        "hit_rate": hit_rate,
        "latency_ms": round(latency_ms, 2),
        "model_name": MODEL_NAME
    }
    
    logger = logging.getLogger("evaluation")
    if not logger.handlers:
        setup_logger()
    
    logger.info("", extra=log_entry)
    
    return log_entry