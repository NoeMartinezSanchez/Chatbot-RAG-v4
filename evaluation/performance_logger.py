import json
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_LOG = LOG_DIR / "retrieval_details.jsonl"
LATENCY_LOG = LOG_DIR / "latency.jsonl"
METRICS_SUMMARY = LOG_DIR / "metrics_summary.json"


class PerformanceLogger:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._metrics = defaultdict(list)
        self._last_summary_time = time.time()
        self._retrieval_count = 0
        self._total_latency = 0.0
        self._total_tokens = 0
        self._total_gen_time = 0.0
    
    def log_retrieval(
        self,
        query: str,
        results: list,
        search_time_ms: float,
        filters: Optional[dict] = None,
        intent: Optional[str] = None
    ):
        chunks = []
        for r in results:
            chunk_info = {
                "chunk_id": r.get("chunk_id", r.get("id", "unknown")),
                "score": round(r.get("score", 0.0), 4),
                "source_file": r.get("source_file", r.get("metadata", {}).get("source_file", "unknown")),
                "text_preview": (r.get("text", r.get("content", ""))[:200] if r.get("text") or r.get("content") else "")
            }
            chunks.append(chunk_info)
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query[:500],
            "intent": intent,
            "chunks_retrieved": len(chunks),
            "chunks": chunks,
            "search_time_ms": round(search_time_ms, 2),
            "filters": filters or {}
        }
        
        with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        self._retrieval_count += 1
    
    def log_latency(
        self,
        retrieval_time_ms: float,
        generation_time_ms: float,
        total_time_ms: float,
        tokens_generated: int,
        question: str
    ):
        tokens_per_sec = tokens_generated / (generation_time_ms / 1000) if generation_time_ms > 0 else 0
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_sec, 2),
            "question_preview": question[:100]
        }
        
        with open(LATENCY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        self._total_latency += total_time_ms
        self._total_tokens += tokens_generated
        self._total_gen_time += generation_time_ms
        
        self._maybe_save_summary()
    
    def _maybe_save_summary(self):
        now = time.time()
        if now - self._last_summary_time >= 3600:
            self.save_summary()
            self._last_summary_time = now
    
    def save_summary(self):
        avg_latency = self._total_latency / max(self._retrieval_count, 1)
        avg_tps = self._total_tokens / max(self._total_gen_time, 1) if self._total_gen_time > 0 else 0
        
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "period_hours": 1,
            "total_queries": self._retrieval_count,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens_per_second": round(avg_tps, 2),
            "total_tokens_generated": self._total_tokens
        }
        
        with open(METRICS_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self._retrieval_count = 0
        self._total_latency = 0.0
        self._total_tokens = 0
        self._total_gen_time = 0.0


perf_logger = PerformanceLogger()


def log_retrieval(*args, **kwargs):
    perf_logger.log_retrieval(*args, **kwargs)


def log_latency(*args, **kwargs):
    perf_logger.log_latency(*args, **kwargs)


def save_metrics_summary():
    perf_logger.save_summary()