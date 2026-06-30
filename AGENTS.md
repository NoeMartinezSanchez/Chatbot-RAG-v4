# Chatbot-RAG-Fuente-Base — AGENTS.md

## Build / Run / Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server (development with hot-reload)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run server via app.py entry point
python app.py

# Run ALL tests (pytest)
python -m pytest tests/ -v

# Run ALL tests (native entry points, no pytest required)
python tests/test_rag.py
python tests/test_api.py

# Run a SINGLE test file with pytest
python -m pytest tests/test_rag.py -v

# Run a SINGLE test function with pytest
python -m pytest tests/test_rag.py::test_rag_initialization -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term

# Docker build & run
docker build -t chatbot-rag .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key chatbot-rag

# Load document chunks into FAISS vector store
python scripts/load_chunks_to_rag.py

# Run automated evaluation
python -m evaluation.automated_evaluator

# Generate dashboard
python -m evaluation.generate_dashboard
```

## Required Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI API key for Gemini 2.5 Flash |
| `GEMINI_API_KEY` | Alias for GOOGLE_API_KEY (fallback) |
| `GROQ_API_KEY` | Groq API key (alternative provider) |
| `LOG_LEVEL` | Logging level (default: INFO) |
| `ENVIRONMENT` | `development`, `staging`, or `production` |

Place in `.env` file or as OS environment variables.

## Python & Tooling

- **Python**: 3.11 (target, also compatible with 3.10+)
- **API Framework**: FastAPI + Uvicorn (no Gunicorn)
- **Config**: `pydantic-settings` via `config/settings.py` (reads from `.env` / env vars)
- **Models**: Pydantic v2 (no attrs, no dataclasses for API schemas)
- **Logging**: `logging` stdlib + `loguru` in some modules; `logger = logging.getLogger(__name__)` per module
- **No linter/formatter configured** (no ruff, black, flake8, mypy in requirements). Do NOT add linters unless asked.

## Code Style Guidelines

### Imports

1. **stdlib first** (os, sys, json, logging, time, uuid, datetime, pathlib, etc.)
2. **Third-party** (fastapi, pydantic, numpy, faiss, sentence_transformers, google.generativeai)
3. **Local** (from config.xxx, from rag.xxx, from models.xxx, from evaluation.xxx, from utils.xxx)
4. Separate groups with a blank line.

```python
import os
import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from config.settings import settings
from rag.core import RAGSystem
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `RAGSystem`, `GeminiWrapper`, `VectorStoreFAISS`)
- **Functions/methods**: `snake_case` (e.g., `process_query`, `_clean_query`, `embed_text`)
- **Variables**: `snake_case` (e.g., `query_embedding`, `top_k`, `context_str`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TOP_K_RESULTS`, `SIMILARITY_THRESHOLD`)
- **Private methods**: prefixed with `_` (e.g., `_rag_process`, `_save`)
- **Module-level logger**: `logger = logging.getLogger(__name__)` (always at module top after imports)
- **Filenames**: `snake_case.py` (e.g., `gemini_wrapper.py`, `optimized_retriever.py`)

### Type Hints

- **Mandatory** for all function signatures (parameters and return types)
- Use `from typing import List, Dict, Any, Optional, Tuple, Callable`
- Use `Optional[str]` rather than `str | None` (Python 3.10+ compat)
- Pydantic models for all API request/response schemas

```python
def process_query(self, query: str) -> Tuple[str, bool, float, list]:
def embed_batch(self, texts: List[str], is_passage: bool = False) -> np.ndarray:
```

### Docstrings

- Google-style docstrings with `Args:` and `Returns:` sections
- Module-level docstring at top of file (triple-quoted string, can be a single line)
- Required for public methods; optional for private helpers

```python
def generate(self, query: str, context: str = "", max_length: int = 256) -> str:
    """Generate a response for the given query.
    
    Args:
        query: User question/query.
        context: Retrieved context from RAG system (optional).
        max_length: Maximum tokens to generate.
    
    Returns:
        Generated response string.
    """
```

### Error Handling Pattern

```python
try:
    # operation
    logger.info(f"Operation succeeded: ...")
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    return "User-facing fallback message"  # or raise HTTPException for API endpoints
```

- Always log the error with `logger.error(...)`
- Use `exc_info=True` for unexpected errors to capture stack traces
- API endpoints raise `HTTPException(status_code=500, detail=str(e))`
- Internal methods return fallback strings (never propagate raw errors to the user in Spanish)
- Graceful degradation: if a component fails (e.g., Gemini), the system returns a friendly message

### Logging

- Module-level: `logger = logging.getLogger(__name__)`
- Use structured prefixes: `"✅ ..."` (success), `"❌ ..."` (error), `"⚠️ ..."` (warning), `"📩 ..."` (incoming message), `"📤 ..."` (outgoing), `"🔍 ..."` (debug)
- Always include enough context (query length, response preview, source count, etc.)

### API Conventions

- FastAPI app in `api/main.py`, sub-routes in `api/endpoints.py` under `/documents`
- Pydantic models in `config/models.py` (`ChatRequest`, `ChatResponse`, `FeedbackRequest`, `Document`)
- CORS: `allow_origins=["*"]` (already configured)
- Standard headers: `X-User-ID`, `X-Conversation-ID`, `X-Message-ID`, `X-Response-Type`
- `/chat` returns JSON with `response`, `sources`, `is_rag_response`, `confidence`
- `/health` returns static `{"status": "healthy", ...}`
- All endpoints wrap body in try/except with `HTTPException`

### RAG Pipeline Conventions

- **Embeddings**: `intfloat/multilingual-e5-small` (384 dims). Always use `"query: "` prefix for user queries, `"passage: "` prefix for indexed chunks.
- **Vector store**: FAISS (FlatL2 index) persisted in `data/vector_store/` via pickle + .bin
- **Retrieval**: `OptimizedRetriever` wraps `VectorStoreFAISS` with query expansion, synonyms, and multi-query support
- **Generator**: `GemmaGenerator` → `GroqWrapper` (class name is legacy; actually uses Gemini 2.5 Flash via Google AI API or Groq)
- **Intents**: JSON-based intent matching for greetings/farewells in `VectorStoreFAISS.search_intents()`. Intents always take priority over RAG for saludos, despedidas, gracias.
- `RAGSystem.process_query()` returns `Tuple[str, bool, float, list]` → (response, is_rag, confidence, sources)

### Session & Conversation ID

- Every `/chat` request may include `user_id`, `conversation_id`, `session_id`
- If not provided, the API generates UUIDs
- Conversations are stored in an in-memory dict (volatile; no database)
- User interactions persisted to `/data/user_interactions.jsonl` for dashboard

### Testing

- Tests live in `tests/test_rag.py` and `tests/test_api.py`
- Tests use `sys.path.append` + direct imports (no conftest.py)
- `test_api.py` uses `fastapi.testclient.TestClient`
- `test_rag.py` uses pytest assertions
- Tests can also be run directly via `if __name__ == "__main__"` blocks
- Do NOT create new test files without following the existing patterns
- No mocking framework used; real components are instantiated

### Project Structure Rules

```
api/           → FastAPI routes (main.py, endpoints.py)
config/        → Settings + Pydantic models
rag/           → Core RAG: embeddings, retriever, generator, optimized_retriever, core
models/        → LLM wrappers (gemini, groq, ollama, tinyllama)
evaluation/    → Automated tests, dashboards, performance logging
scripts/       → One-off data loading and setup scripts
tests/         → pytest test files
data/          → FAISS index, metadata, documents, logs
static/        → Web UI (index.html, dashboard.html)
utils/         → Log capture utilities
langchain_layer/ → LangChain orchestration wrapper (legacy/deprecated)
```

- Keep each module focused: wrappers in `models/`, RAG pipeline logic in `rag/`, API in `api/`
- Do NOT import from `models/` into `api/` directly — go through `rag/core.py`

### Deployment Notes

- Hugging Face Spaces: set `GOOGLE_API_KEY` in Space secrets
- Docker: `python:3.11-slim`, exposes port `7860`
- No database — everything is file-based (FAISS index, JSONL logs, pickle metadata)
- API supports `ENVIRONMENT=production` with debug/workers/reload toggles in Settings
- Healthcheck at `/health` — used by Render/HF Spaces for uptime monitoring
