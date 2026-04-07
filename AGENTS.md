# CONTEXTO ACTUAL DEL PROYECTO: ChatBot con Gemma-2-2b-it (RAG + LLM)

## DESCRIPCIÓN GENERAL
Chatbot educativo para Prepa en Línea SEP que utiliza arquitectura RAG con Gemma-2-2b-it como modelo generativo. El sistema recupera documentos relevantes de FAISS y genera respuestas contextualizadas. Migrado desde TinyLlama.

## ESTADO ACTUAL (Abril 2026)
✅ Migración de TinyLlama a Gemma-2-2b-it COMPLETADA
✅ Wrapper funcional en models/gemma_wrapper.py
✅ Generator en rag/gemma_generator.py
✅ Integración RAG funcionando en rag/core.py
✅ API operativa en api/main.py con endpoint /chat
✅ Interfaz web en static/index.html

## TECNOLOGÍAS PRINCIPALES
- **Framework API**: FastAPI (Python 3.12)
- **Modelo Principal**: google/gemma-2-2b-it (float32, CPU-only)
- **Vector Store**: FAISS (índices de documentos)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Ejecución**: CPU con 32GB RAM (sin GPU)
- **Parámetros actuales**: temperature=0.7, top_p=0.9, max_new_tokens=256, repetition_penalty=1.1

## ESTRUCTURA DE ARCHIVOS CLAVE

Chatbot-RAG-Fuente-Base/
├── api/
│ ├── main.py # Punto de entrada, inicializa RAGSystem
│ └── endpoints.py # Endpoints /upload, /search
├── rag/
│ ├── core.py # Clase RAGSystem con process_query() y _rag_process()
│ ├── gemma_generator.py # GemmaGenerator con generate_with_context()
│ ├── generator.py # TinyLlamaGenerator (legacy)
│ ├── retriever.py # VectorStoreFAISS para búsqueda
│ └── embeddings.py # EmbeddingModel
├── models/
│ ├── gemma_wrapper.py # Wrapper Gemma con device_map="cpu"
│ ├── tinyllama_wrapper.py # TinyLlama (legacy)
│ └── __init__.py
├── static/
│ └── index.html # Interfaz web
└── data/
└── vector_store/ # Índices FAISS y documentos chunked


## PROBLEMAS IDENTIFICADOS (RESUELTOS)

### 1. TinyLlama alucina con conocimiento general
✅ RESUELTO: Migrado a Gemma-2-2b-it que tiene mejor seguimiento de instrucciones