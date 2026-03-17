# CONTEXTO ACTUAL DEL PROYECTO: ChatBot con TinyLlama (RAG + LLM) - VERSIÓN PRODUCCIÓN

## DESCRIPCIÓN GENERAL
Chatbot educativo para Prepa en Línea SEP que utiliza arquitectura RAG con TinyLlama como modelo generativo. El sistema recupera documentos relevantes de FAISS y genera respuestas contextualizadas. Actualmente en fase de ajuste fino de prompts para mejorar calidad de respuestas.

## ESTADO ACTUAL (Marzo 2026) - ¡CRÍTICO!
✅ Migración de BERT a TinyLlama COMPLETADA
✅ Wrapper funcional en models/tinyllama_wrapper.py
✅ Integración RAG funcionando en rag/core.py
✅ API operativa en api/main.py con endpoint /chat
✅ Interfaz web en static/index.html
⚠️ PROBLEMA PENDIENTE: Calidad de respuestas - TinyLlama alucina (menciona universidades/carreras) y a veces se encicla

## TECNOLOGÍAS PRINCIPALES
- **Framework API**: FastAPI (Python 3.12)
- **Modelo Principal**: TinyLlama-1.1B-Chat (Hugging Face Transformers)
- **Vector Store**: FAISS (índices de documentos)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Ejecución**: CPU (sin GPU) con 8GB RAM (85% ocupado)
- **Parámetros actuales**: temperature=0.1, repetition_penalty=1.3, top_p=0.8, max_new_tokens=150

## ESTRUCTURA DE ARCHIVOS CLAVE

ChatBot_5_TinyLlama/
├── api/
│ ├── main.py # Punto de entrada, inicializa RAGSystem
│ └── endpoints.py # Endpoints /upload, /search (NO tiene /chat)
├── rag/
│ ├── core.py # Clase RAGSystem con process_query() y _rag_process()
│ ├── generator.py # TinyLlamaGenerator con generate_with_context()
│ ├── retriever.py # VectorStoreFAISS para búsqueda
│ └── embeddings.py # EmbeddingModel
├── models/
│ ├── tinyllama_wrapper.py # Wrapper con carga del modelo e inferencia
│ └── init.py
├── static/
│ └── index.html # Interfaz web
└── data/
└── vector_store/ # Índices FAISS y documentos chunked


## PROBLEMAS IDENTIFICADOS (URGENTES)

### 1. TinyLlama alucina con conocimiento general
A pesar de tener documentos relevantes, TinyLlama usa su conocimiento pre-entrenado y menciona "universidad", "carreras", "ingeniería" cuando debería limitarse a Prepa en Línea SEP.

### 2. Formato de respuestas inconsistente
A veces usa guiones, a veces números, a veces párrafos. Necesita formato consistente.

### 3. Bucles de repetición
En preguntas como "¿Qué pasa si no tengo mi certificado?" se encicla repitiendo la misma frase.

### 4. Contexto con metadatos
Las fuentes consultadas incluyen [CONVOCATORIA], ##, 📄 que deben limpiarse antes de pasar a TinyLlama.