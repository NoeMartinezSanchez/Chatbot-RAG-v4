# CONTEXTO ACTUAL DEL PROYECTO: ChatBot con Ollama (Gemma 4) + RAG

## DESCRIPCIÓN GENERAL
Chatbot educativo para Prepa en Línea SEP que utiliza arquitectura RAG con Gemma 4 via Ollama como modelo generativo. El sistema recupera documentos relevantes de FAISS y genera respuestas contextualizadas.

## ESTADO ACTUAL (Abril 2026)
✅ Migración COMPLETA a Ollama (Gemma 4 E4B)
✅ Wrapper funcional en models/ollama_wrapper.py
✅ Evaluación automática en evaluation/automated_evaluator.py
✅ Diagnóstico del vector store en logs de inicio
✅ Interfaz web en static/index.html
✅ API beroperasi di FastAPI con endpoint /chat

## TECNOLOGÍAS PRINCIPALES
- **Framework API**: FastAPI (Python 3.11)
- **LLM**: gemma4:e4b via Ollama (port 11434)
- **Vector Store**: FAISS (FlatL2)
- **Embeddings**: intfloat/multilingual-e5-small (384 dims)
- **Ejecución**: CPU con 32GB RAM (HF Spaces: 8 vCPU)
- **Parámetros actuales**: temperature=0.1, top_p=0.85, num_predict=20

## ESTRUCTURA DE ARCHIVOS CLAVE

Chatbot-RAG-Fuente-Base/
├── api/
│   ├── main.py           # Punto de entrada + diagnóstico FAISS al inicio
│   └── endpoints.py     # Endpoints /upload, /search
├── rag/
│   ├── core.py         # Clase RAGSystem
│   ├── gemma_generator.py  # GemmaGenerator → OllamaWrapper
│   ├── optimized_retriever.py  # Retrieval con sinónimos
│   ├── retriever.py     # VectorStoreFAISS
│   └── embeddings.py   # EmbeddingModel (e5-small)
├── models/
│   └── ollama_wrapper.py  # Ollama API wrapper
├── evaluation/
│   ├── automated_evaluator.py  # Tests automáticos
│   └── test_set.json       # 20 preguntas de evaluación
├── scripts/
│   └── load_chunks_to_rag.py  # Carga chunks con prefijo "passage: "
├── config/
│   └── settings.py       # Configuración centralizada
└── data/
    └── vector_store/     # Índice FAISS y metadatos

## CONFIGURACIÓN ACTUAL (Abril 2026)

### Embeddings (e5-small)
- Modelo: intfloat/multilingual-e5-small
- Prefijos: "query: " para preguntas, "passage: " para chunks
- Dimensiones: 384

### Retrieval
- top_k_initial: 15
- top_k_final: 7
- min_similarity: 0.3 (reducido para diagnóstico)
- use_query_expansion: True
- use_multi_query: True
- use_synonyms: True

### Generación (Ollama)
- Modelo: gemma4:e4b
- Timeout: 120s
- num_predict: 20
- temperature: 0.1

## PROBLEMAS IDENTIFICADOS Y SOLUCIONES

### 1. Respuestas vacías de Gemma 4
✅ RESUELTO: Prompt simplificado + timeout aumentado a 120s
✅ Parámetros optimizados: temperature=0.1, num_predict=20

### 2. Retrieval no encuentra chunks
✅ EN PROCESO: Diagnosticado con código en api/main.py
✅ min_similarity reducido a 0.3
✅ Verificar que chunks usen prefijo "passage: " al indexar

### 3. Embeddings no coinciden
✅Cambiado a intfloat/multilingual-e5-small
✅ Re-indexado con prefijos "passage: " requeridos

## EVALUACIÓN

- 20 preguntas de test en evaluation/test_set.json
- Categorías: convocatoria, normativa, guia, protocolo, reglas_comunicacion
- Dificultades: facil (7), medio (7), difficile (6)
- Ejecución automática al inicio del Space

## NOTAS PARA HF SPACES

1. Ollama inicie primero (start.sh)
2. Descargar gemma4:e4b automáticamente
3. Diagnóstico se imprime en logs de inicio
4. Resultados de evaluación en /data/dashboard.html