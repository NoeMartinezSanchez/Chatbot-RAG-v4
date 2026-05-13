# CONTEXTO ACTUAL DEL PROYECTO: ChatBot con Gemini 2.5 Flash + RAG

## DESCRIPCIÓN GENERAL
Chatbot educativo para Prepa en Línea SEP que utiliza arquitectura RAG con Gemini 2.5 Flash como modelo generativo via Google AI API. El sistema recupera documentos relevantes de FAISS y genera respuestas contextualizadas.

## ESTADO ACTUAL (Abril 2026)
✅ Integración con Google AI API (Gemini 2.5 Flash)
✅ Wrapper funcional en models/gemini_wrapper.py
✅ Evaluación automática en evaluation/automated_evaluator.py
✅ Diagnóstico del vector store en logs de inicio
✅ Interfaz web en static/index.html
✅ API FastAPI con endpoint /chat

## TECNOLOGÍAS PRINCIPALES
- **Framework API**: FastAPI (Python 3.11)
- **LLM**: Gemini 2.5 Flash via Google AI API
- **Vector Store**: FAISS (FlatL2)
- **Embeddings**: intfloat/multilingual-e5-small (384 dims)
- **Ejecución**: CPU con 32GB RAM (HF Spaces: 8 vCPU)
- **Parámetros actuales**: temperature=0.1, max_tokens=256

## ESTRUCTURA DE ARCHIVOS CLAVE

Chatbot-RAG-Fuente-Base/
├── api/
│   ├── main.py           # Punto de entrada + diagnóstico FAISS al inicio
│   └── endpoints.py     # Endpoints /upload, /search
├── rag/
│   ├── core.py         # Clase RAGSystem
│   ├── gemma_generator.py  # Generator → GeminiWrapper
│   ├── optimized_retriever.py  # Retrieval con sinónimos
│   ├── retriever.py     # VectorStoreFAISS
│   └── embeddings.py   # EmbeddingModel (e5-small)
├── models/
│   └── gemini_wrapper.py  # Google AI API wrapper
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

### Generación (Gemini 2.5 Flash)
- Modelo: gemini-2.0-flash
- API: Google AI API
- Temperature: 0.1
- Max output tokens: 256
- Timeout: 60s

## PROBLEMAS IDENTIFICADOS Y SOLUCIONES

### 1. Respuestas vacías del modelo
✅ RESUELTO: Prompt simplificado + parámetros optimizados
✅ Parámetros optimizados: temperature=0.1, max_tokens=256

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

## CONFIGURACIÓN DE API

### Variables de entorno requeridas:
```
GOOGLE_API_KEY=tu_api_key_de_google
```

### Para obtener API Key:
1. Ir a https://aistudio.google.com/apikey
2. Crear una nueva API key
3. Agregar al archivo .env o variable de entorno

## NOTAS PARA HF SPACES

1. Configurar GOOGLE_API_KEY en secrets del Space
2. Diagnóstico se imprime en logs de inicio
3. Resultados de evaluación en /data/dashboard.html