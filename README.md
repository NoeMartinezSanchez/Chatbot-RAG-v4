---
title: Chatbot RAG PLS
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🤖 Prepa en Línea SEP - Asistente Educativo RAG

Asistente virtual con tecnología RAG (Retrieval-Augmented Generation) para estudiantes de Prepa en Línea SEP. Chatbot educativo que responde dudas sobre Convocatoria, Normativa, Protocolos y Guías del programa.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![HuggingFace](https://img.shields.io/badge/Deployed%20on-HuggingFace%20Spaces-yellow)

## 🎯 Características Principales

- **RAG con Gemma 4**: Sistema de Retrieval-Augmented Generation usando Ollama con Gemma 4 E4B
- **Búsqueda Vectorial**: FAISS con embeddings multilingües optimizados para español
- **Evaluación Automática**: 20 casos de prueba con métricas de calidad
- **Diagnóstico en Tiempo Real**: Logging detallado para debugging
- **Interfaz Web**: Dashboard interactivo para consulta

## 🛠️ Tecnologías

| Categoría | Tecnología |
|----------|------------|
| **API** | FastAPI + Uvicorn |
| **LLM** | Ollama (gemma4:e4b) |
| **Vector Store** | FAISS (FlatL2) |
| **Embeddings** | intfloat/multilingual-e5-small |
| **Deployment** | Hugging Face Spaces |
| **Frontend** | HTML/CSS/JavaScript |

## 📁 Estructura del Proyecto

```
Chatbot-RAG-Fuente-Base/
├── api/
│   ├── main.py              # FastAPI + diagnóstico
│   └── endpoints.py       # Endpoints adicionales
├── rag/
│   ├── core.py            # RAGSystem principal
│   ├── gemma_generator.py # Generador con Ollama
│   ├── optimized_retriever.py # Retrieval avanzado
│   ├── retriever.py       # FAISS wrapper
│   └── embeddings.py      # EmbeddingModel (e5-small)
├── models/
│   └── ollama_wrapper.py # Wrapper Ollama API
├── evaluation/
│   ├── automated_evaluator.py # Tests automáticos
│   ├── generate_dashboard.py # Visualización
│   └── test_set.json       # 20 preguntas de eval
├── scripts/
│   └── load_chunks_to_rag.py # Carga de documentos
├── config/
│   └── settings.py       # Configuración centralizada
└── data/
    └── vector_store/     # Índice FAISS
```

## 🚀 Instalación Local

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/Chatbot-RAG-Fuente-Base.git
cd Chatbot-RAG-Fuente-Base

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 5. Iniciar Ollama y descargar modelo
ollama serve
ollama pull gemma4:e4b

# 6. Cargar documentos al vector store
python scripts/load_chunks_to_rag.py

# 7. Iniciar API
uvicorn api.main:app --reload
```

## 📖 Uso de la API

### Endpoint Principal

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿El módulo propedéutico es obligatorio?"}'
```

### Respuesta

```json
{
  "response": "Sí, el módulo propedéutico es obligatorio.",
  "sources": ["bases_convocatoria_g85.pdf"],
  "is_rag_response": true,
  "confidence": 0.85
}
```

### Documentación API

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 🧪 Evaluación

El sistema incluye 20 casos de prueba que se ejecutan automáticamente al iniciar:

```bash
# Ver resultados de evaluación
curl http://localhost:8000/evaluation-results
```

O accede al dashboard: http://localhost:8000/dashboard

## 🌐 Despliegue en Hugging Face Spaces

1. Push a GitHub:
```bash
git add .
git commit -m "feat: Chatbot RAG con Ollama"
git push origin main
```

2. Crear nuevo Space en https://huggingface.co/spaces

3. Seleccionar hardware: **CPU (8 vCPU, 32GB RAM)**

4. El Space iniciaría Ollama automáticamente

## 📊 Métricas del Sistema

| Métrica | Valor |
|--------|-------|
| **Chunks Indexados** | 108 |
| **Dimensión Embedding** | 384 |
| **Preguntas de Test** | 20 |
| **Tiempo de Respuesta** | < 2s |

## 👨‍💻 Habilidades Demostradas

- **Machine Learning**: RAG, embeddings, vector search
- **Deep Learning**: LLMs (Gemma 4), prompt engineering
- **APIs**: FastAPI, RESTful design
- **DevOps**: Docker, Hugging Face Spaces
- **Logging**: Diagnóstico avanzado, monitoreo
- **Testing**: Evaluación automática, métricas
- **Database**: FAISS, vector stores

## 📝 Licencia

MIT License - feel free to use for learning and personal projects.
