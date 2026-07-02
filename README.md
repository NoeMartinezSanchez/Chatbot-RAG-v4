---
title: Chatbot RAG PLS
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🤖 Prepa en Línea SEP - Asistente Educativo RAG con Memoria Conversacional (LangChain)

Asistente virtual con tecnología RAG (Retrieval-Augmented Generation) para estudiantes de Prepa en Línea SEP. Chatbot educativo que responde dudas sobre Convocatoria, Normativa, Protocolos y Guías del programa. El chatbot recuerda el contexto de la conversación y responde preguntas relacionadas de forma coherente.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Groq](https://img.shields.io/badge/Groq-GPT%20OSS%20120B-orange)
![HuggingFace](https://img.shields.io/badge/Deployed%20on-HuggingFace%20Spaces-yellow)

## 🎯 Características Principales

- **RAG con Groq API**: Sistema de Retrieval-Augmented Generation usando GPT OSS 120B vía Groq API
- **Sin GPU necesaria**: Todo via APIs externas (Groq, embeddings); ejecución en CPU
- **Búsqueda Vectorial**: FAISS con embeddings multilingües optimizados para español
- **Soporte Multi-formato**: Procesa PDF, Word (docx), Excel (xlsx) y TXT
- **Re-ranking y Query Expansion**: Calidad preservada con menos chunks (top_k=5)
- **Interfaz Web**: Dashboard responsive tipo chat
- **Memoria Conversacional (LangChain)**: El sistema guarda y recupera el historial de la conversación, inyectándolo en consultas posteriores.

## 🛠️ Tecnologías

| Categoría | Tecnología |
|----------|------------|
| **Framework API** | FastAPI + Uvicorn |
| **LLM** | Groq API — GPT OSS 120B |
| **Embeddings** | intfloat/multilingual-e5-small (local, 384 dims) |
| **Vector Store** | FAISS (cpu, FlatL2, top_k=5 optimizado) |
| **Re-ranking** | Activado con query expansion |
| **Procesamiento Documentos** | PyMuPDF (PDF), python-docx (Word), openpyxl (Excel) |
| **Frontend** | HTML/CSS/JS Vanilla (responsive) |
| **Despliegue** | Docker · Hugging Face Spaces |
| **Orquestación** | LangChain (Memoria conversacional y gestión del pipeline RAG) |
| **Evaluación Automática** | Desactivada por defecto (ahorro de tokens) |

## ⚡ Optimización de Tokens (Groq API)

### Problema Identificado

Consumo excesivo de **~4,500 tokens por consulta** que agotaba el límite gratuito de 100K tokens en menos de **25 interacciones**.

### Soluciones Implementadas

| Optimización | Antes | Después | Reducción |
|-------------|-------|---------|-----------|
| **Chunks de contexto** | 10 chunks | 5 chunks | 50% menos tokens |
| **Prompt del sistema** | ~250 tokens | ~60 tokens | **76% menos** |
| **TOP_K_RESULTS** | 10 | 5 | 50% menos chunks |

### Resultados Clave

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Tokens por consulta** | ~4,500 | ~1,500 | **67% menos** |
| **Consultas diarias posibles** | 20-22 | 60-65 | **+200%** |
| **Margen de seguridad** | 0% | 70% | ✅ |

### Mecanismos que Preservan la Calidad

- **Re-ranking activado**: los 5 chunks finales son los más relevantes
- **Filtrado por metadata**: elimina chunks no relacionados al contexto
- **Query expansion**: expande la consulta para mejor recall sin añadir tokens al LLM

### Configuración Final Optimizada

```python
# optimized_retriever.py
self.config = {
    "top_k_initial": 10,
    "top_k_final": 5,
    "min_similarity": 0.2,
    "use_reranking": True,
    "use_query_expansion": True,
    "use_multi_query": True,
}
```

### Prompt Optimizado

```python
# gemma_generator.py → GroqWrapper
system_msg = "Eres asistente de [CLIENTE]. Respuestas claras en español."
user_prompt = f"Contexto oficial: {context}\n\nPregunta: {question}\n\nRespuesta:"
```

> ⚠️ El sistema está optimizado para consumir **~1,500 tokens por consulta**, permitiendo **~65 interacciones diarias** en la capa gratuita de Groq API. El margen de seguridad es del **70%**.

## 🧠 Memoria Conversacional con LangChain

### ¿Qué se implementó?
Se añadió una capa de orquestación con LangChain que envuelve al sistema RAG existente **sin modificarlo**. Esto permite:

1. **Memoria real**: El chatbot recuerda preguntas y respuestas anteriores.
2. **Inyección automática de contexto**: El historial se inyecta en cada nueva consulta.
3. **Sesiones independientes**: Cada usuario tiene su propia memoria (identificada por `session_id`).

### Endpoints Disponibles

| Endpoint | Método | Propósito |
|----------|--------|-----------|
| `/chat` | POST | Original (sin memoria, fallback) |
| `/chat/v2` | POST | Con memoria conversacional (LangChain) |
| `/chat/clear_memory` | POST | Limpiar memoria de una sesión específica |

### Resultados de Pruebas

| Pregunta | Respuesta | Verificación |
|----------|-----------|--------------|
| "¿Qué es Prepa?" | "Servicio educativo gratuito..." | ✅ Correcta |
| "¿Es gratis?" | "Sí, el servicio es gratuito" | ✅ Memoria funcionando |
| "¿Cuánto dura?" | "2 años y 6 meses" | ✅ Contexto preservado |

### Ventajas

- **Sin modificar RAG existente**: Cambios completamente aislados.
- **Fallback seguro**: Endpoint `/chat` original intacto.
- **Base para agentes**: LangChain permite agregar tools fácilmente.

### Desventajas y Mitigación

| Desventaja | Mitigación |
|------------|------------|
| Aumento de tokens (~15%) | Compensado por optimizaciones previas (optimización de tokens) |
| Latencia adicional (~50ms) | Mínima, imperceptible para usuarios |
| Memoria en RAM | Espacios de HF reinician periódicamente, aceptable para pruebas |

## 📊 Dashboard de Métricas Avanzadas

### Problemas resueltos

| Problema | Solución |
|----------|----------|
| Confianza del 98.6% pero respuestas inútiles | **Tasa de Éxito Real**: detecta respuestas como "no encontré información" |
| No se veía distribución horaria de consumo | **Gráfico ASCII por hora**: muestra picos de consumo de tokens |

### Métricas implementadas

**1. Tasa de Éxito Real (Calidad de respuestas)**
- Detecta patrones de respuestas no útiles: "no encontré información", "tuve un problema", "intenta de nuevo"
- Diferencia entre confianza del sistema vs utilidad real

**2. Consumo de tokens por hora**
- Gráfico ASCII mostrando consumo en últimas 24 horas
- Identifica horas pico (ejemplo: 04:00 AM con 2,777 tokens)

### Ejemplo del dashboard
```
┌──────────────────────────────────────────────────────────────────┐
│ 📊 Tasa de Éxito: 94%                                           │
│ útiles: 135 | no útiles: 8                                      │
│ ████████████████████████████████████████░░░░░░░░ 94%            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ 📈 Consumo de Tokens por Hora                                   │
│                                                                  │
│ 04:00 │ ████████████████████ 2777 tokens                        │
│ 20:00 │ ████████████████ 2345 tokens                            │
│ 23:00 │ ████████ 1234 tokens                                    │
│                                                                  │
│ 🔵 Pico máximo: 2777 tokens/hora | 📊 Promedio: 1850 tokens/hora │
└──────────────────────────────────────────────────────────────────┘
```

### Beneficios

- **Antes**: "El sistema tiene 98.6% de confianza" (métrica engañosa)
- **Ahora**: "94% de respuestas RAG son útiles, 6% son 'no encontré información'"
- Permite identificar consultas problemáticas y horas de mayor carga

### Acceso

El dashboard está disponible en: `/user-dashboard`

## 🔄 CI/CD Pipeline

### ¿Qué es y por qué lo necesitamos?

CI/CD automatiza pruebas y despliegues. Cada cambio en el código se valida automáticamente antes de llegar a producción.

| Beneficio | Antes | Después |
|-----------|-------|---------|
| Detección de errores | En producción (usuarios afectados) | Antes del despliegue (2-3 minutos) |
| Tiempo de recuperación | 15-30 minutos manual | 0 minutos (no se despliega código roto) |
| Confianza en despliegues | Baja | Alta (pruebas automáticas) |
| Despliegue a HF Spaces | Manual con git push --force | Automático al hacer push a main |

### Arquitectura del Pipeline

```
git push → GitHub Actions → Pruebas automáticas → ¿OK? → Despliegue a HF Spaces
                                                          ↓
                                             [Fallo] → Notificar (pendiente)
```

### Pruebas implementadas

| Prueba | Qué verifica | Tiempo |
|--------|--------------|--------|
| Conexión Groq | API key válida, modelo responde | 2 seg |
| Importación API | Sin errores de sintaxis o imports | 1 seg |
| GroqWrapper | Inicialización correcta | 1 seg |
| RAGSystem | Carga de índices FAISS (108 vectores) | 2 seg |

**Tiempo total del pipeline: ~1 min 38 seg**

### Flujo de trabajo

```yaml
name: CI/CD - Test & Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - Checkout código
      - Setup Python 3.11
      - Instalar dependencias
      - Verificar archivos críticos
      - Test conexión Groq (real)
      - Test importación API
      - Test RAG system

  deploy:
    needs: test
    if: success()
    - Despliegue automático a Hugging Face Spaces
```

### Variables de entorno requeridas (GitHub Secrets)

| Secret | Propósito |
|--------|-----------|
| `GROQ_API_KEY` | Autenticación con Groq para pruebas reales |
| `HF_TOKEN` | Autenticación para desplegar a Hugging Face Spaces |

### Resultados obtenidos

- ✅ **Tasa de éxito de pruebas**: 100%
- ✅ **Tiempo de detección de errores**: 2 minutos
- ✅ **Impacto en usuarios por errores**: 0 (nunca se despliega código roto)
- ✅ **Despliegues automáticos**: Funcionando

### Compatibilidad con otros destinos

El pipeline es portable. Para migrar a VPS, AWS o Docker Hub solo se necesita agregar un step adicional:

```yaml
# Ejemplo: despliegue a VPS con Docker
- name: Deploy to VPS
  uses: appleboy/ssh-action@v1.0.0
  with:
    host: ${{ secrets.VPS_HOST }}
    script: docker-compose up -d
```

### Próximas mejoras

| Prioridad | Acción | Impacto |
|-----------|--------|---------|
| 1 | Cache de pip en workflow | Reduce tiempo 90s → 30s |
| 2 | Notificaciones a Telegram/Slack | Visibilidad inmediata de fallos |
| 3 | Pruebas unitarias por módulo | Mayor cobertura |
| 4 | Smoke test post-despliegue | Verifica que HF reconstruyó bien |

## 📁 Estructura del Proyecto

```
Chatbot-RAG-Fuente-Base/
├── langchain_layer/           # 🆕 Capa de orquestación con LangChain
│   ├── __init__.py            # Versión 0.1.0
│   ├── config.py              # Configuración (max tokens, TTL)
│   └── wrappers.py            # Wrapper con memoria real
├── api/
│   ├── main.py              # FastAPI + diagnóstico FAISS
│   └── endpoints.py         # Endpoints /upload, /search
├── rag/
│   ├── core.py              # RAGSystem principal
│   ├── gemma_generator.py   # Generador → GroqWrapper
│   ├── generator.py         # Generator legacy (TinyLlama, no usado)
│   ├── optimized_retriever.py  # Retrieval con expansión de queries
│   ├── retriever.py         # VectorStoreFAISS
│   └── embeddings.py        # EmbeddingModel (e5-small)
├── models/
│   ├── groq_wrapper.py      # Groq API wrapper (GPT OSS 120B)
│   ├── gemini_wrapper.py    # Gemini API wrapper (alternativa)
│   ├── ollama_wrapper.py    # Legacy (no usado)
│   └── tinyllama_wrapper.py # Legacy (no usado)
├── evaluation/
│   ├── automated_evaluator.py  # Tests automáticos
│   ├── generate_dashboard.py   # Visualización de resultados
│   └── test_set.json           # 20 preguntas de evaluación
├── scripts/
│   └── load_chunks_to_rag.py   # Carga de documentos al vector store
├── config/
│   └── settings.py          # Configuración centralizada
├── static/
│   └── index.html           # Frontend web responsive
├── data/
│   └── vector_store/        # Índice FAISS + metadatos
├── Dockerfile               # Imagen Docker para HF Spaces
└── requirements.txt         # Dependencias Python
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

# 4. Configurar variables de entorno (crear archivo .env)
# Obtén tu API Key en https://console.groq.com/keys
echo "GROQ_API_KEY=tu_api_key_aqui" > .env

# 5. Cargar documentos al vector store
python scripts/load_chunks_to_rag.py

# 6. Iniciar API
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

## 🧪 Validación del Sistema

4 preguntas críticas para verificar que el RAG funciona correctamente:

| Pregunta | Respuesta Esperada |
|----------|-------------------|
| ¿El módulo propedéutico es obligatorio? | Sí |
| ¿Qué significa escribir en mayúsculas? | Gritar |
| ¿Cuánto dura la prepa? | 2 años 6 meses |
| ¿La prepa es gratuita? | Sí, completamente gratis |

```bash
# Probar validación
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿El módulo propedéutico es obligatorio?"}'
```

## 🌐 Despliegue

### Hugging Face Spaces (Docker)

El proyecto incluye un `Dockerfile` listo para usar. El Space debe configurarse con **sdk: docker** (ya incluido en la cabecera YAML).

1. Configurar `GROQ_API_KEY` en los **Secrets** del Space
2. El hardware **CPU (8 vCPU, 32GB RAM)** es suficiente (no requiere GPU)
3. Hacer push al repositorio de Hugging Face

## 📊 Métricas del Sistema

| Métrica | Valor |
|--------|-------|
| **Chunks Indexados** | 108 |
| **Dimensión Embedding** | 384 |
| **Tokens por Consulta** | ~1,500 (optimizado) |
| **Consultas Diarias (free tier)** | ~65 |
| **Tiempo de Respuesta** | < 2s |
| **Memoria Conversacional** | ✅ Activa (LangChain 0.1.0) |

## 👨‍💻 Habilidades Demostradas

- **Machine Learning**: RAG, embeddings, vector search
- **Deep Learning**: LLMs (GPT OSS 120B vía Groq API), prompt engineering
- **APIs**: FastAPI, RESTful design
- **DevOps**: Docker, Hugging Face Spaces
- **Logging**: Diagnóstico avanzado, monitoreo, métricas de latencia
- **Testing**: Evaluación automática con 20 casos de prueba
- **Database**: FAISS, vector stores, NumPy

## ✅ Estado del Sistema (Junio 2026)

| Componente | Estado | Versión |
|------------|--------|---------|
| LangChain (Orquestación) | ✅ Operativo | 0.1.0 |
| Memoria Conversacional | ✅ Activa | - |
| RAG System | ✅ Sin cambios | Estable |
| Groq API (GPT OSS 120B) | ✅ Operativo | - |
| Retriever FAISS | ✅ Activo | 108 vectores |
| Endpoint `/chat` | ✅ Fallback | Original |
| Endpoint `/chat/v2` | ✅ Con memoria | Nuevo |

**Nota:** El sistema está listo para evaluación con usuarios, ofreciendo una experiencia de diálogo coherente y natural.

## 📝 Licencia

MIT License - feel free to use for learning and personal projects.
