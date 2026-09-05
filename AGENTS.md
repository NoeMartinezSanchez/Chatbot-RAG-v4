# Chatbot-RAG-Fuente-Base — AGENTS.md

## 🚀 Resumen del Proyecto
Chatbot RAG para Prepa en Línea SEP con:

- **Orquestación**: LangChain (v0.2.0) con memoria conversacional
- **Modelo generativo**: Groq con GPT OSS 120B (migrado desde Llama 3.3 70B)
- **Base de conocimiento**: 465 chunks indexados en FAISS (8 tipos de documentos)
- **Placeholders dinámicos**: URLs y fechas actualizables sin reindexación
- **Conciencia temporal**: Extracción automática de fechas y contexto
- **Memoria**: BufferMemory con historial de conversaciones
- **Persistencia**: MongoDB Atlas (motor async) para conversaciones, métricas y feedback
- **Caché de respuestas**: RAGCacheService (MD5 + TTL) para reducir latencia y consumo
- **CI/CD**: Pipeline automatizado con GitHub Actions
- **Despliegue**: Automático a Hugging Face Spaces
- **Monitoreo continuo**: Health check cada 10 min con alertas Telegram (0 tokens gastados)
- **Seguridad**: Sanitización de entrada (inyección, escape de contexto, ofuscación) + monitoreo en memoria con alertas Telegram para incidentes críticos

---

## 📋 Tabla de Contenidos
1. [Comandos de Build/Run/Test](#-comandos-de-buildruntest)
2. [Variables de Entorno](#-variables-de-entorno-requeridas)
3. [Python y Tooling](#-python--tooling)
4. [Base de Conocimiento RAG](#-base-de-conocimiento-rag)
5. [Placeholders Dinámicos](#-placeholders-dinámicos)
6. [Modelo Generativo: Groq con GPT OSS 120B](#-modelo-generativo-groq-con-gpt-oss-120b)
7. [LangChain y Orquestación](#-langchain-y-orquestación)
8. [Conciencia Temporal](#-conciencia-temporal)
9. [Sistema de Memoria](#-sistema-de-memoria)
10. [CI/CD Pipeline](#-cicd-pipeline-github-actions)
11. [Guías de Estilo de Código](#-guías-de-estilo-de-código)
12. [Estructura del Proyecto](#-estructura-del-proyecto)
13. [Estado Actual](#-estado-actual-del-sistema)
14. [Roadmap](#-roadmap-y-próximos-pasos)
15. [Monitoreo Continuo](#-monitoreo-continuo-en-producción)
16. [Sistema de Seguridad](#-sistema-de-seguridad)
17. [Persistencia MongoDB](#-persistencia-mongodb)
18. [Caché, Logging y Retención](#-caché-logging-y-retención)

---

## 🔧 Comandos de Build/Run/Test

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor API (desarrollo con hot-reload)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Ejecutar vía app.py
python app.py

# Ejecutar TODAS las pruebas (pytest)
python -m pytest tests/ -v

# Ejecutar prueba específica
python -m pytest tests/test_rag.py -v
python -m pytest tests/test_api.py -v

# Ejecutar función específica de prueba
python -m pytest tests/test_rag.py::test_rag_initialization -v

# Ejecutar con cobertura
python -m pytest tests/ --cov=. --cov-report=term

# Docker build & run
docker build -t chatbot-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key chatbot-rag

# Cargar chunks al vector store FAISS
python scripts/load_chunks_to_rag.py

# Extraer fechas de documentos automáticamente
python scripts/extract_dates.py

# Ejecutar evaluación automatizada
python -m evaluation.automated_evaluator

# Generar dashboard
python -m evaluation.generate_dashboard

# Verificar sistema de memoria
python verificar_memoria.py

# Probar conexión a MongoDB Atlas
python -m scripts.test_mongodb_connection

# Probar repositorios y servicios MongoDB
python -m scripts.test_services

# Probar integración completa (MongoDB + API con StubRAG)
python -m scripts.test_integration

# Probar cache de respuestas RAG
python -m scripts.test_cache

# Política de retención (dry-run primero)
python scripts/retention_policy.py --dry-run
python scripts/retention_policy.py

# Generar reporte de dashboard (JSON)
python scripts/generate_dashboard_report.py
🔑 Variables de Entorno Requeridas
Variable	Descripción	Obligatoria
GROQ_API_KEY	API key para Groq con GPT OSS 120B	✅ Sí
HF_TOKEN	Token para despliegue automático a HF Spaces (CI/CD)	❌ Solo CI/CD
TIMEZONE	Zona horaria para fechas (ej: America/Mexico_City)	✅ Sí
LOG_LEVEL	Nivel de logging (default: INFO)	❌ No
ENVIRONMENT	development, staging o production	❌ No
TELEGRAM_BOT_TOKEN	Token del bot de Telegram para alertas de monitoreo	❌ Solo monitoreo
TELEGRAM_CHAT_ID	Chat ID de Telegram para recibir alertas	❌ Solo monitoreo
GOOGLE_API_KEY	DEPRECATED: Solo compatibilidad con versiones anteriores	❌ No
GEMINI_API_KEY	DEPRECATED: Solo compatibilidad con versiones anteriores	❌ No
MONGODB_URI	URI de conexión a MongoDB Atlas (ej: mongodb+srv://user:pass@cluster.mongodb.net/)	✅ Sí (si MongoDB habilitado)
MONGODB_DB_NAME	Nombre de la base de datos (default: chatbot_rag_db)	❌ No
RAG_CACHE_TTL_HOURS	TTL del caché de respuestas en horas (default: 24)	❌ No
RAG_CACHE_ENABLED	Habilitar/deshabilitar caché (default: true)	❌ No
RETENTION_CONVERSATIONS_DAYS	Días a conservar conversaciones (default: 90)	❌ No
RETENTION_METRICS_DAYS	Días a conservar métricas (default: 30)	❌ No
RETENTION_LOGS_DAYS	Días a conservar logs (default: 14)	❌ No
RETENTION_FEEDBACK_DAYS	Días a conservar feedback (default: 180)	❌ No
ADMIN_API_KEY	API key opcional para endpoints /admin (vacío = sin protección)	❌ No
📌 IMPORTANTE: Colocar en .env o como variables de entorno del sistema.

🐍 Python & Tooling
Python: 3.11 (target, compatible con 3.10+)

API Framework: FastAPI + Uvicorn (sin Gunicorn)

Config: pydantic-settings vía config/settings.py

Models: Pydantic v2 (no attrs, no dataclasses para schemas API)

Logging: logging stdlib + loguru en algunos módulos

Orquestación: LangChain v0.1.0+ con integración Groq

Persistencia: MongoDB Atlas con motor (async) + pymongo/dnspython

Embeddings: SentenceTransformers con intfloat/multilingual-e5-small

Vector Store: FAISS con índice FlatL2

CI/CD: GitHub Actions con pipeline automatizado

NO linter/formatter configurado: No añadir ruff, black, flake8, mypy a menos que se solicite

📚 Base de Conocimiento RAG
Estado Actual
Métrica	Valor
Total de chunks	465
Tipos de documentos	8
Dimensión de embeddings	384
Modelo de embeddings	intfloat/multilingual-e5-small
Vector store	FAISS (FlatL2)
Distribución por Tipo de Documento
Tipo de documento	Cantidad	Porcentaje
Control Escolar	357	76.8%
Normativa	25	5.4%
Convocatoria	17	3.7%
Protocolo	15	3.2%
Reglas comunicación	15	3.2%
Guía	13	2.8%
Política cero tolerancia	12	2.6%
Decálogo	11	2.4%
Total	465	100%
Categorías de Control Escolar
Categoría	Registros	IDs asignados
Aspirante no cuenta con certificado	31	CE0001 - CE0031
Aspirantes extranjeros	44	CE0032 - CE0075
Estatus de mi registro	38	CE0076 - CE0113
Estudios previos de bachillerato	35	CE0114 - CE0148
Información general	109	CE0149 - CE0257
Inscripción a módulo 1	30	CE0258 - CE0287
Problemas con el registro	70	CE0288 - CE0357
Total	357	CE0001 - CE0357
Estructura de Chunk
python
{
  "chunk_id": "6471d2310de8",
  "text": "Texto del chunk con placeholders (url1, fecha1, etc.)",
  "doc_type": "control_escolar",
  "source_file": "Control_Escolar_Aspirante_no_cuenta_con_cert.xlsx",
  "page_range": "1-1",
  "metadata": {
    "id_control_escolar": "CE0001",
    "asunto": "No cuento con mi certificado de secundaria",
    "categoria": "Aspirante no cuenta con cert",
    "section": "No cuento con mi certificado de secundaria",
    "type": "control_escolar",
    "word_count": 108,
    "fecha_procesamiento": "2026-06-24T07:21:45.029497"
  }
}
🔄 Placeholders Dinámicos
📌 ¿Qué son?
Los placeholders son marcadores que reemplazan elementos dinámicos (URLs, fechas) en los textos almacenados en FAISS. Al momento de generar una respuesta, estos marcadores se resuelven con los valores actuales almacenados en archivos JSON.

Tipos de Placeholders
Elemento	Placeholder	Archivo de Mapeo	Ejemplo de resolución
URL	url1, url2, ...	mapeo_urls_global.json	url1 → https://prepaenlinea.sep.gob.mx/mi-comunidad/
Fecha	fecha1, fecha2, ...	mapeo_fechas_completo.json	fecha1 → del 15 al 26 de Junio de 2026
Categorías de Fechas Implementadas (10 categorías)
Categoría	Clave JSON	Patrón	Valor Actual
Periodo de resultados	periodo_resultados	del XX al XX de XXXX de XXXX	del 15 al 26 de Junio de 2026
Periodo de módulo	periodo_modulo	Módulo 1 del XX al XX de XXXX de XXXX	del 13 de Julio al 15 de Agosto de 2026
Periodo de convocatoria	periodo_convocatoria	del XX al XX de XXXX	del 10 al 20 de Agosto de 2026
Generación	generacion	Generación XX	Generación 62
Fecha límite de inscripción	fecha_limite_inscripcion	a más tardar el XX de XXXX de XXXX	30 de Julio de 2026
Sesión informativa	sesion_informativa	el XX de XXXX a las XX:XX	10 de Junio a las 18:00
Fecha genérica	fecha_generica	XX de XXXX de XXXX	20 de Julio de 2026
Fecha simple	fecha_simple	XX de XXXX	20 de Julio
Mes y año	mes_anio	XXXX de XXXX	Julio de 2026
Hora	hora	XX:XX	18:00
Arquitectura de Resolución
text
1. Chunk almacenado en FAISS:
   "Te informo que fuiste asignado(a) a la fecha1. Cursarás el Módulo 1 fecha2. Ingresa a: url1"

2. Resolución en tiempo real (rag/core.py → _rag_process):
   - Cargar JSONs (con caché en memoria)
   - Reemplazar fecha1 → "Generación 62"
   - Reemplazar fecha2 → "del 13 de Julio al 15 de Agosto de 2026"
   - Reemplazar url1 → "https://prepaenlinea.sep.gob.mx/mi-comunidad/"

3. Respuesta final:
   "Te informo que fuiste asignado(a) a la Generación 62. Cursarás el Módulo 1 del 13 de Julio al 15 de Agosto de 2026. Ingresa a: https://prepaenlinea.sep.gob.mx/mi-comunidad/"
Beneficios
Beneficio	Explicación
✅ Actualización centralizada	Cambiar una URL en un JSON actualiza todas las respuestas
✅ Sin reindexación	No es necesario regenerar embeddings ni reconstruir FAISS
✅ Mantenimiento simplificado	Un solo archivo por tipo de elemento dinámico
✅ Trazabilidad	Los placeholders mantienen la referencia al elemento original
✅ Rendimiento	Caché en memoria de los JSONs para resolución rápida
✅ Flexibilidad	Se pueden agregar nuevas URLs o fechas sin modificar los chunks
Archivos JSON
Archivo	Ubicación	Propósito
mapeo_urls_global.json	data/	Mapeo de URLs a placeholders
mapeo_fechas_completo.json	data/	Mapeo de fechas a placeholders con 10 categorías

🧠 Modelo Generativo: Groq con GPT OSS 120B
Configuración Actual
python
# models/groq_wrapper.py
config = {
    'model': 'openai/gpt-oss-120b',
    'temperature': 0.3,
    'max_tokens': 1024,
    'top_p': 1,
    'max_retries': 3,
    'backoff_factor': 2  # espera: 1, 2, 4 segundos
}
Ventajas sobre Llama 3.3 70B
Aspecto	Llama 3.3 70B	GPT OSS 120B	Impacto
Costo por 1M tokens	$0.79 USD	$0.79 USD	Mismo costo
Capa gratuita real	✅ Sin tarjeta	✅ Sin tarjeta	Sin barrera
Velocidad	~394 tok/seg	~420 tok/seg	Similar
Modelo	Open-source	Open-source	Transparente
Latencia TTFT	~50-150ms	~50-150ms	Similar
Tasa de Éxito
95% (19/20 preguntas en test automatizado)

Única falla: intent faltante en "netiqueta_reglas_comunicacion" (mayúsculas)

Modelos Alternativos
Modelo	Velocidad	Caso de uso
openai/gpt-oss-120b	~420 tok/seg	Default, alta calidad
openai/gpt-oss-20b	~1000 tok/seg	Preguntas simples, mayor velocidad
qwen/qwen3.6-27b	—	Alternativa al 70B deprecado

> ⚠️ **Migración Groq (Sep 2026)**: `llama-3.3-70b-versatile` y `llama-3.1-8b-instant`
> se retiraron de GroqCloud el 2026-08-16 (deprecación confirmada por correo/email
> "Compound/LLama deprecation"). NO usar estos IDs en llamadas a Groq — devuelven
> HTTP 404. Migrar a `openai/gpt-oss-120b` (recomendado) o `openai/gpt-oss-20b`. El
> CI (`sync-to-hf.yml`) ya usa `openai/gpt-oss-120b`.

🔗 LangChain y Orquestación
Versión Actual: 0.2.0
Componentes Implementados
1. Capa de Orquestación (langchain_layer/)
python
# langchain_layer/wrappers.py
class LangChainRAGWrapper:
    def __init__(self, rag_system, memory_enabled: bool = True, mongodb_enabled: bool = True):
        self.rag_system = rag_system
        self.memory = ConversationBufferMemory()
        self.date_extractor = DateExtractor()
        self.llm = GroqWrapper()
        self.conv_service = ConversationService()
        self.metrics_service = MetricsService()
    
    async def query_with_memory(self, question: str, session_id: str, user_id: str = None, conversation_id: str = None):
        # 1. Sanitizar entrada (InputSanitizer)
        # 2. Cargar historial desde MongoDB (_load_history)
        # 3. Detectar si es pregunta general (fecha/saludo/presentación) → sin RAG
        # 4. Separar pregunta REAL para retrieval (¡CRÍTICO!)
        pregunta_retrieval = question  # SIN contaminar con historial/fecha
        # 5. Obtener respuesta del RAG
        response = self.rag_system.process_query(pregunta_retrieval)
        # 6. Enriquecer con contexto temporal
        response = self._mejorar_respuesta_con_fecha(response, question)
        # 7. Guardar en memoria local
        self.memory.save_context({"input": question}, {"output": response})
        # 8. Persistir en MongoDB en BACKGROUND (asyncio.create_task, no bloquea)
        await self._schedule_save(...)
        return response
2. Endpoints Disponibles
Endpoint	Método	Propósito
/chat	POST	Principal: LangChain + memoria + conciencia temporal + placeholders + MongoDB (acepta `question` o `message`)
/chat/v2	POST	LangChain + memoria (versión alternativa)
/chat/clear_memory	POST	Limpiar memoria por session_id
/feedback	POST	Registrar feedback (persistido en MongoDB; acepta `user_rating` o `is_helpful`)
/analytics	GET	Estadísticas combinadas MongoDB: conversaciones, salud del sistema y feedback
/api/docs	GET	Documentación Swagger/OpenAPI
3. Estructura de Archivos
text
langchain_layer/
├── __init__.py          # Versión 0.2.0
├── config.py            # Configuración (max_tokens, TTL, timezone)
└── wrappers.py          # Wrapper async con memoria + conciencia temporal + MongoDB

⏰ Conciencia Temporal
Fecha de Implementación: 16 de Junio de 2026
¿Qué permite?
Saber qué fecha es hoy (zona horaria configurable vía TIMEZONE)

Extraer fechas automáticamente de documentos RAG

Comparar fechas y determinar si un evento ya pasó, está vigente o no ha comenzado

Responder proactivamente: "Ya terminó (hace 132 días)", "¡Está vigente! Faltan 5 días"

Componente: DateExtractor
python
# scripts/extract_dates.py
class DateExtractor:
    def extract_dates(self, text: str) -> List[Dict]:
        """Extrae fechas en formatos:
        - '26 de enero de 2026'
        - 'del 26 de enero al 4 de febrero'
        - '26/01/2026'
        - '26-ene-2026'
        """
        # Patrones regex para múltiples formatos
        # Retorna lista de fechas con metadata
Formatos Soportados
Formato	Ejemplo	Resultado
Fecha individual	"26 de enero de 2026"	{fecha: '2026-01-26', tipo: 'fecha'}
Rango de fechas	"del 26 de enero al 4 de febrero"	{inicio: '2026-01-26', fin: '2026-02-04', tipo: 'rango'}
Detección de Preguntas Generales
El sistema identifica preguntas que NO requieren RAG:

python
palabras_clave = ["fecha", "hoy", "día", "hola", "quién eres", "cómo te llamas"]
if any(palabra in question.lower() for palabra in palabras_clave):
    # Usar LLM directamente SIN RAG
    response = llm.generate(f"Hoy es {fecha_hoy}. Pregunta: {question}")
Ejemplos de Respuestas
Pregunta	Respuesta
"¿Qué fecha es hoy?"	"Hoy es martes 16 de junio de 2026"
"¿Cuándo es la convocatoria?"	"Registro: 26 ene - 4 feb. 📌 Ya terminó (hace 132 días)"
"¿Ya puedo registrarme?"	"No encontré registro vigente. Ya terminó (hace 132 días)"
"Hola"	"¡Hola! ¿En qué puedo ayudarte?"

🧠 Sistema de Memoria
Tipo: ConversationBufferMemory (LangChain)
Características
Almacena historial completo de la conversación

Persistencia: En RAM (volátil, aceptable para HF Spaces)

Inyección automática: El historial se añade al prompt en cada consulta

Limpiable: Endpoint /chat/clear_memory para resetear por sesión

Configuración
python
# langchain_layer/config.py
class LangChainConfig:
    MAX_HISTORY_TOKENS: int = 2000
    MEMORY_KEY: str = "chat_history"
    TIMEZONE: str = "America/Mexico_City"
Ejemplo de Uso
text
Usuario: "¿Qué es Prepa en Línea?"
Bot: "Es un servicio educativo gratuito de nivel medio superior"

Usuario: "¿Es gratis?"  ← ¡Sabe que se refiere a Prepa en Línea!
Bot: "Sí, el servicio es completamente gratuito"

Usuario: "¿Cuánto dura?"
Bot: "El trayecto completo dura 2 años y 6 meses"  ← Contexto preservado
🔄 CI/CD Pipeline (GitHub Actions)
Workflow: CI/CD - Test & Deploy
Jobs del Pipeline
Test & Validate (test):

Verifica archivos críticos

Prueba conexión a Groq (real)

Prueba importación de API

Prueba GroqWrapper

Prueba RAGSystem (465 vectores FAISS)

Deploy to Hugging Face (deploy):

Solo si TODAS las pruebas pasan

Despliegue automático a HF Spaces

Variables de Entorno (Secrets)
GROQ_API_KEY: Autenticación para pruebas con Groq

HF_TOKEN: Autenticación para deploy a HF Spaces

Métricas
Métrica	Valor
Duración promedio	1 min 38 seg
Detección de errores	~2 minutos
Tasa de éxito	100%
Beneficios
Aspecto	Antes	Después
Detección de errores	En producción	Antes del despliegue
Tiempo de recuperación	15-30 min manual	0 min
Despliegue a HF Spaces	Manual	Automático

## 📡 Monitoreo Continuo en Producción

Workflow: `monitor.yml` — cada 10 minutos

### Jobs del Pipeline

**Health Check** (health-check):
1. 💚 **Health Check** — Endpoint `/health` (HTTP 200 = OK)
2. 🔍 **Verificar endpoint `/chat`** — Confirma que existe (HTTP 200 o 405)
3. 📱 **Alerta de Caída** — Se envía a Telegram si health != 200
4. 📱 **Alerta de Recuperación** — Se envía en ejecución manual cuando el servicio ya responde
5. 📊 **Resumen de Ejecución** — Siempre se imprime

### Características Clave

| Característica | Detalle |
|---|---|
| Token consumption | **0 tokens** — solo usa HTTP status codes |
| Frecuencia | Cada 10 minutos (cron: `*/10 * * * *`) |
| Timeout por paso | 15s health, 10s chat endpoint |
| Timeout total del job | 3 minutos |
| Alertas | Telegram vía `sendMessage` API |
| Notificación de caída | Automática si health != 200 |
| Notificación de recuperación | Solo en `workflow_dispatch` manual |

### Variables de Entorno Requeridas (Secrets)

| Variable | Propósito |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | Chat ID para recibir alertas |

### Mensaje de Alerta (Ejemplo)

```text
🚨 ALERTA: CHATBOT CAÍDO O DEGRADADO
• Health Check: ❌ Falló (HTTP 503)
• Endpoint /chat: ❌ Falló (HTTP 000)
📅 Fecha: 14/07/2026 15:30:00
💰 Tokens consumidos en monitoreo: 0
🔧 Acción Requerida: Revisar logs en HF Spaces
```

### Ventajas

| Aspecto | Beneficio |
|---|---|
| ✅ 0 tokens consumidos | No afecta cuota gratuita de Groq |
| ✅ Detección temprana | Máximo 10 min para detectar caída |
| ✅ Sin dependencias externas | Solo curl + API de Telegram |
| ✅ Notificación inmediata | Telegram en tiempo real |
| ✅ Historial en GitHub Actions | Logs disponibles en cada ejecución |

📝 Guías de Estilo de Código
Imports
stdlib primero (os, sys, json, logging, time, datetime, pathlib)

Terceros (fastapi, pydantic, numpy, faiss, groq, langchain)

Locales (from config.xxx, from rag.xxx, from models.xxx)

python
import os
import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from langchain.memory import ConversationBufferMemory

from config.settings import settings
from rag.core import RAGSystem
from langchain_layer.wrappers import LangChainRAGWrapper
Nombramiento
Clases: PascalCase (ej: RAGSystem, GroqWrapper, DateExtractor)

Funciones/métodos: snake_case (ej: process_query, _clean_query)

Variables: snake_case (ej: query_embedding, top_k)

Constantes: UPPER_SNAKE_CASE (ej: TOP_K_RESULTS)

Métodos privados: Prefijo _ (ej: _rag_process)

Logger: logger = logging.getLogger(__name__) (siempre al inicio)

Archivos: snake_case.py (ej: groq_wrapper.py)

Type Hints
Obligatorios para todas las firmas de funciones

Usar from typing import List, Dict, Any, Optional, Tuple

Usar Optional[str] en lugar de str | None

python
def process_query(self, query: str) -> Tuple[str, bool, float, list]:
def embed_batch(self, texts: List[str], is_passage: bool = False) -> np.ndarray:
Docstrings
Estilo Google con secciones Args: y Returns:

Obligatorio para métodos públicos; opcional para privados

python
def generate(self, query: str, context: str = "", max_length: int = 256) -> str:
    """Generate a response for the given query.
    
    Args:
        query: User question/query.
        context: Retrieved context from RAG system (optional).
        max_length: Maximum tokens to generate.
    
    Returns:
        Generated response string.
    """
Manejo de Errores
python
try:
    # operación
    logger.info(f"Operación exitosa: ...")
except Exception as e:
    logger.error(f"Operación falló: {e}", exc_info=True)
    return "Mensaje de fallback para el usuario"
Siempre loguear con logger.error(...)

Usar exc_info=True para errores inesperados

Los endpoints de API deben levantar HTTPException

Degradación graceful: si un componente falla, retornar mensaje amigable

Logging
Logger a nivel de módulo: logger = logging.getLogger(__name__)

Prefijos estructurados:

✅ Éxito

❌ Error

⚠️ Advertencia

📩 Mensaje entrante

📤 Mensaje saliente

🔍 Debug

🧠 Memoria

⏰ Conciencia temporal

🔗 Placeholder resuelto

API Conventions
FastAPI app en api/main.py, rutas en api/endpoints.py

Modelos Pydantic en config/models.py

CORS: allow_origins=["*"]

Headers estándar: X-User-ID, X-Conversation-ID, X-Message-ID

/chat retorna JSON con: response, sources, is_rag_response, confidence

/health retorna {"status": "healthy"}

Nuevo: /chat/v2 y /chat/clear_memory para LangChain

RAG Pipeline Conventions
Embeddings: intfloat/multilingual-e5-small (384 dims)

"query: " para consultas de usuario

"passage: " para chunks indexados

Vector store: FAISS (FlatL2) en data/vector_store/

Retrieval: OptimizedRetriever con query expansion, sinónimos, multi-query

Generator: GemmaGenerator (nombre heredado) → GroqWrapper

Intents: JSON-based, prioridad sobre RAG para saludos/despedidas

Memory: LangChain ConversationBufferMemory en langchain_layer/

Placeholders: Resolución en rag/retriever.py con caché en memoria de JSONs

Placeholder Conventions
URLs: url1, url2, url3 en mapeo_urls_global.json

Fechas: fecha1, fecha2 en mapeo_fechas_completo.json (10 categorías)

Resolución: En rag/retriever.py → resolver_placeholders(text)

Caché: Singleton con carga en memoria al iniciar

Actualización: Solo modificar JSONs, NO reindexar FAISS

CI/CD Testing Guidelines
Pruebas automáticas en cada push: 4 pruebas clave

Conexión a Groq (API key + modelo)
Importación de API (sin errores de sintaxis)
GroqWrapper (inicialización correcta)
RAGSystem (carga de 465 vectores FAISS)
Nunca desplegar sin pruebas: Si falla una prueba, se cancela deploy

Variables en GitHub Secrets: GROQ_API_KEY, HF_TOKEN

📁 Estructura del Proyecto
text
Chatbot-RAG-Fuente-Base/
├── .agents/                        # Skills y agentes de OpenCode
│   └── skills/
├── .github/
│   └── workflows/                  # CI/CD pipeline
│       ├── test-deploy.yml              # CI/CD: Test & Deploy
│       └── monitor.yml                  # 📡 Monitoreo continuo (cada 10 min)
├── api/
│   ├── endpoints.py                # Rutas API (/documents) + mongodb_router (/chat, /feedback, /analytics)
│   ├── main.py                     # FastAPI app principal (endpoints con MongoDB)
│   └── models.py                   # 🆕 Schemas Pydantic compatibles (question/message, user_rating/is_helpful)
├── config/
│   ├── models.py                   # Pydantic models
│   └── settings.py                 # Configuración con pydantic-settings
├── data/
│   ├── vector_store/               # FAISS index (465 vectores)
│   │   ├── faiss_index
│   │   └── metadata.pkl
│   ├── mapeo_urls_global.json      # 🆕 Mapeo de URLs a placeholders
│   ├── mapeo_fechas_completo.json  # 🆕 Mapeo de fechas a placeholders (10 categorías)
│   ├── intents.json
│   └── menu.json
├── evaluation/
│   ├── automated_evaluator.py      # Pruebas automatizadas
│   ├── generate_dashboard.py
│   └── test_set.json
├── langchain_layer/                # Orquestación LangChain
│   ├── __init__.py                 # Versión 0.2.0
│   ├── config.py                   # Configuración (memoria, timezone)
│   └── wrappers.py                 # Wrapper async con memoria + MongoDB
├── models/
│   ├── groq_wrapper.py             # Principal (GPT OSS 120B)
│   ├── gemini_wrapper.py           # DEPRECATED (fallback)
│   └── ollama_wrapper.py
├── mongodb/                        # 🆕 Persistencia MongoDB Atlas
│   ├── __init__.py                 # Exports: conexión, modelos, repositorio, servicios
│   ├── connection.py               # MongoDBConnection — singleton async (motor)
│   ├── models.py                   # MessageRole, ConversationCreate, MetricCreate, FeedbackCreate
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── logging_middleware.py   # 🆕 LoggingMiddleware — registra solicitudes HTTP en colección logs
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── base_repository.py      # BaseRepository[T] — CRUD genérico
│   └── services/
│       ├── __init__.py             # Exports: clases y funciones de servicio
│       ├── conversation_service.py # ConversationService — conversaciones + stats + daily + search
│       ├── metrics_service.py      # MetricsService — métricas por endpoint + salud del sistema
│       ├── feedback_service.py     # FeedbackService — feedback + stats
│       └── cache_service.py        # 🆕 RAGCacheService — caché MD5 + TTL + hit_count
├── rag/
│   ├── core.py                     # RAGSystem principal
│   ├── embeddings.py
│   ├── generator.py
│   ├── optimized_retriever.py
│   └── retriever.py                # 🆕 Función resolver_placeholders()
├── scripts/
│   ├── extract_dates.py            # Extractor automático de fechas
│   ├── load_chunks_to_rag.py
│   ├── test_mongodb_connection.py  # 🆕 Probar conexión a Atlas
│   ├── test_services.py            # 🆕 Probar repositorios y servicios
│   ├── test_integration.py         # 🆕 Prueba de integración (API + MongoDB con StubRAG)
│   ├── test_cache.py               # 🆕 Prueba del RAGCacheService
│   ├── retention_policy.py         # 🆕 Política de retención (limpieza por días)
│   └── generate_dashboard_report.py # 🆕 Reporte de dashboard (JSON)
├── security/                       # 🆕 Sistema de seguridad
│   ├── __init__.py                 # Exports: InputSanitizer, SecurityMonitor
│   ├── sanitizer.py                # InputSanitizer — 3 categorías de detección
│   └── monitor.py                  # SecurityMonitor — deque en RAM, alertas Telegram
├── static/
│   └── index.html
├── tests/
│   ├── test_api.py
│   └── test_rag.py
├── utils/
│   └── log_capture.py
├── .env                             # Variables de entorno
├── AGENTS.md                        # Este archivo
├── MONGODB_GUIDE.md                 # 📖 Guía del equipo: MongoDB, consultas, dashboard y mantenimiento
├── app.py                           # Entry point alternativo
├── Dockerfile
├── requirements.txt
└── verificar_memoria.py             # Verificación de memoria

📊 Estado Actual del Sistema
Componente	Estado	Versión/Detalle
Orquestación	✅ Operativo	LangChain v0.2.0
Modelo	✅ Operativo	Groq GPT OSS 120B
Memoria	✅ Activa	ConversationBufferMemory
Conciencia Temporal	✅ Activa	DateExtractor v1.0
Placeholders	✅ Activo	URLs + 10 categorías de fechas
Base de Conocimiento	✅ 465 chunks	8 tipos de documentos
Control Escolar	✅ Integrado	357 chunks (CE0001-CE0357)
CI/CD Pipeline	✅ Operativo	1:38 min promedio
Despliegue	✅ Automático	HF Spaces
Endpoint /chat	✅ Con LangChain	Memoria + tiempo + placeholders
Endpoint /chat/v2	✅ Con memoria	Alternativo
Endpoint clear_memory	✅ Activo	Limpieza por sesión
Documentación API	✅ Swagger	/api/docs
Monitoreo Continuo	✅ Activo	Health check c/10 min + alertas Telegram
Persistencia MongoDB	✅ Activo	Conversaciones, métricas y feedback en Atlas
Caché de respuestas	✅ Activo	RAGCacheService (MD5 + TTL 24h + hit_count)
Logging HTTP	✅ Activo	LoggingMiddleware → colección logs
Retención de datos	✅ Activo	retention_policy.py + /admin/cleanup
Seguridad (Sanitización)	✅ Activo	InputSanitizer — 3 categorías
Seguridad (Monitoreo)	✅ Activo	SecurityMonitor — deque 1000 en RAM
Seguridad (Alertas)	✅ Activo	Telegram para severidad high/critical
Costos mensuales	✅ $0 USD	Capa gratuita Groq
Métricas de Rendimiento
Métrica	Valor
Tasa de éxito (RAG)	95% (19/20)
Latencia promedio	~2.05 segundos
Tokens por consulta	~1,725
Chunks en FAISS	465
Extracción de fechas	100% automática
Precisión temporal	100% en pruebas
Placeholders resueltos	100% en tiempo real

🗺️ Roadmap y Próximos Pasos
Prioridad	Acción	Impacto esperado	Estado
1	Agregar patrón "mayúsculas" a intent netiqueta	Alcanzar 100% de éxito	⏳ Pendiente
2	Evaluación con 15 usuarios	Validación en entorno real	⏳ Pendiente
3	Cache de respuestas frecuentes	Reducir consumo API y latencia	⏳ Pendiente
4	ConversationSummaryMemory	Resumir historial largo, reducir tokens	⏳ Pendiente
5	Versionar JSONs (URLs y fechas)	Control de cambios	⏳ Pendiente
6	Endpoint para actualización en caliente	Actualizar JSONs sin reiniciar	⏳ Pendiente
7	Agente que decida cuándo usar RAG	Optimizar tokens automáticamente	⏳ Pendiente
8	Monitoreo de placeholders no resueltos	Detectar placeholders sin resolver	⏳ Pendiente
9	Balancear contenido del vector store	Más documentos normativos	⏳ Pendiente
10	Notificaciones Telegram/Slack en CI/CD	Visibilidad inmediata de fallos	✅ Completado

## 🛡️ Sistema de Seguridad

Implementado el 18 de Julio de 2026 — Fase 1 completa.

### Componentes

| Componente | Archivo | Propósito |
|---|---|---|
| `InputSanitizer` | `security/sanitizer.py` | Detecta y bloquea 3 categorías de amenazas |
| `SecurityMonitor` | `security/monitor.py` | Almacena incidentes en deque (máx 1000, solo RAM) |
| Endpoints de visibilidad | `api/main.py` | `/security/stats`, `/security/incidents`, `/security/validate` |

### InputSanitizer — 3 Categorías de Detección

| Categoría | Ejemplos detectados | Severidad |
|---|---|---|
| **Inyección directa** | `<script>`, `javascript:`, `<iframe>`, `DROP TABLE`, `ALTER`, `TRUNCATE`, `EXEC`, `xp_cmdshell`, `eval(`, `exec(`, `os.system(` | critical |
| **Escape de contexto** | "ignora instrucciones", "olvida reglas", "no sigas reglas", "muéstrame tu prompt", "dime tu prompt", "override security", "bypass restrictions" | high/critical |
| **Ofuscación** | Base64 largo, hex encoding, unicode escape, HTML entities, URL encoding, non-ASCII excesivo | medium/high |

### SecurityMonitor

- `deque(maxlen=1000)` — solo RAM, sin disco, sin DB
- `log_incident()` → almacena timestamp, tipo, severidad, snippet, session_id, IP
- `get_stats()` → totales por severidad y tipo, incidentes última hora, % capacidad
- `get_recent_incidents(limit, min_severity)` → incidentes recientes con filtro
- Cache de stats con TTL de 2 segundos para alto rendimiento

### Integración en /chat

La sanitización se ejecuta **al inicio** del endpoint `POST /chat` en `api/main.py:270`:

1. `InputSanitizer.sanitize(request.message)` analiza el texto entrante
2. Si hay amenazas → se registran en el monitor y se retorna:
   ```json
   {
     "response": "⚠️ No puedo procesar esa solicitud.",
     "sources": [],
     "is_rag_response": false,
     "confidence": 0.0,
     "security_blocked": true
   }
   ```
3. Si es seguro → flujo normal con RAG + memoria + placeholders

El bloqueo es **silencioso**: no lanza excepciones HTTP, solo logs con prefijos 🚨 y 🔒.

### Alertas Telegram para Seguridad

| Severidad | Acción |
|---|---|
| `low`, `medium` | Solo log (logger.warning) |
| `high`, `critical` | Log (logger.error) + alerta Telegram vía `sendMessage` API con Markdown |
| Sin token/config | Degradación silenciosa (logger.debug) |

Mensaje de alerta incluye: tipo, severidad, sesión, IP, fragmento, fecha. **0 tokens de Groq consumidos.**

### Endpoints de Visibilidad

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/security/stats` | Estadísticas de incidentes en tiempo real |
| GET | `/security/incidents?limit=50&min_severity=low` | Incidentes recientes |
| POST | `/security/validate?text=...` | Validar texto contra el sanitizer (pruebas manuales) |

### Prompt Reforzado

El system prompt de Groq en `models/groq_wrapper.py` se actualizó con:
- Prohibición explícita de revelar instrucciones internas
- Respuesta fija ante intentos de jailbreak: *"No puedo procesar esa solicitud"*
- Instrucción de ignorar comandos de override de seguridad

### Métricas

| Métrica | Valor |
|---|---|
| Overhead por consulta | ~2ms |
| Incidentes en memoria | Máx 1,000 (deque) |
| Alertas Telegram | Solo high/critical |
| Tokens consumidos | 0 (solo HTTP + logs) |
| Persistencia en disco | Ninguna |

📌 **Convenciones de Seguridad**
- SIEMPRE sanitizar antes de RAG
- NUNCA dejar pasar consultas con severidad ≥ medium
- SIEMPRE usar el monitor singleton vía `get_monitor()`
- NUNCA guardar incidentes en disco
- SIEMPRE responder con mensaje amigable, no con error HTTP

---

## 🍃 Persistencia MongoDB

Implementado el 4 de Agosto de 2026 — Persistencia de conversaciones, métricas y feedback en MongoDB Atlas.

### Componentes

| Componente | Archivo | Propósito |
|---|---|---|
| `MongoDBConnection` | `mongodb/connection.py` | Singleton async (motor) — conexión + índices |
| `BaseRepository[T]` | `mongodb/repositories/base_repository.py` | CRUD genérico por colección |
| `ConversationService` | `mongodb/services/conversation_service.py` | Conversaciones + stats + daily + search |
| `MetricsService` | `mongodb/services/metrics_service.py` | Métricas por endpoint + salud del sistema |
| `FeedbackService` | `mongodb/services/feedback_service.py` | Feedback + stats |
| Schemas Pydantic | `api/models.py` | `ChatRequest`/`ChatResponse`/`FeedbackRequest` (backward compat) |

### Colecciones e Índices

| Colección | Índices (creados en `connect()`) |
|---|---|
| `conversations` | `conversation_id` (único), `session_id`, `user_id` |
| `metrics` | `request_timestamp` (desc), `session_id`, `endpoint` |
| `feedback` | `conversation_id`, `session_id` |
| `users`, `sessions`, `rag_cache`, `logs` | Índices según configuración |

### Flujo de Persistencia en /chat

1. `LangChainRAGWrapper.query_with_memory()` es **async** (`langchain_layer/wrappers.py`)
2. `_load_history()` carga historial previo desde MongoDB (fallback a memoria local si falla)
3. La respuesta se genera primero; luego `_schedule_save()` persiste **en background** con `asyncio.create_task()`, sin bloquear la respuesta al usuario
4. Guarda `ConversationCreate` + `MetricCreate` (con tokens estimados `len(text)//4` y `latency_ms`)

### Configuración (config/settings.py)

```python
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "chatbot_rag_db")
MONGODB_ENABLE_LOGGING: bool = True
MONGODB_MAX_POOL_SIZE: int = 50
MONGODB_TIMEOUT_MS: int = 5000
```

### Endpoints con MongoDB

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/chat` | Acepta `question` o `message` (legacy); persiste conversación + métrica |
| POST | `/feedback` | Acepta `user_rating`/`message_index` o `is_helpful`/`message_id` (legacy) |
| GET | `/analytics?session_id=&days=` | Conversaciones + salud del sistema + feedback |

### Compatibilidad Backward

- `api/models.py` mapea `message` → `question` vía `model_validator`
- `/feedback` deriva `user_rating` desde `is_helpful` si el campo nuevo no viene
- `mongodb_router` en `api/endpoints.py` implementa el schema limpio (usado por `test_integration.py`); los endpoints de producción están en `api/main.py`

### Scripts de Prueba

```bash
python -m scripts.test_mongodb_connection   # Conexión a Atlas
python -m scripts.test_services             # Repositorios y servicios
python -m scripts.test_integration          # Integración completa (StubRAG, httpx ASGI)
```

- `test_integration.py` usa `StubRAG` (evita FAISS/Groq), monta `mongodb_router` en un FastAPI temporal y limpia sus datos al final.

📖 **Guía para el equipo**: consultar `MONGODB_GUIDE.md` para entender MongoDB (qué es y por qué se usa), la estructura de colecciones, cómo consultar la base desde Python, ejemplos de consultas comunes, cómo interpretar el dashboard y la guía de mantenimiento/troubleshooting.

📌 **Convenciones MongoDB**
- SIEMPRE usar el singleton `MongoDBConnection()` para conectarse
- NUNCA bloquear la respuesta con el guardado → usar `_schedule_save()` (background)
- SIEMPRE degradar con graceful si MongoDB falla (la respuesta al usuario no debe romperse)
- Los servicios exponen clases (`ConversationService`, etc.) y funciones; usar clases en los endpoints

---

## ⚡ Caché, Logging y Retención

Implementado el 4 de Agosto de 2026 — Caché de respuestas RAG, logging HTTP y política de retención.

### RAGCacheService (`mongodb/services/cache_service.py`)

| Método | Descripción |
|---|---|
| `_query_hash(query, context)` | Hash MD5 de consulta + contexto (normaliza minúsculas/espacios) |
| `get_cached_response(query, context)` | Retorna entrada si existe y no expiró; incrementa `hit_count` |
| `cache_response(query, response, sources, confidence, context)` | Guarda con `expires_at = now + TTL` |
| `get_stats()` | Totales de entradas y hits |
| `cleanup_expired()` | Elimina entradas expiradas |

- TTL configurable vía `RAG_CACHE_TTL_HOURS` (default 24h); desactivable con `RAG_CACHE_ENABLED=false`
- MongoDB devuelve datetimes **naive** (UTC implícito): normalizar con `_ensure_utc()` antes de comparar con `datetime.now(timezone.utc)`
- Guardado con `replace_one(..., upsert=True)` para evitar duplicados por `query_hash`

### LoggingMiddleware (`mongodb/middleware/logging_middleware.py`)

- Registrado en `api/main.py` vía `app.add_middleware(LoggingMiddleware)`
- Persiste en colección `logs`: timestamp, method, path, client_ip, user_agent, status_code, response_time_ms, body (máx 1000 chars)
- Extrae `session_id` de header `X-Session-ID`, query param o body (fallback `"unknown"`)
- **No bloquea la respuesta**: falla de MongoDB solo loguea `logger.debug`
- Captura body con `await request.body()` (no consume el stream para el endpoint)

### Política de Retención (`scripts/retention_policy.py`)

| Colección | Campo timestamp | Días |
|---|---|---|
| `conversations` | `created_at` | 90 (`RETENTION_CONVERSATIONS_DAYS`) |
| `metrics` | `request_timestamp` | 30 (`RETENTION_METRICS_DAYS`) |
| `logs` | `timestamp` | 14 (`RETENTION_LOGS_DAYS`) |
| `feedback` | `created_at` | 180 (`RETENTION_FEEDBACK_DAYS`) |

- `run_retention(dry_run=True)` es importable y retorna resumen por colección
- CLI: `python scripts/retention_policy.py [--dry-run]`
- Endpoint: `POST /admin/cleanup?dry_run=true` (protegido por `X-Admin-Key` o `admin_key` si `ADMIN_API_KEY` está configurada)

### Reporte Dashboard (`scripts/generate_dashboard_report.py`)

- `generate_report(days=7)` genera estadísticas diarias (7 días), salud del sistema, feedback, cache y resumen ejecutivo
- Guarda en `data/dashboard_report.json`
- Endpoint: `GET /admin/dashboard-report?days=7`

### Endpoints de Administración (`api/main.py`)

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/admin/cleanup?dry_run=` | Ejecuta retención (protegido por API key opcional) |
| GET | `/admin/dashboard-report?days=` | Genera reporte JSON |

`_check_admin_key()` valida header `X-Admin-Key` o query `admin_key` contra `ADMIN_API_KEY`; si la key no está configurada, el endpoint es abierto.

---

📌 Notas Finales
Convenciones Importantes
NUNCA modificar el RAG system para inyectar memoria o fechas

SIEMPRE usar LangChain para orquestación

SIEMPRE mantener endpoint /chat como fallback

SIEMPRE usar pregunta_retrieval = question (SIN historial ni fechas)

NUNCA hardcodear URLs o fechas en los chunks → usar placeholders

SIEMPRE actualizar JSONs en data/ para cambios de URLs/fechas

Flujo de Actualización de Placeholders
Cambia URL o fecha en data/mapeo_urls_global.json o data/mapeo_fechas_completo.json

NO necesitas reindexar FAISS

El sistema resuelve los placeholders en tiempo real

Todas las respuestas se actualizan automáticamente

Enlaces Útiles
API Docs: https://ecotecds-Chatbot-RAG-v4.hf.space/api/docs

Drive (Control Escolar): https://drive.google.com/drive/folders/1F-4jh_OQKukr5QF24LtN7yadjhKLTpwJ

Drive (Documentación general): https://drive.google.com/drive/folders/1dL29njdNFeeLCTo5BpSj5k9IwS9j-DNC

Última actualización: 4 de Agosto de 2026
Versión del AGENTS.md: 3.6.0

## Learned User Preferences
- Prefer silent graceful degradation (debug logging, no user-facing errors) for non-critical features like Telegram notifications
- Maintain existing CSS design/style when adding new dashboard sections; reuse `--azul-*` / `--verde*` / `--rojo*` CSS variables
- Dashboard SLA thresholds should be clearly documented and visually color-coded (green/yellow/red)

## Learned Workspace Facts
- FAISS `index.search()` returns tuple `(distances, indices)` where distances is a 2D numpy array; always unpack and access scalars via `array[0][0]`, never compare an array slice directly in an `if`
- Token stats from `get_token_stats()` live in a separate dict from the `metrics` dict; merge them via `{**metrics, "token_porcentaje": porcentaje}` when SLA needs token usage data
- `send_telegram_alert()` uses 3 retries with exponential backoff (2s, 4s, 8s), 30s timeout, and only `logger.debug()` — never interrupts dashboard generation on failure
- Dashboard HTML is one large Python f-string; JavaScript braces must be doubled (`{{` / `}}`) and all Python variable substitutions use single braces `{var}`
- `requests` is already in `requirements.txt`; no install needed for Telegram integration
- Dashboard has SLA (response P95, success rate, not-found rate, token usage) and ROI (time/cost savings vs human agent at $15/hr, 10 min/query) tabs exportable to PDF via html2pdf.js CDN
- Monitoring workflow (`monitor.yml`) uses `curl` HTTP status codes only — 0 Groq tokens consumed; runs every 10 min via `cron: '*/10 * * * *'`
- Telegram alerts use `sendMessage` API with Markdown parse mode; alert on health != 200, recovery only on `workflow_dispatch`
- Monitor has 5 steps: Health Check, Chat Endpoint Check, Failure Alert, Recovery Alert, Execution Summary; total job timeout is 3 minutes
- Secrets for monitoring: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` — these are GitHub Actions secrets, not in `.env`
- Security sanitizer in `security/sanitizer.py` uses 3 categories of regex patterns (injection, escape, obfuscation) with severity ordering: none < low < medium < high < critical
- `InputSanitizer.sanitize()` is called at the top of `/chat` endpoint in `api/main.py` before any RAG processing; blocked queries return JSON with `security_blocked: true`
- `SecurityMonitor` in `security/monitor.py` uses `deque(maxlen=1000)` — purely in-memory, no disk I/O, singleton via `get_monitor()`
- Telegram alerts for security fire only on severity `high` or `critical`, reuse existing `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars, 0 Groq tokens consumed
- System prompt in `models/groq_wrapper.py` prohibits revealing internal instructions and has fixed jailbreak response
- MongoDB persistence uses `MongoDBConnection()` singleton (motor async); `_create_indexes()` runs only once per process via `_indexes_ready` flag
- `LangChainRAGWrapper.query_with_memory()` is `async def (question, session_id="default", user_id=None, conversation_id=None)`; saving to MongoDB goes through `_schedule_save()` → `asyncio.create_task()` so it never blocks the HTTP response
- `_load_history()` reads prior conversations from MongoDB (`get_conversations_by_session`) and falls back to in-memory `_session_memories` if MongoDB fails
- `_estimate_tokens()` approximates tokens as `max(1, len(text)//4)`; `MetricCreate` stores `latency_ms` and estimated `tokens_used`
- `api/models.py` provides backward compat: `ChatRequest` maps `message`→`question` via `model_validator`, `FeedbackRequest` derives `user_rating` from `is_helpful`
- `mongodb_router` in `api/endpoints.py` (clean `question`/`user_rating` schema) is NOT mounted in `api/main.py` — mounting would conflict with the existing `/chat`, `/feedback`, `/analytics` routes already defined there; it's the test target for `scripts/test_integration.py`
- `scripts/test_integration.py` uses a `StubRAG` (avoids FAISS/Groq) with httpx `ASGITransport` and a temporary FastAPI app, then deletes its test data (`itest-session-*`) at the end
- Local `import api.main` is blocked by a corrupted leftover TensorFlow 2.18.0 install (not in `requirements.txt`, so production/CI is unaffected); verify API changes via `scripts/test_integration.py` instead
- Run Python scripts with `PYTHONIOENCODING=utf-8` on Windows to avoid `UnicodeEncodeError` (cp1252) when printing emoji (✅)
- `RAGCacheService` stores entries in the `rag_cache` collection with `query_hash` = MD5 of normalized query (+ JSON context); `get_cached_response()` increments `hit_count` and returns None if the entry is expired (and deletes it)
- MongoDB returns naive datetimes (implicit UTC): always normalize with `_ensure_utc()` (or `.replace(tzinfo=timezone.utc)`) before comparing with `datetime.now(timezone.utc)`
- `LoggingMiddleware` (added in `api/main.py` via `app.add_middleware`) persists HTTP requests to the `logs` collection — body captured up to 1000 chars with `await request.body()`, never blocks the response, fails silently (logger.debug)
- Retention rules in `scripts/retention_policy.py` (`RETENTION_RULES`): conversations 90d / metrics 30d / logs 14d / feedback 180d, driven by `RETENTION_*_DAYS` env vars; `run_retention(dry_run=True)` is importable for the `/admin/cleanup` endpoint
- `/admin/cleanup` and `/admin/dashboard-report` are protected by `ADMIN_API_KEY` (optional): `_check_admin_key()` reads `X-Admin-Key` header or `admin_key` query param; empty key = endpoint open
- `scripts/generate_dashboard_report.py` → `save_report(days)` writes `data/dashboard_report.json`; `generate_report()` and `save_report()` are importable for the admin endpoint
- Keep a generated `data/dashboard_report.json` in the repo after running the report script (it's an expected artifact)
- `/user-dashboard` (`evaluation/generate_user_dashboard.py`) reads interactions and token stats from MongoDB (`conversations` + `metrics`), with `user_interactions.jsonl` / `token_usage*.jsonl` fallback via graceful degradation (`logger.debug` only); generation is async (`generate_user_dashboard_async`) and called with `await` from `api/main.py` startup + `/user-dashboard/refresh` — the sync `generate_user_dashboard()` wrapper uses `asyncio.run()` for CLI only
- `fetch_mongodb_interactions()` maps each `conversations` doc to interaction dicts pairing `user` msg → next `assistant` msg; `fuentes_usadas` read from `sources_used[].metadata.source_file`; `tokens_used` from message tokens (or `total_tokens`); timestamps normalized to UTC via `_ensure_utc()` before `.isoformat()` (avoid naive/aware serialization errors); interactions sorted globally by timestamp via `_interaction_sort_key`
- `ConversationMessage` carries per-turn `latency_ms`, `confidence_score`, `is_rag`, `sources_used` (assistant msg) so the dashboard shows accurate per-interaction values; `fetch_mongodb_interactions()` prefers per-message fields and falls back to conversation-level summary fields (`latency_ms`, `confidence_score`, `sources_used` reflect the **last** saved turn)
- `save_conversation()` ACCUMULATES: on an existing `conversation_id` it does a single atomic `update_one` with `{"messages.timestamp": {"$nin": [new_ts...]}}` + `$push`/`$set` — appends the new turn to the `messages` array without duplicating retries (same timestamps) and without losing concurrent turns (no read-modify-write race). Same `conversation_id` (e.g. the frontend's `web_interface`) therefore grows its history across turns instead of overwriting the last turn
- `fetch_mongodb_token_stats()` aggregates `metrics.tokens_used` since UTC midnight (tokens "hoy"); `fetch_mongodb_tokens_por_hora()` uses `$hour` on `request_timestamp`; Mongo token percent limit fixed at 100000 to match Groq free tier
- `ConversationService.get_recent_conversations(limit)` (added) returns latest conversations across all sessions sorted by `created_at` desc; exported in `mongodb/services/__init__.py` and `mongodb/__init__.py`
- `/api/logs` tries the `logs` collection first (synthesizing level from status_code: >=500 ERROR, >=400 WARNING, else INFO), falls back to `data/system_logs.jsonl`
- `scripts/test_dashboard_mongodb.py` verifies: metrics math, JSONL fallback generation, and Mongo mapping (inserts + cleans up `dash-itest-*`); skips Mongo checks if `MONGODB_URI` is the default `localhost`; run with `PYTHONIOENCODING=utf-8` on Windows
- `settings.DASHBOARD_USE_MONGODB` (default true) and `DASHBOARD_MONGODB_INTERACTIONS_LIMIT` (default 2000) control the `/user-dashboard` data source
- `MONGODB_GUIDE.md` is the team-facing reference (in Spanish) for MongoDB: what it is, collection structure, Python query examples, dashboard interpretation, and maintenance/troubleshooting; link to it from AGENTS.md whenever MongoDB topics are referenced