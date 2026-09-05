# MONGODB_IMPLEMENTACION.md — Resumen de Implementación MongoDB

Resumen técnico de **todo lo implementado** sobre persistencia en MongoDB Atlas para el Chatbot RAG de Prepa en Línea SEP.

📖 Para la guía operativa del equipo (qué es MongoDB, cómo consultarlo, dashboard, mantenimiento), ver **[MONGODB_GUIDE.md](./MONGODB_GUIDE.md)**. Este documento se enfoca en **qué se construyó y cómo funciona por dentro**.

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#-resumen-ejecutivo)
2. [Arquitectura del Paquete `mongodb/`](#-arquitectura-del-paquete-mongodb)
3. [Conexión y Singleton](#-conexión-y-singleton)
4. [Modelos Pydantic](#-modelos-pydantic)
5. [Repositorio Base Genérico](#-repositorio-base-genérico)
6. [Servicios](#-servicios)
7. [LoggingMiddleware (bug corregido)](#-loggingmiddleware)
8. [Colecciones e Índices](#-colecciones-e-índices)
9. [Configuración (env vars)](#-configuración)
10. [Integración con la API](#-integración-con-la-api)
11. [Scripts de Prueba y Mantenimiento](#-scripts-de-prueba-y-mantenimiento)
12. [Dashboard de Usuarios con MongoDB](#-dashboard-de-usuarios-con-mongodb)
13. [Convenciones](#-convenciones)

---

## ✅ Resumen Ejecutivo

| Componente | Estado | Detalle |
|---|---|---|
| Motor | ✅ Activo | `motor` (async) sobre MongoDB Atlas (free tier M0) |
| Conexión | ✅ Singleton | `MongoDBConnection()` comparte un solo cliente |
| Colecciones | ✅ 7 | conversations, metrics, feedback, logs, rag_cache, users, sessions |
| Índices | ✅ Automáticos | Se crean en `connect()` con `asyncio.gather` |
| Persistencia | ✅ Background | `asyncio.create_task` — no bloquea la respuesta HTTP |
| Caché RAG | ✅ Activo | MD5 + TTL 24h + hit_count |
| Logging HTTP | ✅ Activo | Middleware → colección `logs` |
| Retención | ✅ Activo | `retention_policy.py` + endpoint `/admin/cleanup` |
| Degradación | ✅ Graceful | Si MongoDB falla, la respuesta al usuario no se rompe |
| Bug crítico | ✅ Resuelto | Middleware colgaba POST /chat (FastAPI/Starlette < 0.28) |

---

## 🏗️ Arquitectura del Paquete `mongodb/`

```
mongodb/
├── __init__.py                     # Exports públicos (clases + funciones)
├── connection.py                   # MongoDBConnection — singleton async (motor)
├── models.py                       # Modelos Pydantic (7 modelos)
├── middleware/
│   └── logging_middleware.py       # LoggingMiddleware — logs HTTP → colección logs
├── repositories/
│   └── base_repository.py          # BaseRepository[T] — CRUD genérico
└── services/
    ├── conversation_service.py     # Conversaciones + stats + daily + search
    ├── metrics_service.py          # Métricas por endpoint + salud del sistema
    ├── feedback_service.py         # Feedback + stats + reflejo en conversación
    └── cache_service.py            # RAGCacheService — hash MD5 + TTL + hit_count
```

Cada módulo de servicios expone **funciones** (para importar directo) y **clases** de alto nivel (usadas por los endpoints de la API).

---

## 🔌 Conexión y Singleton

**Archivo**: `mongodb/connection.py`

- Patrón **singleton** vía `__new__`: todas las llamadas a `MongoDBConnection()` retornan la misma instancia.
- Usa `AsyncIOMotorClient` con:
  - `maxPoolSize` (default 50)
  - `serverSelectionTimeoutMS` (default 5000ms)
  - `retryWrites=True`
  - `appName="chatbot-rag-cluster"`
- **Índices creados solo una vez por proceso** (flag `_indexes_ready`); si fallan, se loguea warning pero la conexión sigue activa.

```python
db = await MongoDBConnection().connect()   # conecta + ping + crea índices
await MongoDBConnection().ping()           # ping al servidor
db = MongoDBConnection().get_db()          # solo si ya conectado
await MongoDBConnection().disconnect()     # cierra cliente
```

---

## 📦 Modelos Pydantic

**Archivo**: `mongodb/models.py` (Pydantic v2, timestamps por defecto en UTC aware)

| Modelo | Colección | Campos clave |
|---|---|---|
| `ConversationCreate` / `ConversationDocument` | conversations | `conversation_id`, `session_id`, `user_id`, `messages[]`, `sources_used`, `total_tokens`, `latency_ms`, `is_rag_response`, `confidence_score`, `created_at` |
| `MetricCreate` | metrics | `session_id`, `endpoint`, `request_timestamp`, `latency_ms`, `tokens_used`, `is_rag_response`, `confidence_score`, `cache_hit` |
| `FeedbackCreate` | feedback | `session_id`, `conversation_id`, `message_index`, `user_rating` (1-5), `user_comment`, `is_correct`, `created_at` |
| `RAGCacheEntry` | rag_cache | `query_hash`, `query`, `response`, `sources`, `confidence`, `context`, `hit_count`, `created_at`, `expires_at`, `last_accessed` |
| `LogEntry` | logs | `timestamp`, `method`, `path`, `client_ip`, `user_agent`, `status_code`, `response_time_ms`, `body`, `session_id` |
| `MessageRole` / `ConversationMessage` | — | `role` (`user`/`assistant`), `content`, `timestamp`, `tokens` |

`ConversationDocument` hereda de `ConversationCreate` y agrega `id` (interno) y `updated_at`.

---

## 🧩 Repositorio Base Genérico

**Archivo**: `mongodb/repositories/base_repository.py`

```python
class BaseRepository(Generic[T]):   # T = modelo Pydantic
```

| Método | Descripción |
|---|---|
| `create(document)` | Inserta un documento y retorna el modelo con `id` poblado |
| `create_many(documents)` | Inserción masiva |
| `find_one(filter)` | Primer documento que coincida o `None` |
| `find_many(filter, limit, skip, sort)` | Paginado + ordenamiento |
| `update_one(filter, data, upsert)` | Actualiza con `$set`; soporta upsert |
| `delete_one(filter)` | Elimina un documento |
| `count(filter)` | Conteo de documentos |

`_to_model()` convierte el documento MongoDB (`_id` → `id` string) al modelo Pydantic con `model_validate`.

---

## ⚙️ Servicios

### 1. ConversationService (`services/conversation_service.py`)

| Función | Descripción |
|---|---|
| `save_conversation(conv_data)` | **Upsert + acumulación por `conversation_id`**: inserta si no existe; si existe, hace un `update_one` **atómico** que **agrega** los mensajes nuevos al array (`$push $each`) siempre que sus `timestamp` no existan (`messages.timestamp $nin [...]`). Esto evita duplicados por reintento en background y no pierde turnos concurrentes sin carrera de lectura-modificación-escritura. Los resúmenes (`total_tokens`, `latency_ms`, `confidence_score`, `sources_used`) reflejan el último turno |
| `get_conversation(conversation_id)` | Recupera por ID |
| `get_conversations_by_session(session_id, limit)` | Más recientes de una sesión (orden desc) |
| `get_conversations_by_user(user_id, limit)` | Más recientes de un usuario |
| `get_conversation_stats(session_id)` | Pipeline agregación: total conversaciones/mensajes, latencia promedio, tokens, respuestas RAG |
| `get_daily_stats(days)` | Conversaciones agrupadas por día (`$dateToString`) |
| `search_conversations(query, limit)` | Búsqueda regex case-insensitive en `messages[].content` (`$elemMatch`) |

### 2. MetricsService (`services/metrics_service.py`)

| Función | Descripción |
|---|---|
| `record_metric(metric)` | Inserta métrica de rendimiento |
| `get_endpoint_metrics(endpoint, hours)` | Agregación por endpoint: total requests, latencia promedio, tokens, cache hits, respuestas RAG, confianza promedio |
| `get_system_health()` | Ping + conteos (conversations, feedback) + latencia promedio última hora |

### 3. FeedbackService (`services/feedback_service.py`)

| Función | Descripción |
|---|---|
| `record_feedback(feedback)` | Inserta feedback **y además** refleja `user_rating` dentro de `messages[message_index].feedback` de la conversación (best-effort) |
| `get_feedback_stats(days)` | Total, calificación promedio, distribución por rating, correctos/incorrectos |

### 4. RAGCacheService (`services/cache_service.py`)

| Función | Descripción |
|---|---|
| `get_cached_response(query, context)` | Hash MD5; si existe y no expiró → incrementa `hit_count`, actualiza `last_accessed`, retorna entrada. Si expiró → la borra y retorna `None` |
| `cache_response(query, response, sources, confidence, context)` | Guarda con `replace_one(upsert=True)` y `expires_at = now + TTL` |
| `get_cache_stats()` | Total de entradas, hits, promedio |
| `cleanup_expired()` | Borra entradas expiradas |
| `get_expired_entries(limit)` | Lista entradas expiradas (para limpieza) |

Clave del hash: `MD5(json{query.lower().strip(), context})`.

**Normalización de UTC**: MongoDB devuelve datetimes **naive** (UTC implícito); `_ensure_utc()` los convierte a aware para compararlos con `datetime.now(timezone.utc)`.

---

## 🛡️ LoggingMiddleware

**Archivo**: `mongodb/middleware/logging_middleware.py`

Registra cada solicitud HTTP en la colección `logs`: timestamp, method, path, client_ip, user_agent, status_code, `response_time_ms` y body (máx 1000 chars, truncado). Extrae `session_id` del header `X-Session-ID`, query param o body (fallback `"unknown"`).

**🚨 Bug crítico resuelto (agosto 2026)**: Con `FastAPI 0.104.1` / `Starlette 0.27.0` (versiones fijadas en `requirements.txt`), leer `await request.body()` dentro de un `BaseHTTPMiddleware` **antes** de `call_next` consumía el stream y colgaba el endpoint para siempre (POST `/chat` y `/feedback` tardaban 45-60s+ y se caían, mientras los GET respondían normal). Es un bug conocido de Starlette (issues fastapi#394, fastapi#8187, #5386), corregido en FastAPI 0.108+.

**El fix**: después de leer el body, se **reinyecta** en el request para que el endpoint pueda leerlo de nuevo:

```python
async def receive() -> Message:
    return {"type": "http.request", "body": body, "more_body": False}
request._receive = receive
```

**Verificación empírica**:
- Sin fix (Starlette 0.27.0): POST /chat → `TimeoutError` (13.7s)
- Con fix (Starlette 0.27.0): POST /chat → `200 OK` en 0.3s
- Con fix (Starlette 1.3.1 local): POST /chat → `200 OK` en 0.5s

El guardado del log es **best-effort**: si MongoDB falla, solo `logger.debug`, nunca bloquea la respuesta.

---

## 🗄️ Colecciones e Índices

**Base de datos**: `chatbot_rag_db` (configurable con `MONGODB_DB_NAME`)

| Colección | Índices (creados en `connect()`) | Propósito |
|---|---|---|
| `conversations` | `conversation_id` (único), `session_id`, `user_id` | Conversaciones completas |
| `metrics` | `request_timestamp` (desc), `session_id`, `endpoint` | Métricas de rendimiento |
| `feedback` | `conversation_id`, `session_id` | Feedback de usuarios |
| `rag_cache` | `cache_key` (único) | Caché de respuestas RAG |
| `logs` | `timestamp` (desc) | Logs HTTP del middleware |
| `users` | `user_id` (único) | Usuarios (reservado) |
| `sessions` | `session_id` (único) | Sesiones (reservado) |

Nota: `rag_cache` usa `cache_key` como nombre de índice; el campo real del documento es `query_hash`.

---

## 🔧 Configuración

**Archivo**: `config/settings.py` (pydantic-settings, `.env`)

| Variable | Default | Descripción |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | URI de Atlas (ej: `mongodb+srv://...`) |
| `MONGODB_DB_NAME` | `chatbot_rag_db` | Nombre de la base |
| `MONGODB_ENABLE_LOGGING` | `true` | Logs de conexión |
| `MONGODB_RETRY_WRITES` | `true` | Retry de escrituras |
| `MONGODB_MAX_POOL_SIZE` | `50` | Pool del cliente |
| `MONGODB_TIMEOUT_MS` | `5000` | Timeout de selección de servidor |
| `MONGODB_COLL_*` | `conversations`, `metrics`, ... | Nombres de colección |
| `RAG_CACHE_ENABLED` | `true` | Activa/desactiva caché |
| `RAG_CACHE_TTL_HOURS` | `24` | TTL de caché en horas |
| `RETENTION_CONVERSATIONS_DAYS` | `90` | Retención conversaciones |
| `RETENTION_METRICS_DAYS` | `30` | Retención métricas |
| `RETENTION_LOGS_DAYS` | `14` | Retención logs |
| `RETENTION_FEEDBACK_DAYS` | `180` | Retención feedback |
| `ADMIN_API_KEY` | vacío | Protección opcional de `/admin/*` |

---

## 🔗 Integración con la API

### Flujo de persistencia en POST /chat (`api/main.py`)

1. Sanitización de entrada (seguridad) → validación del body.
2. `LangChainRAGWrapper.query_with_memory()` (**async**) genera la respuesta primero.
3. `_load_history()` carga historial previo desde MongoDB (`get_conversations_by_session`); si falla, usa memoria en RAM.
4. `_schedule_save()` persiste **en background** con `asyncio.create_task()` → guarda `ConversationCreate` + `MetricCreate` (tokens estimados `len(text)//4`, `latency_ms` real). **No bloquea la respuesta.**
5. Caché: antes de generar, consulta `RAGCacheService.get_cached_response()`; si hay hit, responde sin llamar al LLM.

### Endpoints que usan MongoDB

| Endpoint | Colecciones | Descripción |
|---|---|---|
| `POST /chat` | conversations, metrics, rag_cache, logs | Flujo principal |
| `POST /feedback` | feedback, conversations | Registra y refleja rating en el mensaje |
| `GET /analytics` | conversations, metrics, feedback | Conversaciones + salud + feedback |
| `POST /admin/cleanup` | todas | Ejecuta retención (protegido por `ADMIN_API_KEY`) |
| `GET /admin/dashboard-report` | todas | Genera reporte JSON |

`mongodb_router` en `api/endpoints.py` (schema limpio `question`/`user_rating`) **no está montado** en `api/main.py` para no chocar con las rutas existentes; es el objetivo de prueba de `scripts/test_integration.py`.

---

## 🗒️ Dashboard de Usuarios con MongoDB

**Archivo**: `evaluation/generate_user_dashboard.py` · `/user-dashboard` · `/user-dashboard/refresh`

El dashboard de interacciones reales usa **MongoDB como fuente principal** de datos, con fallback a los archivos JSONL locales del Space:

| Dato | Fuente principal (MongoDB) | Fallback (archivos) |
|---|---|---|
| Interacciones (pregunta, respuesta, tiempo, confianza, fuentes, tokens) | Colección `conversations` (`get_recent_conversations`) | `user_interactions.jsonl` |
| Tokens del día + promedio | Colección `metrics` (agregación `tokens_used` desde medianoche UTC) | `token_usage.json` / `token_usage_per_query.jsonl` |
| Tokens por hora (24h) | Colección `metrics` (`$hour` en `request_timestamp`) | `token_usage_per_query.jsonl` |
| Pestaña Logs del Sistema | Colección `logs` (middleware) vía `/api/logs` | `data/system_logs.jsonl` |

### Flujo de generación (`generate_user_dashboard_async`)

1. `fetch_mongodb_interactions(limit)` → lee las conversaciones más recientes y las mapea al formato de interacción (parea cada mensaje `user` con el siguiente `assistant`).
2. `fetch_mongodb_tokens_por_hora()` → agregación por hora en `metrics`.
3. `fetch_mongodb_token_stats()` → tokens acumulados desde medianoche UTC.
4. Si MongoDB falla o no tiene datos → **degradación graceful**: se usan los JSONL (solo `logger.debug`, sin romper el dashboard).
5. El HTML se escribe en `/data/user_dashboard.html`.

> 💡 **Interacciones completas**: como `save_conversation` **acumula** mensajes en el mismo `conversation_id` (p. ej. `web_interface`), una sola conversación contiene TODOS los turnos de la sesión y el dashboard los lista todos. Cada mensaje `assistant` guarda sus propias métricas por turno (`latency_ms`, `confidence_score`, `is_rag`, `sources_used`); el dashboard prefiere esas por interacción y cae a los resúmenes de la conversación si no existen (mensajes antiguos).

### Async-aware

- `generate_user_dashboard_async()` es `async` y la usan los endpoints (`api/main.py` arranque y `/user-dashboard/refresh`) con `await` — evita el `RuntimeError` de `asyncio.run()` dentro de un loop activo.
- `generate_user_dashboard()` (síncrono, wrapper) usa `asyncio.run()` y queda para CLI / compatibilidad.

### Configuración

| Variable | Default | Descripción |
|---|---|---|
| `DASHBOARD_USE_MONGODB` | `true` | Usa MongoDB con fallback a JSONL |
| `DASHBOARD_MONGODB_INTERACTIONS_LIMIT` | `2000` | Conversaciones máx. leídas |

### Verificación

```bash
python -m scripts.test_dashboard_mongodb   # métricas + fallback JSONL + mapeo MongoDB
```

---

## 🧪 Scripts de Prueba y Mantenimiento

```bash
python -m scripts.test_mongodb_connection   # Conexión a Atlas (ping, índices)
python -m scripts.test_services             # Repositorios y servicios CRUD
python -m scripts.test_integration          # API completa con StubRAG (httpx ASGITransport)
python -m scripts.test_cache                # RAGCacheService (hash, TTL, hits)
python scripts/retention_policy.py --dry-run # Política de retención (dry-run)
python scripts/retention_policy.py           # Ejecuta limpieza
python scripts/generate_dashboard_report.py  # Reporte → data/dashboard_report.json
```

- `test_integration.py` usa `StubRAG` (evita FAISS/Groq), monta `mongodb_router` en un FastAPI temporal y limpia sus datos al final (`itest-session-*`).
- ⚠️ En Windows ejecutar con `PYTHONIOENCODING=utf-8` para evitar `UnicodeEncodeError` con los emojis de los logs.

---

## 📌 Convenciones

- **SIEMPRE** usar el singleton `MongoDBConnection()`.
- **NUNCA** bloquear la respuesta con el guardado → usar `_schedule_save()` (background).
- **SIEMPRE** degradar con graceful si MongoDB falla (la respuesta al usuario no debe romperse).
- **SIEMPRE** normalizar datetimes a UTC aware al leer de MongoDB (`_ensure_utc`).
- Los servicios exponen clases y funciones; los endpoints usan las **clases**.
- Guardado de caché con `replace_one(upsert=True)` para evitar duplicados por `query_hash`.

---

📎 **Referencias**: [AGENTS.md](./AGENTS.md) · [MONGODB_GUIDE.md](./MONGODB_GUIDE.md)
