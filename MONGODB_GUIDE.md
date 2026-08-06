# MONGODB_GUIDE.md — Guía de MongoDB para el equipo

Guía completa para entender, consultar y mantener la base de datos MongoDB del Chatbot RAG de Prepa en Línea SEP.

---

## 📋 Tabla de Contenidos

1. [¿Qué es MongoDB y por qué lo usamos?](#-qué-es-mongodb-y-por-qué-lo-usamos)
2. [Estructura de la base de datos](#-estructura-de-la-base-de-datos)
3. [Cómo consultar datos desde Python](#-cómo-consultar-datos-desde-python)
4. [Ejemplos de consultas comunes](#-ejemplos-de-consultas-comunes)
5. [Cómo interpretar el dashboard](#-cómo-interpretar-el-dashboard)
6. [Mantenimiento y resolución de problemas](#-mantenimiento-y-resolución-de-problemas)

---

## 🧠 ¿Qué es MongoDB y por qué lo usamos?

**MongoDB** es una base de datos **NoSQL orientada a documentos**. En lugar de tablas con filas y columnas (como MySQL/PostgreSQL), guarda datos en **colecciones** de **documentos JSON (BSON)**.

| Característica | Detalle |
|---|---|
| **Tipo** | NoSQL, documental |
| **Formato** | Documentos JSON/BSON |
| **Escalabilidad** | Horizontal (sharding) y vertical |
| **Esquema** | Flexible: cada documento puede tener campos distintos |
| **Alojamiento** | Atlas (nube, free tier) |
| **Motor** | `motor` (async) en Python |

### ¿Por qué lo usamos en este proyecto?

1. **Historial conversacional**: el chatbot necesita guardar cada conversación con sus mensajes, fuentes y métricas. Un documento por conversación encaja perfecto.
2. **Esquema flexible**: los campos de una conversación pueden crecer sin migraciones (agregar feedback por mensaje, por ejemplo).
3. **Velocidad de desarrollo**: los datos se guardan como objetos JSON, igual que las respuestas de la API. Sin mapeo objeto-relacional (ORM).
4. **Async nativo**: el driver `motor` es asíncrono y encaja con FastAPI/uvicorn.
5. **Costo $0**: el cluster gratuito (M0) de Atlas es suficiente para el volumen actual.
6. **Operaciones atómicas**: `update_one`, `replace_one(upsert=True)`, pipelines de agregación — ideales para métricas y caché.

### Alternativas consideradas (y por qué no)

| Opción | Por qué no |
|---|---|
| SQLite/MySQL | Relacional, requiere migraciones de esquema, sin sharding nativo |
| Redis | En memoria, no es persistente por defecto, costo extra |
| Firestore | Vendor lock-in, límites gratuitos más restrictivos |

---

## 🗄️ Estructura de la base de datos

**Base de datos**: `chatbot_rag_db` (configurable con `MONGODB_DB_NAME`)

```
chatbot_rag_db
├── conversations      → Conversaciones completas con mensajes y feedback
├── metrics            → Métricas de rendimiento por solicitud HTTP
├── feedback           → Feedback individual de usuarios
├── logs               → Registro de solicitudes HTTP (middleware)
├── rag_cache          → Caché de respuestas RAG (hash MD5 + TTL)
├── users              → Usuarios (reservado)
└── sessions           → Sesiones (reservado)
```

### Diagrama de relaciones

```
USUARIO (users)
   │
   │ 1:N
   ▼
SESIONES (sessions)
   │
   │ 1:N
   ▼
CONVERSACIONES (conversations) ──── 1:N ──── FEEDBACK (feedback)
   │
   │ (cada solicitud genera)
   ▼
MÉTRICAS (metrics) ──── registra latencia/tokens/cache_hit

CACHÉ (rag_cache) ──── respuestas reutilizables por hash

LOGS (logs) ──── auditoría de solicitudes HTTP
```

### Índices creados automáticamente

| Colección | Índices |
|---|---|
| `conversations` | `conversation_id` (único), `session_id`, `user_id` |
| `metrics` | `request_timestamp` (desc), `session_id`, `endpoint` |
| `feedback` | `conversation_id`, `session_id` |
| `users` | `user_id` (único) |
| `sessions` | `session_id` (único) |
| `rag_cache` | `cache_key` (único) |
| `logs` | `timestamp` (desc) |

Los índices se crean **una sola vez por proceso** (`_indexes_ready` en `mongodb/connection.py`) al conectar.

---

## 📄 Estructura de los documentos

### `conversations`

```json
{
  "conversation_id": "a1b2c3d4-...",
  "session_id": "default",
  "user_id": "user_123",
  "messages": [
    {
      "role": "user",
      "content": "¿Qué es el módulo propedéutico?",
      "timestamp": "2026-08-04T21:00:00Z",
      "tokens": 12,
      "feedback": { "user_rating": 5, "timestamp": "2026-08-04T21:00:05Z" }
    },
    {
      "role": "assistant",
      "content": "El módulo propedéutico es obligatorio...",
      "timestamp": "2026-08-04T21:00:02Z",
      "tokens": 40
    }
  ],
  "sources_used": [{"metadata": {"source_file": "Control_Escolar.xlsx"}}],
  "total_tokens": 52,
  "latency_ms": 2048.5,
  "is_rag_response": true,
  "confidence_score": 0.95,
  "created_at": "2026-08-04T21:00:02Z",
  "updated_at": "2026-08-04T21:00:05Z"
}
```

### `metrics`

```json
{
  "session_id": "default",
  "endpoint": "/chat",
  "request_timestamp": "2026-08-04T21:00:02Z",
  "latency_ms": 2048.5,
  "tokens_used": 52,
  "is_rag_response": true,
  "confidence_score": 0.95,
  "cache_hit": false
}
```

### `feedback`

```json
{
  "session_id": "default",
  "conversation_id": "a1b2c3d4-...",
  "message_index": 0,
  "user_rating": 5,
  "user_comment": "Excelente respuesta",
  "is_correct": true,
  "created_at": "2026-08-04T21:00:05Z"
}
```

### `rag_cache`

```json
{
  "query_hash": "e72802cf6417a038651407886eeeaa8d",
  "query": "¿Qué es el módulo propedéutico?",
  "response": "El módulo propedéutico es obligatorio...",
  "sources": [],
  "confidence": 0.95,
  "context": null,
  "hit_count": 4,
  "created_at": "2026-08-04T20:00:00Z",
  "expires_at": "2026-08-05T20:00:00Z",
  "last_accessed": "2026-08-04T21:29:53Z"
}
```

### `logs`

```json
{
  "timestamp": "2026-08-04T21:00:02Z",
  "method": "POST",
  "path": "/chat",
  "client_ip": "10.0.0.1",
  "user_agent": "Mozilla/5.0 ...",
  "status_code": 200,
  "response_time_ms": 2100.4,
  "body": "{\"question\": \"hola\", \"session_id\": \"default\"}",
  "session_id": "default"
}
```

---

## 🐍 Cómo consultar datos desde Python

### 1. Conectarse (siempre usar el singleton)

```python
import asyncio
from mongodb.connection import MongoDBConnection
from config.settings import settings

async def main():
    connection = MongoDBConnection()   # Singleton
    db = await connection.connect()    # Crea índices la primera vez
    print(f"Conectado a: {settings.MONGODB_DB_NAME}")

    # Consultas directas
    coleccion = db["conversations"]
    conversaciones = await coleccion.count_documents({})
    print(f"Conversaciones totales: {conversaciones}")

    await connection.disconnect()      # Al terminar

asyncio.run(main())
```

> ⚠️ **IMPORTANTE**: MongoDB devuelve datetimes **naive** (UTC implícito). Antes de comparar con `datetime.now(timezone.utc)`, normaliza con `.replace(tzinfo=timezone.utc)`.

### 2. Usar los servicios (recomendado)

Los servicios encapsulan las consultas más comunes y ya gestionan la conexión.

```python
from mongodb.services import ConversationService, MetricsService, FeedbackService, RAGCacheService

conv_service = ConversationService()
metrics_service = MetricsService()
feedback_service = FeedbackService()
cache_service = RAGCacheService()

# Ejemplos
convs = await conv_service.get_conversations_by_session("default", limit=10)
stats = await conv_service.get_conversation_stats("default")
daily = await conv_service.get_daily_stats(days=7)
health = await metrics_service.get_system_health()
ep_metrics = await metrics_service.get_endpoint_metrics("/chat", hours=24)
fb_stats = await feedback_service.get_feedback_stats(days=30)
cache_stats = await cache_service.get_stats()
```

### 3. Usar el repositorio genérico (CRUD bajo nivel)

```python
from mongodb.connection import MongoDBConnection
from mongodb.models import ConversationDocument
from mongodb.repositories.base_repository import BaseRepository

async def main():
    db = await MongoDBConnection().connect()
    repo = BaseRepository(db["conversations"], ConversationDocument)

    conv = await repo.find_one({"conversation_id": "abc"})
    todos = await repo.find_many({"session_id": "default"}, limit=20, sort=[("created_at", -1)])
    total = await repo.count({"session_id": "default"})

    # Guardar (inserta o actualiza)
    from mongodb.models import ConversationCreate
    await ConversationService().save_conversation(ConversationCreate(
        conversation_id="abc", session_id="default", messages=[]
    ))
```

### 4. Insertar una métrica manualmente

```python
from mongodb.models import MetricCreate
from mongodb.services import MetricsService

await MetricsService().record_metric(MetricCreate(
    endpoint="/chat",
    latency_ms=1800.5,
    tokens_used=52,
    is_rag_response=True,
    confidence_score=0.95,
    cache_hit=False,
))
```

### 5. Registro rápido (script de prueba)

```bash
python -m scripts.test_mongodb_connection   # Verificar conexión
python -m scripts.test_services             # Probar repositorios y servicios
python -m scripts.test_cache                # Probar caché
python -m scripts.test_integration          # Probar API + MongoDB
```

---

## 🔍 Ejemplos de consultas comunes

### Total de conversaciones por sesión

```python
from mongodb.services import ConversationService
stats = await ConversationService().get_conversation_stats("default")
# {'total_conversations': 5, 'total_messages': 12, 'avg_latency_ms': 2048.5,
#  'total_tokens': 420, 'rag_responses': 4}
```

### Conversaciones por día (últimos 7 días)

```python
from mongodb.services import ConversationService
daily = await ConversationService().get_daily_stats(days=7)
# [{'date': '2026-08-01', 'count': 3}, {'date': '2026-08-02', 'count': 5}, ...]
```

### Salud del sistema

```python
from mongodb.services import MetricsService
health = await MetricsService().get_system_health()
# {'status': 'healthy', 'total_conversations': 12, 'metrics_last_hour': 8,
#  'total_feedback': 3, 'avg_latency_ms_last_hour': 1890.2}
```

### Estadísticas de feedback

```python
from mongodb.services import FeedbackService
fb = await FeedbackService().get_feedback_stats(days=30)
# {'days': 30, 'total_feedback': 15, 'avg_rating': 4.3,
#  'correct_count': 12, 'incorrect_count': 3,
#  'rating_distribution': {'5': 8, '4': 4, '3': 1, '2': 1, '1': 1}}
```

### Buscar conversaciones por contenido

```python
from mongodb.services import ConversationService
matches = await ConversationService().search_conversations("certificado", limit=5)
```

### Obtener y verificar un caché

```python
from mongodb.services import RAGCacheService

cache = RAGCacheService()
entrada = await cache.get_cached_response("¿qué es el módulo propedéutico?")
if entrada:
    print("CACHE HIT", entrada.response, "hits:", entrada.hit_count)
else:
    print("CACHE MISS")
```

### Consulta SQL-equivalente con agregación (ej. promedio de latencia por endpoint)

```python
from mongodb.connection import MongoDBConnection

db = await MongoDBConnection().connect()
pipeline = [
    {"$group": {"_id": "$endpoint", "avg_latency": {"$avg": "$latency_ms"}, "total": {"$sum": 1}}},
    {"$sort": {"total": -1}},
]
rows = await db["metrics"].aggregate(pipeline).to_list(length=None)
```

### Borrar datos de una sesión de prueba

```python
from mongodb.connection import MongoDBConnection

db = await MongoDBConnection().connect()
await db["conversations"].delete_many({"session_id": "itest-session-abc"})
```

---

## 📊 Cómo interpretar el dashboard

El dashboard se compone de un **reporte JSON** (`data/dashboard_report.json`) y el **dashboard HTML** servido en `/dashboard`.

### Generar el reporte

```bash
python scripts/generate_dashboard_report.py [--days 7]
# o vía API:
# GET /admin/dashboard-report?days=7
```

### Campos del reporte (`dashboard_report.json`)

| Campo | Qué significa | Cómo interpretarlo |
|---|---|---|
| `generated_at` | Fecha de generación | — |
| `days_window` | Días analizados | — |
| `daily_stats` | Conversaciones por día | Tendencia: ¿crece o decrece el uso? |
| `system_health.status` | `healthy` o no | `healthy` = sistema operativo |
| `system_health.total_conversations` | Conversaciones totales en BD | Volumen acumulado |
| `system_health.metrics_last_hour` | Solicitudes en la última hora | Tráfico reciente |
| `system_health.avg_latency_ms_last_hour` | Latencia promedio última hora | **< 3000 ms = óptimo**; > 3000 ms = revisar |
| `feedback_stats.total_feedback` | Feedback recibido | Participación de usuarios |
| `feedback_stats.avg_rating` | Calificación promedio (1-5) | **≥ 4.0 = bueno**; 3.0–3.9 = aceptable; < 3.0 = revisar respuestas |
| `feedback_stats.rating_distribution` | Conteo por calificación | Ver distribución de satisfacción |
| `cache_stats.total_entries` | Entradas en caché | Cobertura de respuestas frecuentes |
| `cache_stats.total_hits` | Veces que se reutilizó una respuesta | **Hits altos = ahorro de tokens** |
| `executive_summary.message` | Resumen automático | Lectura rápida de estado |

### Semáforo de interpretación

| Métrica | 🟢 Bueno | 🟡 Atención | 🔴 Crítico |
|---|---|---|---|
| Latencia promedio | < 3 s | 3–5 s | > 5 s |
| Calificación feedback | ≥ 4.0 | 3.0–3.9 | < 3.0 |
| Estado del sistema | healthy | — | otro valor |
| Cache hits | creciendo | estable | cayendo |

> 💡 El `executive_summary` genera un mensaje en lenguaje natural con estos umbrales, para una lectura rápida sin abrir tablas.

---

## 🔧 Mantenimiento y resolución de problemas

### Política de retención (limpieza de datos viejos)

Elimina automáticamente datos antiguos para controlar el tamaño de la base:

| Colección | Días conservados | Variable |
|---|---|---|
| `conversations` | 90 | `RETENTION_CONVERSATIONS_DAYS` |
| `metrics` | 30 | `RETENTION_METRICS_DAYS` |
| `logs` | 14 | `RETENTION_LOGS_DAYS` |
| `feedback` | 180 | `RETENTION_FEEDBACK_DAYS` |

```bash
# Primero en modo seguro (solo cuenta, no elimina):
python scripts/retention_policy.py --dry-run

# Ejecución real:
python scripts/retention_policy.py

# O vía API:
# POST /admin/cleanup?dry_run=true
# POST /admin/cleanup
```

**Salida esperada** (dry-run):
```
=== RESUMEN DE RETENCIÓN ===
Modo: DRY RUN (nada se eliminó)
  Conversaciones   Eliminaría     0 docs  (retención: 90 días)
  Métricas         Eliminaría     0 docs  (retención: 30 días)
  Logs             Eliminaría     0 docs  (retención: 14 días)
  Feedback         Eliminaría     0 docs  (retención: 180 días)
Total: 0 documentos
```

### Limpieza del caché

```python
from mongodb.services import RAGCacheService
eliminadas = await RAGCacheService().cleanup_expired()
print(f"Entradas expiradas eliminadas: {eliminadas}")
```

### Problemas comunes

| Problema | Síntoma | Solución |
|---|---|---|
| **Datetime naive vs aware** | `TypeError: can't compare offset-naive and offset-aware datetimes` | Normalizar con `.replace(tzinfo=timezone.utc)` al leer de MongoDB |
| **No se puede conectar** | Timeout o `ServerSelectionTimeoutError` | Verificar `MONGODB_URI`, IP en Atlas Allow List, driver `dnspython` instalado |
| **Índices no creados** | Consultas lentas | Revisar log `⚠️ No se pudieron crear índices`; verificar permisos del usuario en Atlas |
| **Guardado en background no aparece** | La conversación no se ve en MongoDB | Los guardados son **asíncronos** (`asyncio.create_task`). Esperar 1–2 s; revisar `logger.debug` del wrapper |
| **Caché no responde** | Siempre `CACHE MISS` | Verificar `RAG_CACHE_ENABLED=true` y que el TTL no haya expirado |
| **Logs vacíos** | La colección `logs` no crece | El middleware falla silenciosamente (`logger.debug`); verificar conexión MongoDB al arrancar |
| **Base de datos crece rápido** | Costo/espacio en Atlas | Ejecutar `retention_policy.py`; ajustar días vía env vars |
| **Errores de duplicados** | `duplicate key error` en `conversations` | La API debería usar el mismo `conversation_id` para actualizar (upsert). Verificar que se genera un UUID nuevo por turno |

### Comandos útiles (shell de Atlas / Compass)

```javascript
// Conteos por colección
db.conversations.countDocuments({})
db.metrics.countDocuments({})
db.feedback.countDocuments({})
db.logs.countDocuments({})
db.rag_cache.countDocuments({})

// Últimas 10 conversaciones
db.conversations.find({}).sort({created_at: -1}).limit(10)

// Latencia promedio del /chat (última hora)
db.metrics.aggregate([
  { $match: { endpoint: "/chat", request_timestamp: { $gte: new Date(Date.now() - 3600_000) } } },
  { $group: { _id: null, avg: { $avg: "$latency_ms" }, count: { $sum: 1 } } }
])

// Entradas de caché más usadas
db.rag_cache.find({}).sort({hit_count: -1}).limit(10)

// Borrar datos de una sesión de prueba
db.conversations.deleteMany({ session_id: /^itest-/ })
```

### Buenas prácticas

- ✅ **SIEMPRE** usar `MongoDBConnection()` (singleton) — nunca crear clientes sueltos.
- ✅ **NUNCA** bloquear la respuesta del usuario con el guardado (usar `_schedule_save()`).
- ✅ Degradar con *graceful* si MongoDB falla: la respuesta al usuario nunca debe romperse.
- ✅ Usar las **clases de servicio** (`ConversationService`, etc.) en los endpoints, no consultas sueltas.
- ✅ Ejecutar `retention_policy.py --dry-run` antes de cada limpieza en producción.
- ✅ Al leer datetimes de MongoDB, normalizar a UTC aware.
- ✅ Revisar `data/dashboard_report.json` generado como artefacto esperado del repo.
- ⚠️ Las colecciones `users` y `sessions` están **reservadas** (aún sin uso).
- ⚠️ `MONGODB_URI` contiene credenciales reales — **nunca** commitearla en repos públicos; usar variables de entorno.

---

Última actualización: 4 de Agosto de 2026
