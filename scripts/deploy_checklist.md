# 📋 Checklist Despliegue HF Spaces

Lista de verificación para desplegar el Chatbot RAG en Hugging Face Spaces.
Completar en orden: **variables → verificación → despliegue → validación final**.

---

## 1. Variables de Entorno

Configurar en el Space: **Settings → Variables and secrets** (NO usar archivo `.env`).

| Variable | ¿Obligatoria? | Estado |
|----------|---------------|--------|
| `GROQ_API_KEY` | ✅ Sí | ☐ |
| `MONGODB_URI` | ✅ Sí | ☐ |
| `MONGODB_DB_NAME` | ✅ Sí | ☐ |
| `TIMEZONE` | ✅ Sí | ☐ |
| `ADMIN_API_KEY` | ⭐ Recomendada | ☐ |
| `RAG_CACHE_*` | ⭐ Recomendada | ☐ |
| `RETENTION_*_DAYS` | ⭐ Recomendada | ☐ |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | ➖ Opcional | ☐ |

> 📌 Referencia: `.env.example` en la raíz del repo contiene todas las variables con sus defaults.

---

## 2. Verificación

Ejecutar **antes del push** para validar configuración y conexiones.

```bash
# Verificar variables de entorno del despliegue
python scripts/verify_hf_spaces.py

# Verificar conexión a MongoDB Atlas
python scripts/test_mongodb_connection.py

# Verificar servicios MongoDB (repositorios y servicios)
python -m scripts.test_services

# Verificar caché de respuestas RAG
python -m scripts.test_cache

# Verificar integración completa (API + MongoDB)
python -m scripts.test_integration
```

| Verificación | Estado |
|--------------|--------|
| ☐ `python scripts/verify_hf_spaces.py` pasa (exit 0) |
| ☐ `python scripts/test_mongodb_connection.py` pasa |
| ☐ `python -m scripts.test_integration` pasa |
| ☐ Health check en `/health` retorna `{"status": "healthy"}` |
| ☐ Endpoint `/chat` responde con HTTP 200 |
| ☐ `/admin/cleanup?dry_run=true` responde (protegido por `ADMIN_API_KEY`) |

---

## 3. Despliegue

1. ☐ Commitear y hacer **push a GitHub** (rama `main`).
2. ☐ Confirmar que **GitHub Actions** ejecuta el pipeline *CI/CD - Test & Deploy*.
3. ☐ Esperar a que el job `test` pase (≈1 min 38 seg).
4. ☐ Confirmar que el job `deploy` se ejecutó (solo si `test` pasó).
5. ☐ Verificar en **HF Spaces** que el build terminó **exitoso**.
6. ☐ Confirmar que la app inició **sin errores** en los logs del Space.

---

## 4. Validación Final (post-despliegue)

```bash
# Health check
curl https://TU_SPACE.hf.space/health

# Consulta de prueba
curl -X POST "https://TU_SPACE.hf.space/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿El módulo propedéutico es obligatorio?"}'
```

| Validación | Estado |
|------------|--------|
| ☐ `/health` retorna HTTP 200 |
| ☐ `/chat` retorna respuesta RAG con `is_rag_response: true` |
| ☐ La conversación aparece en MongoDB Atlas (colección `conversations`) |
| ☐ La métrica aparece en `metrics` (endpoint `/chat`) |
| ☐ El monitoreo de GitHub Actions (`monitor.yml`) no envía alertas |

---

## Problemas Comunes

| Problema | Solución |
|----------|----------|
| Build falla al instalar dependencias | Verificar `requirements.txt`; espacio con sdk `docker` |
| `MONGODB_URI` no conecta | Revisar Allow List de IPs en Atlas (incluir IPs de HF) |
| Variable no aplica | Revisar nombre exacto; HF Spaces las inyecta como secrets, reiniciar el Space |
| `/admin/*` devuelve 401 | Configurar `ADMIN_API_KEY` y enviar header `X-Admin-Key` |
