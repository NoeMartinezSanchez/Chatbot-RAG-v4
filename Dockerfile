FROM python:3.11-slim

# Metadata para HuggingFace Spaces
LABEL io.spaceflake.name="prepa-chatbot"
LABEL io.spaceflake.title="Prepa en Línea ChatBot"
LABEL io.spaceflake.description="Chatbot educativo para Prepa en Línea SEP con Gemini API"
LABEL io.spaceflake.license="mit"
LABEL io.spaceflake.author="Tu Nombre"
LABEL io.spaceflake.tags="education, chatbot, rag, gemini"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/vector_store data/documents logs

# ==== Variables de entorno ====
# MongoDB
ENV MONGODB_URI=${MONGODB_URI}
ENV MONGODB_DB_NAME=${MONGODB_DB_NAME:-chatbot_rag_db}
ENV MONGODB_ENABLE_LOGGING=${MONGODB_ENABLE_LOGGING:-true}
ENV MONGODB_RETRY_WRITES=${MONGODB_RETRY_WRITES:-true}
ENV MONGODB_MAX_POOL_SIZE=${MONGODB_MAX_POOL_SIZE:-50}
ENV MONGODB_TIMEOUT_MS=${MONGODB_TIMEOUT_MS:-5000}

# Cache
ENV RAG_CACHE_TTL_HOURS=${RAG_CACHE_TTL_HOURS:-24}
ENV RAG_CACHE_ENABLED=${RAG_CACHE_ENABLED:-true}

# Retención (días)
ENV RETENTION_CONVERSATIONS_DAYS=${RETENTION_CONVERSATIONS_DAYS:-90}
ENV RETENTION_METRICS_DAYS=${RETENTION_METRICS_DAYS:-30}
ENV RETENTION_LOGS_DAYS=${RETENTION_LOGS_DAYS:-14}
ENV RETENTION_FEEDBACK_DAYS=${RETENTION_FEEDBACK_DAYS:-180}

# API / Admin
ENV ADMIN_API_KEY=${ADMIN_API_KEY}
ENV TIMEZONE=${TIMEZONE:-America/Mexico_City}
ENV LOG_LEVEL=${LOG_LEVEL:-INFO}
ENV ENVIRONMENT=${ENVIRONMENT:-production}

# Modelos
ENV GROQ_API_KEY=${GROQ_API_KEY}

# Monitoreo (opcional)
ENV TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
ENV TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]