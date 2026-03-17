# Usar imagen oficial de Python 3.11
FROM python:3.11-slim

# Metadata para HuggingFace Spaces
LABEL io.spaceflake.name="prepa-chatbot"
LABEL io.spaceflake.title="Prepa en Línea ChatBot"
LABEL io.spaceflake.description="Chatbot educativo para Prepa en Línea SEP con TinyLlama"
LABEL io.spaceflake.license="mit"
LABEL io.spaceflake.author="Tu Nombre"
LABEL io.spaceflake.tags="education, chatbot, rag, llm"

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/vector_store data/documents logs

# Exponer puerto (Render asignará automáticamente)
EXPOSE 8000

# Comando de inicio para HuggingFace Spaces
CMD uvicorn api.main:app --host 0.0.0.0 --port 7860