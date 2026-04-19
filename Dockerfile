# Usar imagen oficial de Python 3.11
FROM python:3.11-slim

# Metadata para HuggingFace Spaces
LABEL io.spaceflake.name="prepa-chatbot"
LABEL io.spaceflake.title="Prepa en Línea ChatBot"
LABEL io.spaceflake.description="Chatbot educativo para Prepa en Línea SEP con Ollama (Gemma 4)"
LABEL io.spaceflake.license="mit"
LABEL io.spaceflake.author="Tu Nombre"
LABEL io.spaceflake.tags="education, chatbot, rag, llm, ollama"

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copiar requirements primero (mejor caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/vector_store data/documents logs

# Exponer puertos (FastAPI y Ollama)
EXPOSE 7860 11434

# Hacer start.sh ejecutable
RUN chmod +x start.sh

# Comando de inicio
CMD ["./start.sh"]