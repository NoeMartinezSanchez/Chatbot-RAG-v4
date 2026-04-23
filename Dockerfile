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

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]