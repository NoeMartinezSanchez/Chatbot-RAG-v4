#!/bin/bash

set -e

echo "🚀 Iniciando Prepa ChatBot con Ollama..."

# Iniciar Ollama en segundo plano
echo "📥 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Esperar a que Ollama esté listo
echo "⏳ Esperando a Ollama..."
sleep 5

# Verificar que Ollama esté funcionando
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama está listo"
        break
    fi
    echo "⏳ Esperando... ($i/30)"
    sleep 2
done

# Descargar modelo gemma4:e4b si no existe
echo "📦 Verificando modelo gemma4:e4b..."
if ollama list | grep -q "gemma4:e4b"; then
    echo "✅ Modelo gemma4:e4b ya está instalado"
else
    echo "📥 Descargando modelo gemma4:e4b (puede tomar varios minutos)..."
    ollama pull gemma4:e4b
    echo "✅ Modelo descargado"
fi

# Iniciar la aplicación FastAPI
echo "🚀 Starting FastAPI application..."
cd /app
exec uvicorn api.main:app --host 0.0.0.0 --port 7860