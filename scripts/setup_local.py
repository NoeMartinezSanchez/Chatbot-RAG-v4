#!/usr/bin/env python3
"""
Script para configurar el entorno local
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Configurar entorno local"""
    
    print("🚀 Configurando entorno local para Asistente Educativo RAG")
    
    # 1. Crear estructura de directorios
    directories = [
        "data/documents",
        "data/vector_store",
        "logs",
        "config",
        "api",
        "rag",
        "tests",
        "scripts",
        "docker"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 Directorio creado: {directory}")
    
    # 2. Crear archivo .env
    env_content = """# Configuración del entorno
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000

# Modelos
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es

# RAG
CHUNK_SIZE=768
CHUNK_OVERLAP=128
TOP_K_RESULTS=3
SIMILARITY_THRESHOLD=0.7

# Vector Store
VECTOR_STORE=chroma
PERSIST_DIRECTORY=./data/vector_store
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("  ⚙️  Archivo .env creado")
    
    # 3. Crear intents.json básico si no existe
    if not os.path.exists("data/intents.json"):
        basic_intents = {
            "intents": [
                {
                    "tag": "saludo",
                    "patterns": [
                        "hola", "buenos días", "buenas tardes", "hola asistente"
                    ],
                    "responses": [
                        "¡Hola! Soy tu asistente del módulo propedéutico. ¿En qué puedo ayudarte?"
                    ],
                    "context": "welcome"
                },
                {
                    "tag": "despedida",
                    "patterns": [
                        "adiós", "gracias", "hasta luego", "chao"
                    ],
                    "responses": [
                        "¡Hasta luego! Recuerda que estoy aquí para ayudarte con el módulo.",
                        "¡Nos vemos! Si tienes más dudas, no dudes en preguntar."
                    ],
                    "context": "goodbye"
                }
            ]
        }
        
        import json
        with open("data/intents.json", "w", encoding="utf-8") as f:
            json.dump(basic_intents, f, ensure_ascii=False, indent=2)
        print("  💬 Archivo intents.json creado con ejemplos básicos")
    
    # 4. Crear requirements.txt
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sentence-transformers==2.2.2
chromadb==0.4.18
transformers==4.36.0
torch==2.1.0
numpy==1.24.3
python-multipart==0.0.6
pytest==7.4.3
python-dotenv==1.0.0
requests==2.31.0
langchain==0.0.339
tiktoken==0.5.2
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("  📦 requirements.txt creado")
    
    print("\n✅ Entorno configurado correctamente!")
    print("\n📋 Próximos pasos:")
    print("1. Instalar dependencias: pip install -r requirements.txt")
    print("2. Ejecutar tests: python -m pytest tests/")
    print("3. Iniciar API: python -m api.main")
    print("4. Acceder a: http://localhost:8000/docs")

if __name__ == "__main__":
    setup_environment()