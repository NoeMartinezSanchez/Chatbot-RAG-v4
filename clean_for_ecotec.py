#!/usr/bin/env python3
"""
clean_for_ecotec.py - Script ÚNICO de limpieza
Crea el proyecto ecotec-chatbot desde cero con todo limpio

Uso: python clean_for_ecotec.py
"""

import os
import shutil
import sys
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_NAME = "ecotec-chatbot"
SOURCE_DIR = Path(".")  # Directorio actual (Chatbot-RAG-Fuente-Base)
TARGET_DIR = Path(f"../{PROJECT_NAME}")  # Proyecto nuevo

# ============================================================================
# ARCHIVOS Y CARPETAS A ELIMINAR
# ============================================================================

DELETE_PATTERNS = [
    # Vector stores y FAISS
    "data/vector_store/",
    "data/faiss_index",
    "data/*.faiss",
    
    # Placeholders
    "data/mapeo_urls_global.json",
    "data/mapeo_fechas_completo.json",
    "data/mapeo_fechas_categorizadas.json",
    
    # Modelos no usados
    "models/gemini_wrapper.py",
    "models/ollama_wrapper.py",
    "models/gemma_wrapper.py",
    "models/tinyllama_wrapper.py",
    "models/cache/",
    
    # Scripts específicos de SEP
    "scripts/extract_dates.py",
    "scripts/load_chunks_to_rag.py",
    "scripts/update_placeholders.py",
    "scripts/setup_local.py",
    "scripts/upload_documents.py",
    
    # Evaluación (SEP)
    "evaluation/",
    
    # Tests específicos
    "tests/test_rag.py",
    
    # Archivos de datos SEP
    "data/Navegación Jerárquica_FER.xlsx",
    "data/create_prepa_excel.py",
    "data/backups/",
    
    # Archivos sueltos innecesarios
    "Navegación Jerárquica_FER.xlsx",
    "reset_faiss.py",
    "test_groq.py",
    "verificar_memoria.py",
    "notas.txt",
    "generate_tree.py",
    "estructura_proyecto.txt",
    "packages.txt",
    "skills-lock.json",
    ".cursorrules",
    "GENERAR_MENU.md",
    "clean_for_ecotec.py",  # No copiar el script al nuevo proyecto
    
    # Entorno virtual
    "tinyllama_env/",
]

# ============================================================================
# NUEVOS ARCHIVOS A CREAR
# ============================================================================

NEW_FILES = {
    "rag/dify_retriever.py": '''"""
DifyRetriever - Reemplaza FAISS con Dify como vector store
"""

import os
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DifyRetriever:
    """Retriever que consulta Dify API en lugar de FAISS"""
    
    def __init__(self, api_url: str = None, api_key: str = None, dataset_id: str = None):
        self.api_url = api_url or os.getenv("DIFY_API_URL", "http://localhost:5001/v1")
        self.api_key = api_key or os.getenv("DIFY_API_KEY")
        self.dataset_id = dataset_id or os.getenv("DIFY_DATASET_ID")
        self.top_k = int(os.getenv("DIFY_TOP_K", "5"))
        self.score_threshold = float(os.getenv("DIFY_SCORE_THRESHOLD", "0.3"))
        
        if not self.api_key:
            logger.warning("DIFY_API_KEY no configurada")
        if not self.dataset_id:
            logger.warning("DIFY_DATASET_ID no configurada")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Consultar Dify para recuperar chunks relevantes"""
        if not self.api_key or not self.dataset_id:
            logger.error("Dify no configurado correctamente")
            return []
        
        try:
            top_k = top_k or self.top_k
            url = f"{self.api_url}/datasets/{self.dataset_id}/retrieve"
            
            payload = {
                "query": query,
                "retrieval_model": {
                    "search_method": "hybrid_search",
                    "reranking_enable": False,
                    "reranking_model": {"provider": "", "model": ""},
                    "top_k": top_k,
                    "score_threshold": self.score_threshold
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("records", [])
            
            # Formatear a la estructura esperada
            chunks = []
            for record in records:
                segment = record.get("segment", {})
                content = segment.get("content", "")
                score = record.get("score", 0.0)
                metadata = segment.get("metadata", {})
                
                chunks.append({
                    "text": content,
                    "score": score,
                    "metadata": metadata,
                    "source": metadata.get("source", "desconocido")
                })
            
            logger.info(f"Dify recuperó {len(chunks)} chunks para: {query[:50]}...")
            return chunks
            
        except requests.RequestException as e:
            logger.error(f"Error consultando Dify: {e}")
            return []
        except Exception as e:
            logger.error(f"Error inesperado en Dify: {e}", exc_info=True)
            return []
    
    def get_document_count(self) -> int:
        """Obtener número de documentos en el dataset"""
        try:
            url = f"{self.api_url}/datasets/{self.dataset_id}/documents"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("total", 0)
            return 0
        except:
            return 0
''',

    "docker-compose.yml": '''version: '3.8'

services:
  dify-web:
    image: langgenius/dify-web:latest
    ports:
      - "5001:5001"
    environment:
      - CONSOLE_URL=${DIFY_CONSOLE_URL:-http://localhost:5001}
      - API_URL=${DIFY_API_URL:-http://localhost:5001}
    restart: unless-stopped

  dify-api:
    image: langgenius/dify-api:latest
    environment:
      - MODE=api
      - SECRET_KEY=${DIFY_SECRET_KEY:-sk-}
      - INIT_PASSWORD=${DIFY_INIT_PASSWORD:-password123}
      - DATABASE_URL=${DIFY_DATABASE_URL:-sqlite:///dify.db}
      - STORAGE_TYPE=local
      - STORAGE_LOCAL_PATH=/app/storage
      - VECTOR_STORE=weaviate
      - WEAVIATE_ENDPOINT=http://weaviate:8080
    volumes:
      - dify_data:/app/storage
    depends_on:
      - weaviate
    restart: unless-stopped

  weaviate:
    image: semitechnologies/weaviate:1.24.1
    ports:
      - "8080:8080"
    environment:
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: unless-stopped

  chatbot:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - DIFY_API_URL=http://dify-api:5001/v1
      - DIFY_API_KEY=${DIFY_API_KEY}
      - DIFY_DATASET_ID=${DIFY_DATASET_ID}
      - TIMEZONE=${TIMEZONE:-America/Mexico_City}
      - ENVIRONMENT=${ENVIRONMENT:-production}
    depends_on:
      - dify-api
    restart: unless-stopped

volumes:
  dify_data:
  weaviate_data:
''',

    ".env.example": '''# Ecotec Chatbot - Variables de Entorno
# Copiar a .env y llenar valores reales

# Groq
GROQ_API_KEY=tu_api_key_aqui

# Dify
DIFY_API_URL=http://localhost:5001/v1
DIFY_API_KEY=tu_api_key_de_dify
DIFY_DATASET_ID=id_del_dataset
DIFY_CONSOLE_URL=http://localhost:5001
DIFY_SECRET_KEY=clave_secreta_dify
DIFY_INIT_PASSWORD=password123
DIFY_TOP_K=5
DIFY_SCORE_THRESHOLD=0.3

# General
TIMEZONE=America/Mexico_City
ENVIRONMENT=development
LOG_LEVEL=INFO

# Telegram (para monitoreo)
TELEGRAM_BOT_TOKEN=tu_telegram_bot_token
TELEGRAM_CHAT_ID=tu_telegram_chat_id
''',
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def create_project():
    """Crear proyecto limpio desde cero"""
    print(f"🧹 Creando proyecto {PROJECT_NAME}...")
    print(f"📁 Origen: {SOURCE_DIR.absolute()}")
    print(f"📁 Destino: {TARGET_DIR.absolute()}")
    print()
    
    # 1. Eliminar destino si existe
    if TARGET_DIR.exists():
        print(f"🗑️  Eliminando {TARGET_DIR} existente...")
        shutil.rmtree(TARGET_DIR)
    
    # 2. Copiar proyecto base
    print(f"📋 Copiando archivos base...")
    shutil.copytree(
        SOURCE_DIR, 
        TARGET_DIR, 
        symlinks=False, 
        ignore_dangling_symlinks=True,
        ignore=shutil.ignore_patterns(
            '__pycache__', 
            '*.pyc', 
            '*.pyo',
            '.git',
            '.pytest_cache',
            '.coverage',
            'htmlcov'
        )
    )
    
    # 3. Eliminar archivos no deseados
    print(f"🗑️  Eliminando archivos no deseados...")
    for pattern in DELETE_PATTERNS:
        items = list(TARGET_DIR.glob(pattern))
        for item in items:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"   📁 Eliminado: {item.relative_to(TARGET_DIR)}")
            elif item.is_file():
                item.unlink()
                print(f"   📄 Eliminado: {item.relative_to(TARGET_DIR)}")
    
    # 4. Crear archivos nuevos
    print(f"📝 Creando archivos nuevos...")
    for filename, content in NEW_FILES.items():
        filepath = TARGET_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ Creado: {filename}")
    
    # 5. MODIFICAR archivos existentes
    print(f"🔧 Modificando archivos existentes...")
    modify_files(TARGET_DIR)
    
    # 6. Limpiar requirements.txt
    print(f"📦 Limpiando requirements.txt...")
    clean_requirements(TARGET_DIR)
    
    print()
    print("=" * 60)
    print("✅ PROYECTO CREADO EXITOSAMENTE!")
    print("=" * 60)
    print(f"📁 Ubicación: {TARGET_DIR.absolute()}")
    print()
    print("📋 PRÓXIMOS PASOS:")
    print("  1. cd ../ecotec-chatbot")
    print("  2. cp .env.example .env")
    print("  3. pip install -r requirements.txt")
    print("  4. docker-compose up -d  (para Dify)")
    print("  5. python app.py")
    print("=" * 60)


def modify_files(target_dir):
    """Modificar archivos existentes para eliminar referencias"""
    
    # 1. groq_wrapper.py - Eliminar referencias a Gemini/Ollama
    filepath = target_dir / "models" / "groq_wrapper.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        content = content.replace("from models.gemini_wrapper import", "# Gemini removed")
        content = content.replace("from models.ollama_wrapper import", "# Ollama removed")
        content = content.replace("self.fallback_model =", "# fallback_model removed")
        content = content.replace("GeminiWrapper", "# GeminiWrapper removed")
        content = content.replace("OllamaWrapper", "# OllamaWrapper removed")
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: models/groq_wrapper.py")
    
    # 2. settings.py - Agregar Dify, eliminar placeholders
    filepath = target_dir / "config" / "settings.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        
        # Eliminar placeholders
        content = content.replace("PLACEHOLDER_URLS", "# PLACEHOLDER_URLS removed")
        content = content.replace("PLACEHOLDER_DATES", "# PLACEHOLDER_DATES removed")
        
        # Agregar Dify si no existe
        if "DIFY_API_URL" not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'class Settings(BaseSettings):' in line:
                    insert_pos = i + 1
                    new_lines = [
                        '',
                        '    # Dify Configuration',
                        '    DIFY_API_URL: str = "http://localhost:5001/v1"',
                        '    DIFY_API_KEY: Optional[str] = None',
                        '    DIFY_DATASET_ID: Optional[str] = None',
                        '    DIFY_TOP_K: int = 5',
                        '    DIFY_SCORE_THRESHOLD: float = 0.3',
                    ]
                    lines[insert_pos:insert_pos] = new_lines
                    break
            content = '\n'.join(lines)
        
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: config/settings.py")
    
    # 3. rag/core.py - Eliminar placeholders
    filepath = target_dir / "rag" / "core.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        content = content.replace("resolver_placeholders", "# resolver_placeholders removed")
        content = content.replace("from rag.retriever import resolver_placeholders", "# Placeholders removed")
        content = content.replace("from rag.gemma_generator", "# GEMMA removed")
        content = content.replace("GemmaGenerator", "# GEMMA removed")
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: rag/core.py")
    
    # 4. rag/retriever.py - Reemplazar con Dify
    filepath = target_dir / "rag" / "retriever.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        content = content.replace("import faiss", "# FAISS removed")
        content = content.replace("from sentence_transformers", "# SentenceTransformers removed")
        
        # Marcar como deprecated y usar Dify
        content = """# DEPRECATED: Este archivo será reemplazado por DifyRetriever
# Mantenido por compatibilidad, pero usar rag.dify_retriever.DifyRetriever

\"\"\"
Retriever usando Dify como vector store
\"\"\"

import logging
from typing import List, Dict, Any, Optional

# Importar DifyRetriever
from rag.dify_retriever import DifyRetriever

logger = logging.getLogger(__name__)


class OptimizedRetriever:
    \"\"\"Wrapper para DifyRetriever - mantiene compatibilidad con código existente\"\"\"
    
    def __init__(self, top_k: int = 5):
        self.dify = DifyRetriever()
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        \"\"\"Recuperar documentos usando Dify\"\"\"
        return self.dify.retrieve(query, top_k or self.top_k)
    
    def get_document_count(self) -> int:
        \"\"\"Obtener número de documentos\"\"\"
        return self.dify.get_document_count()
"""
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: rag/retriever.py")
    
    # 5. rag/optimized_retriever.py - Usar Dify
    filepath = target_dir / "rag" / "optimized_retriever.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        content = content.replace("from sentence_transformers", "# SentenceTransformers removed")
        content = content.replace("import faiss", "# FAISS removed")
        content = """# DEPRECATED: Usar rag.retriever.OptimizedRetriever en su lugar
# Este archivo se mantiene por compatibilidad
            
from rag.retriever import OptimizedRetriever

# Re-exportar para compatibilidad
__all__ = ['OptimizedRetriever']
"""
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: rag/optimized_retriever.py")
    
    # 6. langchain_layer/wrappers.py - Actualizar
    filepath = target_dir / "langchain_layer" / "wrappers.py"
    if filepath.exists():
        content = read_file_safe(filepath)
        content = content.replace("resolver_placeholders", "# placeholders removed")
        content = content.replace("from rag.retriever import resolver_placeholders", "# placeholders removed")
        write_file_safe(filepath, content)
        print(f"   ✅ Modificado: langchain_layer/wrappers.py")


def clean_requirements(target_dir):
    """Limpiar requirements.txt"""
    filepath = target_dir / "requirements.txt"
    if filepath.exists():
        content = read_file_safe(filepath)
        
        to_remove = [
            "google-generativeai",
            "ollama",
            "faiss-cpu",
            "faiss-gpu",
            "sentence-transformers",
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "accelerate",
            "safetensors",
            "tokenizers",
        ]
        
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(remove in line.lower() for remove in to_remove):
                continue
            filtered_lines.append(line)
        
        new_deps = [
            "requests>=2.31.0",
            "python-dotenv>=1.0.0",
        ]
        
        existing = '\n'.join(filtered_lines).lower()
        for dep in new_deps:
            dep_name = dep.split('>=')[0].split('==')[0].lower()
            if dep_name not in existing:
                filtered_lines.append(dep)
        
        write_file_safe(filepath, '\n'.join(filtered_lines))
        print(f"   ✅ Limpiado: requirements.txt")


def read_file_safe(filepath):
    """Leer archivo con encoding seguro"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            with open(filepath, 'r', encoding='cp1252', errors='ignore') as f:
                return f.read()


def write_file_safe(filepath, content):
    """Escribir archivo con encoding UTF-8"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not (SOURCE_DIR / "api").exists() or not (SOURCE_DIR / "rag").exists():
        print("❌ ERROR: Este script debe ejecutarse desde el directorio raíz del proyecto Chatbot-RAG-Fuente-Base")
        print(f"   Directorio actual: {SOURCE_DIR.absolute()}")
        print("   No se encuentran las carpetas api/ o rag/")
        sys.exit(1)
    
    # Preguntar confirmación
    print("=" * 60)
    print("⚠️  ADVERTENCIA")
    print("=" * 60)
    print(f"Este script creará el proyecto '{PROJECT_NAME}' en:")
    print(f"  {TARGET_DIR.absolute()}")
    print()
    print("=" * 60)
    
    response = input("¿Continuar? (si/no): ").lower().strip()
    if response not in ['si', 's', 'yes', 'y']:
        print("❌ Operación cancelada")
        sys.exit(0)
    
    print()
    create_project()