import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuración centralizada para el sistema RAG de Prepa en Línea SEP
    """
    
    # ===== ENTORNO Y DEPLOYMENT =====
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ===== API CONFIGURATION =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    # ===== RAG CORE CONFIGURATION =====
    # Pipeline principal
    RAG_ENABLED: bool = True
    INTENTS_ENABLED: bool = True  # Si quieres poder desactivar intents fácilmente
    
    # Búsqueda y recuperación
    TOP_K_RESULTS: int = Field(default=3, ge=1, le=10)
    SIMILARITY_THRESHOLD: float = Field(default=0.75, ge=0.1, le=1.0)
    MAX_CONTEXT_LENGTH: int = 4000  # Tokens máximos para contexto
    
    # ===== EMBEDDING MODEL CONFIGURATION =====
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_DIMENSIONS: int = 384  # Dimensiones fijas para MiniLM-L12
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" o "cuda"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ===== VECTOR DATABASE (FAISS) CONFIGURATION =====
    # Confirmar que usas FAISS según tu código
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss", "chroma", "pinecone"
    
    # FAISS específico
    FAISS_INDEX_TYPE: str = "FlatL2"  # "FlatL2", "IVFFlat", "IVFPQ"
    FAISS_METRIC: str = "cosine"      # "cosine", "l2", "inner_product"
    FAISS_NLIST: int = 100  # Para índices IVF (opcional)
    FAISS_NPROBE: int = 10  # Para búsquedas IVF (opcional)

    FAISS_PERSIST_DIR: str = "./data/vector_store"
    
    # Persistencia
    FAISS_INDEX_PATH: str = "./data/vector_store/faiss_index.bin"
    DOCUMENTS_METADATA_PATH: str = "./data/vector_store/documents.json"
    
    # ===== RESPONSE GENERATOR CONFIGURATION =====
    # Modelo para generación (si usas uno)
    GENERATION_MODEL: str = "hackathon-somos-nlp-2023/BSC-LT-Project/roberta-base-bne-capitel-ner-plus"
    GENERATION_MAX_LENGTH: int = 500
    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_DO_SAMPLE: bool = True
    
    # Si NO usas generación con modelo, sino templates:
    USE_TEMPLATE_RESPONSES: bool = True
    RESPONSE_TEMPLATE: str = """
    Basado en los documentos de Prepa en Línea SEP:

    {context}

    Respuesta:
    """
    
    # ===== DOCUMENT PROCESSING =====
    CHUNK_SIZE: int = Field(default=768, ge=128, le=2048)
    CHUNK_OVERLAP: int = Field(default=128, ge=0, le=512)
    
    # Procesamiento de Excel
    EXCEL_SHEET_NAMES: List[str] = ["Sheet1", "Tickets", "Respuestas"]
    EXCEL_REQUIRED_COLUMNS: List[str] = ["Asunto", "Descripción", "Respuesta Institucional"]
    
    # ===== INTENTS CONFIGURATION =====
    INTENTS_FILE_PATH: str = "./data/vector_store/intents.json"
    INTENTS_MIN_CONFIDENCE: float = 0.95  # Confianza mínima para usar intents
    INTENTS_MAX_QUERY_LENGTH: int = 50    # Longitud máxima para considerar intents
    
    # ===== MONITORING & LOGGING =====
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_FILE_PATH: str = "./logs/chatbot.log"
    
    # ===== CACHE CONFIGURATION =====
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 300  # 5 minutos
    CACHE_MAX_SIZE: int = 1000
    
    # ===== SECURITY =====
    ENABLE_RATE_LIMITING: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # segundos
    
    # ===== AWS (FUTURO) =====
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = "us-east-1"
    S3_BUCKET: Optional[str] = None
    S3_VECTOR_STORE_PATH: Optional[str] = None
    
    # ===== RENDER/DEPLOYMENT =====
    RENDER_DEPLOYMENT: bool = False
    DOCKER_IMAGE: str = "python:3.12-slim"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Ejemplo de validación personalizada
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # Prioridad: variables de entorno > .env > valores por defecto
            return env_settings, init_settings, file_secret_settings

# Instancia global de configuración
settings = Settings()

# Validaciones adicionales
def validate_settings():
    """Validaciones de consistencia de configuración"""
    errors = []
    
    # Validar que FAISS tenga las dimensiones correctas
    if settings.EMBEDDING_MODEL_DIMENSIONS != 384:
        errors.append(f"MiniLM-L12 debe tener 384 dimensiones, no {settings.EMBEDDING_MODEL_DIMENSIONS}")
    
    # Validar que CHUNK_OVERLAP sea menor que CHUNK_SIZE
    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({settings.CHUNK_OVERLAP}) debe ser menor que CHUNK_SIZE ({settings.CHUNK_SIZE})")
    
    # Validar umbral de similitud
    if not 0 <= settings.SIMILARITY_THRESHOLD <= 1:
        errors.append(f"SIMILARITY_THRESHOLD debe estar entre 0 y 1, no {settings.SIMILARITY_THRESHOLD}")
    
    # Validar configuración de FAISS
    valid_faiss_metrics = ["cosine", "l2", "inner_product"]
    if settings.FAISS_METRIC not in valid_faiss_metrics:
        errors.append(f"FAISS_METRIC debe ser uno de {valid_faiss_metrics}, no {settings.FAISS_METRIC}")
    
    if errors:
        raise ValueError("Errores en configuración:\n" + "\n".join(f"  • {e}" for e in errors))

# Ejecutar validaciones al importar
try:
    validate_settings()
except ValueError as e:
    import logging
    logging.error(str(e))
    raise

# Función para mostrar configuración actual (útil para debug)
def print_config_summary():
    """Muestra resumen de configuración"""
    import logging
    logger = logging.getLogger(__name__)
    
    summary = f"""
    ============================================
    CONFIGURACIÓN RAG - PREPA EN LÍNEA SEP
    ============================================
    Entorno: {settings.ENVIRONMENT}
    Debug: {settings.DEBUG}
    
    🌐 API:
      Host: {settings.API_HOST}:{settings.API_PORT}
      Workers: {settings.API_WORKERS}
    
    🤖 RAG:
      Modelo Embedding: {settings.EMBEDDING_MODEL}
      Dimensiones: {settings.EMBEDDING_MODEL_DIMENSIONS}
      Top K resultados: {settings.TOP_K_RESULTS}
      Umbral similitud: {settings.SIMILARITY_THRESHOLD}
    
    🗄️ Vector Store:
      Tipo: {settings.VECTOR_STORE_TYPE}
      Índice FAISS: {settings.FAISS_INDEX_TYPE}
      Métrica: {settings.FAISS_METRIC}
    
    📄 Documentos:
      Chunk Size: {settings.CHUNK_SIZE}
      Chunk Overlap: {settings.CHUNK_OVERLAP}
    
    💬 Intents:
      Habilitados: {settings.INTENTS_ENABLED}
      Confianza mínima: {settings.INTENTS_MIN_CONFIDENCE}
    ============================================
    """
    
    logger.info(summary)