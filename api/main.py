from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import uuid
from datetime import datetime
import os
import sys

# AÑADIR ESTAS LÍNEAS PARA PRODUCCIÓN
# Asegurar que el directorio raíz está en el path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Subir un nivel desde api/
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Project root added to path: {project_root}")


from config.settings import settings, print_config_summary
from config.models import ChatRequest, ChatResponse, FeedbackRequest
from rag.core import RAGSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar aplicación
app = FastAPI(
    title="Asistente Educativo RAG - Prepa en Línea SEP",
    description="Sistema de asistencia educativa 24/7 con RAG para Prepa en Línea SEP",
    version="2.0.0",
    docs_url="/api/docs",  # Cambiado de /docs a /api/docs
    redoc_url="/api/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG
rag_system = RAGSystem()

# Almacenamiento simple en memoria para feedback
feedback_store = {}
conversation_store = {}

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Inicializar sistema al arrancar"""
    try:
        print_config_summary()  # <-- MUESTRA CONFIGURACIÓN
        # Cargar intents
        rag_system.load_intents("data/intents.json")
        logger.info("Sistema RAG inicializado correctamente")
        logger.info("Interfaz web disponible en: http://localhost:8000")
        logger.info("API Docs disponible en: http://localhost:8000/api/docs")
    except Exception as e:
        logger.error(f"Error inicializando RAG: {e}")

@app.get("/")
async def root():
    """Servir la interfaz web principal"""
    # Verificar si el archivo index.html existe
    index_path = "static/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {
            "status": "online",
            "service": "Asistente Educativo RAG - Prepa en Línea SEP",
            "version": "2.0.0",
            "endpoints": {
                "web_interface": "http://localhost:8000/",
                "api_docs": "http://localhost:8000/api/docs",
                "chat": "POST /chat",
                "health": "GET /health",
                "stats": "GET /stats",
                "feedback": "POST /feedback"
            },
            "note": "Para la interfaz web, crea static/index.html"
        }

@app.get("/health")
async def health():
    """Health check para Render"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "chatbot-rag-api",
        "version": "2.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint principal para chat"""
    try:
        logger.info(f"📩 Mensaje recibido: {request.message[:50]}...")
        
        # Generar IDs si no existen
        user_id = request.user_id or str(uuid.uuid4())
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Procesar consulta
        response_text, is_rag, confidence, sources = rag_system.process_query(
            request.message
        )
        
        # DEBUG: Verificar qué se recibe
        logger.info(f"🔍 DEBUG - response_text tipo: {type(response_text)}, largo: {len(response_text) if response_text else 0}")
        logger.info(f"🔍 DEBUG - response_text contenido: '{response_text[:100]}...'")
        logger.info(f"🔍 DEBUG - sources count: {len(sources) if sources else 0}")
        logger.info(f"📤 Respuesta generada: {'RAG' if is_rag else 'Intent'} - Confianza: {confidence:.2%}")
        
        # Crear respuesta
        response = ChatResponse(
            response=response_text,
            sources=sources,
            is_rag_response=is_rag,
            confidence=confidence
        )
        
        # Almacenar conversación
        message_id = str(uuid.uuid4())
        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = []
        
        conversation_store[conversation_id].append({
            "message_id": message_id,
            "user_message": request.message,
            "assistant_response": response_text,
            "timestamp": datetime.now().isoformat(),
            "is_rag": is_rag,
            "confidence": confidence,
            "sources": sources
        })
        
        # Añadir headers con IDs
        headers = {
            "X-User-ID": user_id,
            "X-Conversation-ID": conversation_id,
            "X-Message-ID": message_id,
            "X-Response-Type": "rag" if is_rag else "intent",
            "X-Confidence": str(confidence)
        }
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Error en chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error procesando la consulta")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Endpoint para recibir feedback"""
    try:
        feedback_store[request.message_id] = {
            "conversation_id": request.conversation_id,
            "is_helpful": request.is_helpful,
            "feedback_text": request.feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"📝 Feedback recibido: {request.message_id} - Útil: {request.is_helpful}")
        
        return {
            "status": "success",
            "message": "Feedback registrado",
            "message_id": request.message_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error guardando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error guardando feedback")

@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema"""
    try:
        # Obtener estadísticas del sistema RAG
        rag_stats = rag_system.get_stats()
        
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat()
            },
            "rag_system": rag_stats,
            "conversations": {
                "total_conversations": len(conversation_store),
                "total_messages": sum(len(msgs) for msgs in conversation_store.values()),
                "feedback_count": len(feedback_store)
            },
            "endpoints": {
                "web_interface": "/",
                "api_documentation": "/api/docs",
                "chat_endpoint": "POST /chat",
                "feedback_endpoint": "POST /feedback"
            }
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        return {
            "system": {
                "status": "operational",
                "error": str(e)
            }
        }

@app.get("/api/rag-stats")
async def get_rag_stats():
    """Estadísticas detalladas del sistema RAG"""
    try:
        stats = rag_system.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Iniciando servidor en {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"📁 Directorio estático: {os.path.abspath('static')}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )