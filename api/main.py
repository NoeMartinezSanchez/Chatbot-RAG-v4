from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import uuid
from datetime import datetime
import os
import sys
import time

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
from data.build_menu_json import load_menu_json
from evaluation.performance_logger import log_latency
from evaluation.automated_evaluator import run_automated_evaluation

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

# Estado del menú jerárquico
app.state.menu = {}

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Inicializar sistema al arrancar"""
    try:
        print_config_summary()
        
        # Cargar intents
        rag_system.load_intents("data/intents.json")
        
        # Cargar menú desde JSON (NO desde Excel en producción)
        menu_json_path = "data/menu.json"
        
        if os.path.exists(menu_json_path):
            app.state.menu = load_menu_json(menu_json_path)
            logger.info(f"✅ Menú cargado desde: {menu_json_path}")
            logger.info(f"   Categorías: {len(app.state.menu)}")
        else:
            logger.warning(f"⚠️ Archivo menu.json no encontrado: {menu_json_path}")
            logger.warning("   El menú jerárquico no estará disponible")
            logger.warning("   Para generar menu.json, ejecuta: python data/build_menu_json.py")
            app.state.menu = {}
        
        logger.info("Sistema RAG inicializado correctamente")
        logger.info("Interfaz web disponible en: http://localhost:8000")
        logger.info("API Docs disponible en: http://localhost:8000/api/docs")
        
        # Ejecutar evaluación automática en segundo plano
        logger.info("🚀 Iniciando evaluación automática...")
        
        # Verificar que existe el archivo de test
        test_set_path = "evaluation/test_set.json"
        if os.path.exists(test_set_path):
            logger.info(f"✅ test_set.json encontrado: {test_set_path}")
            run_automated_evaluation(
                retriever=rag_system.optimized_retriever,
                generator=rag_system.generator,
                test_set_path=test_set_path
            )
        else:
            logger.warning(f"⚠️ test_set.json no encontrado en: {test_set_path}")
            logger.warning("   La evaluación automática no se ejecutará")
            # Listar archivos en evaluation/
            eval_dir = "evaluation"
            if os.path.exists(eval_dir):
                files = os.listdir(eval_dir)
                logger.info(f"   Archivos en evaluation/: {files}")
            else:
                logger.warning(f"   Directorio evaluation/ no existe")
        
    except Exception as e:
        logger.error(f"Error inicializando RAG: {e}")
        app.state.menu = {}

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
    start_time = time.time()
    retrieval_start = start_time
    
    try:
        logger.info(f"📩 Mensaje recibido: {request.message[:50]}...")
        
        # Generar IDs si no existen
        user_id = request.user_id or str(uuid.uuid4())
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Detectar saludos y responder directamente sin RAG
        msg_lower = request.message.lower().strip()
        saludos = ["hola", "buenos días", "buenas tardes", "buenas", "holi", "hello", "hey", "qué tal", "cómo estás", "buen día"]
        despedidas = ["adiós", "chao", "bye", "hasta luego", "me voy", "nos vemos", "me retiro"]
        gracias = ["gracias", "thank", "agradezco", "muchas gracias", "te agradezco"]
        
        if any(s in msg_lower for s in saludos):
            response_text = "¡Hola! Bienvenido a Prepa en Línea SEP. Estoy aquí para ayudarte con tus dudas sobre el programa. ¿Qué necesitas saber?"
            is_rag = False
            confidence = 1.0
            sources = []
            retrieval_time = 0
            generation_time = 0
        elif any(s in msg_lower for s in despedidas):
            response_text = "¡Hasta luego! Éxito en tus estudios. Cuando tengas dudas sobre Prepa en Línea, vuelve a escribirme."
            is_rag = False
            confidence = 1.0
            sources = []
            retrieval_time = 0
            generation_time = 0
        elif any(s in msg_lower for s in gracias):
            response_text = "¡De nada! Si tienes más dudas sobre Prepa en Línea, con gusto te ayudo. ¡Éxito en tus estudios!"
            is_rag = False
            confidence = 1.0
            sources = []
            retrieval_time = 0
            generation_time = 0
        else:
            # Procesar consulta normal con RAG
            retrieval_end = time.time()
            retrieval_time = (retrieval_end - retrieval_start) * 1000
            
            generation_start = time.time()
            response_text, is_rag, confidence, sources = rag_system.process_query(
                request.message
            )
            generation_end = time.time()
            generation_time = (generation_end - generation_start) * 1000
        
        # DEBUG: Verificar qué se recibe
        logger.info(f"🔍 DEBUG - response_text tipo: {type(response_text)}, largo: {len(response_text) if response_text else 0}")
        logger.info(f"🔍 DEBUG - response_text contenido: '{response_text[:100]}...'")
        logger.info(f"🔍 DEBUG - sources count: {len(sources) if sources else 0}")
        logger.info(f"📤 Respuesta generada: {'RAG' if is_rag else 'Intent'} - Confianza: {confidence:.2%}")
        
        # Crear respuesta
        conf_value = confidence if confidence is not None else 0.5
        response = ChatResponse(
            response=response_text,
            sources=sources,
            is_rag_response=is_rag,
            confidence=conf_value
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
        
        # Añadir headers con IDs (sin confianza)
        headers = {
            "X-User-ID": user_id,
            "X-Conversation-ID": conversation_id,
            "X-Message-ID": message_id,
            "X-Response-Type": "rag" if is_rag else "intent",
        }
        
        # Log de latencia
        total_time = (time.time() - start_time) * 1000
        tokens_generated = len(response_text.split()) if response_text else 0
        
        log_latency(
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            tokens_generated=tokens_generated,
            question=request.message
        )
        
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

@app.get("/menu")
async def get_menu():
    """Endpoint para obtener la estructura del menú jerárquico"""
    if hasattr(app.state, 'menu') and app.state.menu:
        return {"menu": app.state.menu}
    return {"menu": {}}

@app.get("/dashboard")
async def get_dashboard():
    """Servir el dashboard HTML de evaluación"""
    import os
    import tempfile
    from pathlib import Path
    
    # Leer desde /tmp (ubicación persistente en HF Spaces)
    dashboard_path = Path(tempfile.gettempdir()) / "dashboard.html"
    
    logger.info(f"🔍 Buscando dashboard en: {dashboard_path}")
    logger.info(f"🔍 Existe: {dashboard_path.exists()}")
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    else:
        # Listar archivos en /tmp para debugging
        temp_dir = Path(tempfile.gettempdir())
        temp_files = list(temp_dir.glob("*")) if temp_dir.exists() else []
        logger.info(f"📁 Archivos en /tmp: {temp_files[:10]}")  # Primeros 10
        
        return {
            "status": "no_evaluation",
            "message": "La evaluación se está ejecutando o no ha terminado todavía",
            "dashboard_path": str(dashboard_path),
            "dashboard_exists": dashboard_path.exists(),
            "temp_files_count": len(temp_files)
        }

@app.get("/evaluation-results")
async def get_evaluation_results():
    """Obtener resultados de evaluación en JSON"""
    import json
    import tempfile
    from pathlib import Path
    
    # Leer desde /tmp
    results_path = Path(tempfile.gettempdir()) / "logs" / "evaluation_results.jsonl"
    
    if not results_path.exists():
        return {"results": [], "message": "No hay resultados de evaluación"}
    
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return {"results": results, "total": len(results)}


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