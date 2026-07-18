from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import json
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
from evaluation.show_results import show_results
from langchain_layer.wrappers import LangChainRAGWrapper

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar handler para logs del dashboard
from utils.log_capture import DashboardLogHandler
dashboard_log_handler = DashboardLogHandler()
dashboard_log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(dashboard_log_handler)

print("=" * 50)
print("🚀 CARGANDO API - Registro de endpoints:")
print("=" * 50)

# Inicializar aplicación
app = FastAPI(
    title="Asistente Educativo RAG - Prepa en Línea SEP",
    description="Sistema de asistencia educativa 24/7 con RAG para Prepa en Línea SEP",
    version="2.0.0",
    docs_url="/api/docs",  # Cambiado de /docs a /api/docs
    redoc_url="/api/redoc"
)

# ============================================================
# ENDPOINT DEL DASHBOARD
# ============================================================
@app.get("/dashboard")
async def get_dashboard():
    """Servir el dashboard desde múltiples ubicaciones posibles"""
    import os
    
    posibles_ubicaciones = [
        "/data/dashboard.html",
        "static/dashboard.html",
        "/app/static/dashboard.html",
        "evaluation/dashboard.html"
    ]
    
    for path in posibles_ubicaciones:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            logger.info(f"✅ Dashboard encontrado en: {path}")
            return HTMLResponse(content=html_content)
    
    # Si no existe, mostrar diagnóstico
    data_dir = "/data"
    static_dir = "static"
    
    return HTMLResponse(content=f"""
    <html>
    <head><title>Dashboard no disponible</title></head>
    <body>
    <h1>📊 Dashboard no disponible</h1>
    <p>El dashboard no se ha generado aún.</p>
    <h2>Contenido de /data:</h2>
    <pre>{os.listdir(data_dir) if os.path.exists(data_dir) else 'No existe'}</pre>
    <h2>Contenido de static/:</h2>
    <pre>{os.listdir(static_dir) if os.path.exists(static_dir) else 'No existe'}</pre>
    </body>
    </html>
    """)

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
langchain_wrapper = LangChainRAGWrapper(rag_system, memory_enabled=True)

# Almacenamiento simple en memoria para feedback
feedback_store = {}
conversation_store = {}

# Estado del menú jerárquico
app.state.menu = {}

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

print("✅ Endpoints registrados hasta ahora:")
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        print(f"   {list(route.methods)[0] if route.methods else 'GET'} {route.path}")

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
        
        # ============================================================
        # DIAGNÓSTICO DEL VECTOR STORE
        # ============================================================
        logger.info("=" * 60)
        logger.info("🔍 DIAGNÓSTICO DEL VECTOR STORE")
        logger.info("=" * 60)
        
        vector_store_path = "data/vector_store"
        logger.info(f"📁 Revisando: {vector_store_path}")
        logger.info(f"   ¿Existe? {os.path.exists(vector_store_path)}")
        
        if os.path.exists(vector_store_path):
            files = os.listdir(vector_store_path)
            logger.info(f"   Archivos encontrados: {files}")
            for f in files:
                fpath = os.path.join(vector_store_path, f)
                size = os.path.getsize(fpath)
                logger.info(f"      - {f} ({size} bytes)")
        
        # Verificar el índice FAISS
        try:
            from rag.retriever import VectorStoreFAISS
            logger.info("🔄 Intentando cargar VectorStoreFAISS...")
            vs = VectorStoreFAISS()
            if vs.index:
                logger.info(f"   ✅ Índice FAISS cargado correctamente")
                logger.info(f"   📊 Número de vectores: {vs.index.ntotal}")
                logger.info(f"   📐 Dimensión: {vs.embedding_dim}")
            else:
                logger.warning("   ❌ El índice FAISS es None")
        except Exception as e:
            logger.error(f"   ❌ Error cargando índice: {e}")
        
        # Probar una búsqueda de ejemplo
        try:
            from rag.embeddings import EmbeddingModel
            embedder = EmbeddingModel()
            test_query = "¿El módulo propedéutico es obligatorio?"
            query_embedding = embedder.embed_query(test_query)
            logger.info(f"🔍 Probando búsqueda con: '{test_query[:50]}...'")
            
            if vs and vs.index:
                q_emb = query_embedding.reshape(1, -1).astype('float32')
                distances, indices = vs.index.search(q_emb, 3)
                distancia_array = distances[0]
                num_results = distancia_array.size if hasattr(distancia_array, 'size') else len(distancia_array)
                distancias_lista = distancia_array.tolist() if hasattr(distancia_array, 'tolist') else list(distancia_array)
                logger.info(f"   Resultados (distancias): {distancias_lista}")
                if num_results > 0 and distancia_array[0] < 1000:
                    logger.info(f"   ✅ Búsqueda exitosa! Distancia: {float(distancia_array[0]):.4f}")
                else:
                    logger.warning(f"   ❌ Sin resultados próximos. Distancias: {distancias_lista}")
            else:
                logger.warning("   ❌ No se pudo probar búsqueda - índice no disponible")
        except Exception as e:
            logger.error(f"   ❌ Error en búsqueda de prueba: {e}")
        
        logger.info("=" * 60)
        
        # [DESACTIVADO] Evaluación automática desactivada para ahorrar tokens de API.
        # Para ejecutarla manualmente, usar: GET /evaluation-results
        logger.info("⏭️ Evaluación automática desactivada (ahorro de tokens)")
        
        # Generar dashboard de usuarios automáticamente
        try:
            from evaluation.generate_user_dashboard import generate_user_dashboard
            generate_user_dashboard()
            logger.info("✅ Dashboard de usuarios generado automáticamente")
        except Exception as e:
            logger.error(f"❌ Error generando dashboard de usuarios: {e}")
        
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
        
        # Usar LangChain wrapper (detecta saludos, fecha, memoria, RAG)
        wrapper_result = langchain_wrapper.query_with_memory(
            question=request.message,
            session_id=request.session_id or "default"
        )
        response_text = wrapper_result["response"]
        is_rag = wrapper_result["is_rag_response"]
        confidence = wrapper_result["confidence"]
        sources = wrapper_result.get("sources", [])
        retrieval_time = 0
        generation_time = 0

        logger.info(f"🔍 DEBUG - response_text tipo: {type(response_text)}, largo: {len(response_text) if response_text else 0}")
        logger.info(f"🔍 DEBUG - response_text contenido: '{response_text[:100]}...'")
        logger.info(f"🔍 DEBUG - sources count: {len(sources) if sources else 0}")
        logger.info(f"📤 Respuesta generada: {'RAG' if is_rag else 'Directo'} - Confianza: {confidence:.2%}")
        
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
        
        # Guardar interacción de usuario para dashboard dinámico
        try:
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "pregunta": request.message,
                "respuesta": response_text,
                "tiempo_total_ms": round(total_time, 2),
                "tiempo_retrieval_ms": round(retrieval_time, 2),
                "tiempo_generacion_ms": round(generation_time, 2),
                "confianza": round(conf_value, 4),
                "fuentes_usadas": list(set(s.get("metadata", {}).get("source_file", "unknown") for s in sources)) if sources else [],
                "es_rag": is_rag,
                "tokens_generados": tokens_generated,
                "session_id": conversation_id
            }
            
            log_file = "/data/user_interactions.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
            logger.info(f"✅ Interacción guardada en {log_file}")
        except Exception as e:
            logger.error(f"❌ Error guardando interacción: {e}")
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Error en chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/security/stats")
async def get_security_stats():
    """Estadísticas de incidentes de seguridad en tiempo real"""
    from security.monitor import get_monitor
    monitor = get_monitor()
    return {
        "status": "success",
        "data": monitor.get_stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/security/incidents")
async def get_security_incidents(limit: int = 50, min_severity: str = "low"):
    """Incidentes de seguridad recientes"""
    from security.monitor import get_monitor
    monitor = get_monitor()
    incidents = monitor.get_recent_incidents(limit=limit, min_severity=min_severity)
    return {
        "status": "success",
        "incidents": incidents,
        "count": len(incidents),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/security/validate")
async def validate_text(text: str):
    """Validar texto contra el sanitizer (pruebas manuales)"""
    from security.sanitizer import InputSanitizer
    result = InputSanitizer.sanitize(text)
    return {
        "status": "success",
        "is_safe": result.is_safe,
        "severity": result.severity,
        "threats": [
            {"type": t.threat_type, "severity": t.severity, "position": t.position, "snippet": t.snippet[:60]}
            for t in result.threats
        ],
        "threat_count": len(result.threats)
    }

@app.get("/menu")
async def get_menu():
    """Endpoint para obtener la estructura del menú jerárquico"""
    if hasattr(app.state, 'menu') and app.state.menu:
        return {"menu": app.state.menu}
    return {"menu": {}}


@app.get("/evaluation-results")
async def get_evaluation_results():
    """Obtener resultados de evaluación en JSON"""
    return show_results()

@app.get("/evaluation-summary")
async def get_evaluation_summary():
    """Mostrar resumen de evaluación en HTML"""
    try:
        from evaluation.show_results import show_results
        import io
        import sys
        
        # Capturar output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        show_results()
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return HTMLResponse(content=f"""
        <html>
        <head><title>Resultados - Prepa en Línea SEP</title></head>
        <body>
        <pre>{output}</pre>
        </body>
        </html>
        """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)
    
    # Leer desde logs/
    base_dir = Path(__file__).parent.resolve().parent
    results_path = base_dir / "logs" / "evaluation_results.jsonl"
    
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


@app.get("/user-dashboard", response_class=HTMLResponse)
async def get_user_dashboard():
    """Servir el dashboard de interacciones de usuarios"""
    dashboard_path = "/data/user_dashboard.html"
    
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    return HTMLResponse(content="<h1>Dashboard no disponible</h1><p>Aún no hay interacciones de usuarios.</p>", status_code=202)


@app.get("/user-dashboard/refresh")
async def refresh_user_dashboard():
    """Regenerar el dashboard de interacciones de usuarios"""
    try:
        from evaluation.generate_user_dashboard import generate_user_dashboard
        
        dashboard_path = "/data/user_dashboard.html"
        generate_user_dashboard(output_path=dashboard_path)
        
        return {"status": "success", "message": "Dashboard de usuarios regenerado"}
    except Exception as e:
        logger.error(f"❌ Error regenerando dashboard: {e}")
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)


@app.get("/debug-user-logs")
async def debug_user_logs():
    """Diagnosticar acceso al archivo de logs de usuarios"""
    import os
    log_file = "/data/user_interactions.jsonl"
    result = {
        "file_exists": os.path.exists(log_file),
        "file_size": os.path.getsize(log_file) if os.path.exists(log_file) else 0,
        "data_dir_contents": os.listdir("/data") if os.path.exists("/data") else []
    }
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            result["first_line"] = f.readline()
    
    return result


@app.get("/api/logs")
async def get_logs(level: str = None, limit: int = 200, since: str = None):
    from pathlib import Path
    logs = []
    log_file = Path("data/system_logs.jsonl")
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    if level:
        logs = [l for l in logs if l.get("level") == level.upper()]
    if since:
        logs = [l for l in logs if l.get("timestamp", "") > since]
    total = len(logs)
    return {"logs": logs[:limit], "total": total, "available_levels": ["INFO", "WARNING", "ERROR", "DEBUG"]}


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