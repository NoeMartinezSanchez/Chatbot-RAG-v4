from fastapi import APIRouter, UploadFile, File, HTTPException
import json
import tempfile
import os
from typing import List
import logging
from datetime import datetime
from langchain_layer.wrappers import LangChainRAGWrapper
from pydantic import BaseModel
from typing import Optional

from config.models import Document
from api import models as api_models
from mongodb.models import FeedbackCreate
from mongodb.services import ConversationService, MetricsService, FeedbackService

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

router = APIRouter(prefix="/documents", tags=["documents"])
mongodb_router = APIRouter(tags=["mongodb"])
logger = logging.getLogger(__name__)

# Instancia global del sistema RAG (carga diferida para no bloquear el import)
_rag_system = None

def get_rag_system():
    """Retorna el sistema RAG (lazy)"""
    global _rag_system
    if _rag_system is None:
        from rag.core import RAGSystem
        _rag_system = RAGSystem()
    return _rag_system

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Subir documento para enriquecer la base de conocimientos"""
    try:
        # Leer contenido
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Extraer metadata básica
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Procesar documento
        get_rag_system().add_document(text_content, metadata)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "Documento procesado correctamente"
        }
        
    except Exception as e:
        logger.error(f"Error subiendo documento: {e}")
        raise HTTPException(status_code=500, detail="Error procesando documento")

@router.post("/upload-json")
async def upload_json_documents(documents: List[Document]):
    """Subir documentos en formato estructurado"""
    try:
        processed_count = 0
        
        for doc in documents:
            get_rag_system().add_document(doc.content, doc.metadata)
            processed_count += 1
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "message": f"{processed_count} documentos procesados"
        }
        
    except Exception as e:
        logger.error(f"Error subiendo documentos JSON: {e}")
        raise HTTPException(status_code=500, detail="Error procesando documentos")

@router.get("/search")
async def search_documents(query: str, top_k: int = 5):
    """Buscar directamente en documentos"""
    try:
        from rag.embeddings import EmbeddingModel
        from rag.retriever import VectorStoreFAISS
        
        embedder = EmbeddingModel()
        vector_store = VectorStoreFAISS()
        
        query_embedding = embedder.embed_text(query).tolist()
        results = vector_store.search_documents(query_embedding, top_k=top_k)
        
        # Formatear resultados
        formatted_results = []
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                formatted_results.append({
                    "rank": i + 1,
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "metadata": metadata,
                    "similarity": 1 - (results['distances'][0][i] if results['distances'] else 0)
                })
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error buscando documentos: {e}")
        raise HTTPException(status_code=500, detail="Error buscando documentos")

# Inicialización del wrapper LangChain (lazy)
_langchain_wrapper = None

def get_langchain_wrapper():
    global _langchain_wrapper
    if _langchain_wrapper is None:
        _langchain_wrapper = LangChainRAGWrapper(get_rag_system(), memory_enabled=False, mongodb_enabled=True)
    return _langchain_wrapper

@router.post("/chat/v2")
async def chat_v2(request: ChatRequest):
    """
    Nuevo endpoint con LangChain (memoria en desarrollo + persistencia MongoDB)
    """
    wrapper = get_langchain_wrapper()
    result = await wrapper.query_with_memory(request.message, request.session_id or "default")
    
    return {
        "response": result["response"],
        "sources": result.get("sources", []),
        "is_rag_response": result.get("is_rag_response", True),
        "confidence": result.get("confidence", 0.0),
        "session_id": request.session_id,
        "conversation_id": result.get("conversation_id"),
        "langchain_version": True
    }

@router.post("/chat/clear_memory")
async def clear_memory(session_id: str = "default"):
    """
    Limpiar memoria (placeholder)
    """
    wrapper = get_langchain_wrapper()
    return wrapper.clear_memory(session_id)

# ============================================================
# ENDPOINTS MONGODB (rutas /chat, /feedback, /analytics)
# ============================================================

@mongodb_router.post("/chat", response_model=api_models.ChatResponse)
async def chat_mongodb(request: api_models.ChatRequest):
    """Chat con persistencia en MongoDB (schema `question`, async wrapper)."""
    wrapper = get_langchain_wrapper()
    result = await wrapper.query_with_memory(
        question=request.question,
        session_id=request.session_id,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
    )
    return api_models.ChatResponse(
        response=result["response"],
        sources=result.get("sources", []),
        is_rag_response=result.get("is_rag_response", True),
        confidence=result.get("confidence", 0.0),
        conversation_id=result.get("conversation_id"),
        session_id=request.session_id,
    )


@mongodb_router.post("/feedback")
async def feedback_mongodb(request: api_models.FeedbackRequest):
    """Registrar feedback en MongoDB."""
    user_rating = request.user_rating
    is_correct = request.is_correct
    if user_rating is None and request.is_helpful is not None:
        user_rating = 5 if request.is_helpful else 1
        is_correct = request.is_helpful

    feedback_service = FeedbackService()
    feedback_id = await feedback_service.record_feedback(FeedbackCreate(
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        message_index=request.message_index or 0,
        user_rating=user_rating,
        user_comment=request.user_comment or request.feedback_text,
        is_correct=is_correct,
    ))
    return {"status": "success", "feedback_id": feedback_id}


@mongodb_router.get("/analytics")
async def analytics_mongodb(session_id: Optional[str] = None, days: int = 7):
    """Analíticas combinadas (conversaciones, salud del sistema, feedback)."""
    conv_service = ConversationService()
    metrics_service = MetricsService()
    feedback_service = FeedbackService()

    conversation_stats = None
    if session_id:
        conversation_stats = await conv_service.get_conversation_stats(session_id)

    daily_stats = await conv_service.get_daily_stats(days=days)
    system_health = await metrics_service.get_system_health()
    feedback_stats = await feedback_service.get_feedback_stats(days=days)

    return {
        "status": "success",
        "analytics": {
            "conversation_stats": conversation_stats,
            "daily_stats": daily_stats,
            "system_health": system_health,
            "feedback_stats": feedback_stats,
        },
        "timestamp": datetime.now().isoformat(),
    }
