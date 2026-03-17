from fastapi import APIRouter, UploadFile, File, HTTPException
import json
import tempfile
import os
from typing import List
import logging

from rag.core import RAGSystem
from config.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Instancia global del sistema RAG
rag_system = RAGSystem()

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
        rag_system.add_document(text_content, metadata)
        
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
            rag_system.add_document(doc.content, doc.metadata)
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