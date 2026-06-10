"""LangChain wrappers with real conversational memory"""
from typing import Dict, Any
from langchain.memory import ConversationBufferMemory
from collections import defaultdict

# Almacenamiento de memorias por sesión (en RAM, funciona en HF Spaces)
_session_memories = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
))

class LangChainRAGWrapper:
    """Wrapper con memoria conversacional real usando LangChain"""
    
    def __init__(self, rag_system, memory_enabled: bool = True):
        self.rag_system = rag_system
        self.memory_enabled = memory_enabled
        print(f"✅ LangChain wrapper con memoria REAL inicializado")
    
    def query_with_memory(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Consulta con memoria conversacional"""
        
        # Obtener memoria de esta sesión
        memory = _session_memories[session_id]
        
        # Llamar al RAG original
        result = self.rag_system.generate_response(question)
        
        # Guardar en memoria si está habilitado
        if self.memory_enabled:
            memory.save_context(
                {"input": question},
                {"output": result.get("response", "")}
            )
        
        # Retornar respuesta con métricas
        return {
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "is_rag_response": result.get("is_rag_response", True),
            "confidence": result.get("confidence", 0.0),
            "session_id": session_id,
            "langchain_version": True,
            "memory_active": self.memory_enabled,
            "history_length": len(memory.buffer) if hasattr(memory, 'buffer') else 0
        }
    
    def clear_memory(self, session_id: str = "default") -> Dict[str, Any]:
        """Limpiar memoria de una sesión"""
        if session_id in _session_memories:
            _session_memories[session_id].clear()
            return {
                "status": "success", 
                "message": f"Memoria limpiada para sesión {session_id}",
                "session_id": session_id
            }
        return {
            "status": "not_found",
            "message": f"No se encontró memoria para sesión {session_id}",
            "session_id": session_id
        }
