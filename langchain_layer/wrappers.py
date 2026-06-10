"""LangChain wrappers with REAL memory injection - FIXED VERSION"""
from typing import Dict, Any
from langchain.memory import ConversationBufferMemory
from collections import defaultdict

# Almacenamiento de memorias por sesión
_session_memories = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
))

class LangChainRAGWrapper:
    """Wrapper que inyecta el historial en cada pregunta"""
    
    def __init__(self, rag_system, memory_enabled: bool = True):
        self.rag_system = rag_system
        self.memory_enabled = memory_enabled
        print(f"✅ LangChain wrapper con INYECCIÓN DE MEMORIA activada")
    
    def query_with_memory(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        # Obtener memoria de esta sesión
        memory = _session_memories[session_id]
        
        # 🔥 RECUPERAR el historial guardado
        history_text = ""
        if self.memory_enabled:
            history_vars = memory.load_memory_variables({})
            if "chat_history" in history_vars:
                messages = history_vars["chat_history"]
                # Formatear historial como texto
                history_text = "\n".join([f"- {msg.type}: {msg.content}" for msg in messages])
        
        # 🔥 INYECTAR historial en la pregunta (si existe)
        if history_text:
            enhanced_question = f"""Historial de la conversación:
{history_text}

Pregunta actual: {question}

Responde basándote en el historial si es relevante."""
        else:
            enhanced_question = question
        
        # Llamar al RAG con la pregunta enriquecida
        result = self.rag_system.generate_response(enhanced_question)
        
        # Guardar la interacción en memoria para el futuro
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
            "history_length": len(memory.buffer) if hasattr(memory, 'buffer') else 0,
            "history_injected": bool(history_text)  # Nuevo: indica si se usó historial
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