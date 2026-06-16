"""LangChain wrappers with REAL memory injection + temporal awareness + direct responses"""
from datetime import datetime
from typing import Dict, Any
from langchain.memory import ConversationBufferMemory
from collections import defaultdict
from models.groq_wrapper import GroqWrapper
from scripts.extract_dates import DateExtractor

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
        self.date_extractor = DateExtractor()
        print(f"✅ LangChain wrapper con INYECCIÓN DE MEMORIA activada")
    
    def _agregar_contexto_temporal(self, response_text: str, question: str) -> str:
        palabras_fecha = ["fecha", "plazo", "convocatoria", "registro", "inscripción", "cuándo", "cuando", "resultados"]
        if not any(p in question.lower() for p in palabras_fecha):
            return response_text
        fechas = self.date_extractor.extract_dates(response_text)
        if not fechas:
            return response_text
        fecha_actual = datetime.now().date()
        mensajes = []
        for f in fechas:
            if f.get('tipo') == 'fecha' and 'fecha' in f:
                fecha = datetime.fromisoformat(f['fecha']).date()
                if fecha < fecha_actual:
                    mensajes.append(f"Ya pasó (era el {f['texto_original']})")
                elif fecha == fecha_actual:
                    mensajes.append("¡Es hoy!")
                else:
                    dias = (fecha - fecha_actual).days
                    mensajes.append(f"Faltan {dias} días (es el {f['texto_original']})")
            elif f.get('tipo') == 'rango' and 'fecha_inicio' in f and 'fecha_fin' in f:
                fecha_inicio = datetime.fromisoformat(f['fecha_inicio']).date()
                fecha_fin = datetime.fromisoformat(f['fecha_fin']).date()
                if fecha_fin < fecha_actual:
                    dias = (fecha_actual - fecha_fin).days
                    mensajes.append(f"Ya terminó (hace {dias} días)")
                elif fecha_inicio <= fecha_actual <= fecha_fin:
                    dias = (fecha_fin - fecha_actual).days
                    mensajes.append(f"¡Está vigente! Faltan {dias} días")
                elif fecha_inicio > fecha_actual:
                    dias = (fecha_inicio - fecha_actual).days
                    mensajes.append(f"Comienza en {dias} días")
        if mensajes:
            response_text += f"\n\n📌 **Actualización:** {mensajes[0]}."
        return response_text

    @staticmethod
    def _fecha_actual_es() -> str:
        meses = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ]
        now = datetime.now()
        return f"{now.day} de {meses[now.month - 1]} de {now.year}"

    def query_with_memory(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        memory = _session_memories[session_id]

        history_text = ""
        if self.memory_enabled:
            history_vars = memory.load_memory_variables({})
            if "chat_history" in history_vars:
                messages = history_vars["chat_history"]
                history_text = "\n".join([f"- {msg.type}: {msg.content}" for msg in messages])

        fecha_hoy = self._fecha_actual_es()

        partes = [f"📅 Hoy es {fecha_hoy}."]
        if history_text:
            partes.append(f"Contexto previo: {history_text}")
        partes.append(f"Pregunta: {question}")

        enhanced_question = "\n".join(partes)

        palabras_clave_fecha = ["fecha", "hoy", "día"]
        palabras_clave_saludo = ["hola", "saludos", "buenos días", "buenas tardes", "buenas noches"]
        palabras_clave_presentacion = ["quién eres", "quien eres", "cómo te llamas", "como te llamas"]

        es_general = (
            any(palabra in question.lower() for palabra in palabras_clave_fecha) or
            any(palabra in question.lower() for palabra in palabras_clave_saludo) or
            any(palabra in question.lower() for palabra in palabras_clave_presentacion)
        )

        if "fecha" in question.lower() and "hoy" in question.lower():
            es_general = True

        print(f"🔍 Detección general: question='{question}' → es_general={es_general}")

        if es_general and not self.memory_enabled:
            es_general = False

        if es_general:
            print(f"📢 Pregunta general detectada (sin RAG): {question}")
            llm = GroqWrapper()
            prompt_directo = f"""Eres un asistente educacional de Prepa en Línea SEP.
Hoy es {fecha_hoy}.

Pregunta del usuario: {question}

Responde de manera natural, amigable y breve (máximo 2-3 oraciones).
Si preguntan por la fecha, dila claramente.
Si saludan, saluda cordialmente."""
            response_text = llm.generate(prompt_directo)
            response_text = self._agregar_contexto_temporal(response_text, question)
            is_rag, confidence, sources = False, 0.0, []

            if self.memory_enabled:
                memory.save_context({"input": question}, {"output": response_text})

            return {
                "response": response_text,
                "sources": sources,
                "is_rag_response": is_rag,
                "confidence": confidence,
                "session_id": session_id,
                "langchain_version": True,
                "memory_active": self.memory_enabled,
                "history_length": len(memory.buffer) if hasattr(memory, 'buffer') else 0,
                "history_injected": bool(history_text),
                "current_date": fecha_hoy,
                "direct_response": True
            }

        print(f"🧪 ENHANCED_QUESTION enviada al RAG:\n---\n{enhanced_question}\n---")

        response_text, is_rag, confidence, sources = self.rag_system.process_query(enhanced_question)
        response_text = self._agregar_contexto_temporal(response_text, question)

        if self.memory_enabled:
            memory.save_context(
                {"input": question},
                {"output": response_text}
            )

        return {
            "response": response_text,
            "sources": sources,
            "is_rag_response": is_rag,
            "confidence": confidence,
            "session_id": session_id,
            "langchain_version": True,
            "memory_active": self.memory_enabled,
            "history_length": len(memory.buffer) if hasattr(memory, 'buffer') else 0,
            "history_injected": bool(history_text),
            "current_date": fecha_hoy,
            "direct_response": False
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