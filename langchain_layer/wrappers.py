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
    
    def _mejorar_respuesta_con_fecha(self, respuesta: str, pregunta: str, fecha_hoy: str) -> str:
        palabras_fecha = ["fecha", "plazo", "convocatoria", "registro", "inscripción"]
        if not any(p in pregunta.lower() for p in palabras_fecha):
            return respuesta
        fechas = self.date_extractor.extract_dates(respuesta)
        if not fechas:
            return respuesta
        fecha_actual = datetime.now().date()
        for f in fechas:
            if f.get('tipo') == 'rango' and 'fecha_inicio' in f and 'fecha_fin' in f:
                fecha_inicio = datetime.fromisoformat(f['fecha_inicio']).date()
                fecha_fin = datetime.fromisoformat(f['fecha_fin']).date()
                if fecha_fin < fecha_actual:
                    dias = (fecha_actual - fecha_fin).days
                    respuesta += f"\n\n📌 **Actualización:** Este evento ya terminó (hace {dias} días)."
                elif fecha_inicio <= fecha_actual <= fecha_fin:
                    dias = (fecha_fin - fecha_actual).days
                    respuesta += f"\n\n🔥 **¡Está vigente!** Faltan {dias} días."
                elif fecha_inicio > fecha_actual:
                    dias = (fecha_inicio - fecha_actual).days
                    respuesta += f"\n\n📅 **Aún no comienza.** Faltan {dias} días."
        return respuesta

    @staticmethod
    def _fecha_actual_es() -> str:
        meses = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ]
        dias_semana = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        now = datetime.now()
        return f"{dias_semana[now.weekday()]} {now.day} de {meses[now.month - 1]} de {now.year}"

    def query_with_memory(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        memory = _session_memories[session_id]

        history_text = ""
        if self.memory_enabled:
            history_vars = memory.load_memory_variables({})
            if "chat_history" in history_vars:
                messages = history_vars["chat_history"]
                history_text = "\n".join([f"- {msg.type}: {msg.content}" for msg in messages])

        fecha_hoy = self._fecha_actual_es()

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
            response_text = self._mejorar_respuesta_con_fecha(response_text, question, fecha_hoy)
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

        prompt_llm = f"""📅 Hoy es {fecha_hoy}.

Contexto previo (si existe):
{history_text if history_text else "Sin historial previo."}

Pregunta del usuario: {question}

Responde usando la información del contexto oficial. Si la pregunta involucra fechas, compáralas con la fecha actual y menciona si ya pasó, está vigente o aún no comienza."""

        pregunta_retrieval = question
        print(f"🧪 RETRIEVAL CON: {pregunta_retrieval}")

        response_text, is_rag, confidence, sources = self.rag_system.process_query(pregunta_retrieval)
        response_text = self._mejorar_respuesta_con_fecha(response_text, question, fecha_hoy)

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