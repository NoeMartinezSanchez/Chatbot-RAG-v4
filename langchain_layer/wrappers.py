"""LangChain wrappers with REAL memory injection + temporal awareness + direct responses + MongoDB persistence"""
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain.memory import ConversationBufferMemory
from collections import defaultdict
from langchain_layer.config import langchain_config
from models.groq_wrapper import GroqWrapper
from scripts.extract_dates import DateExtractor
from security.sanitizer import InputSanitizer
from security.monitor import get_monitor
from mongodb.models import MessageRole, ConversationMessage, ConversationCreate, MetricCreate
from mongodb.services import ConversationService, MetricsService

logger = logging.getLogger(__name__)

# Almacenamiento de memorias por sesión
_session_memories = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
))

class LangChainRAGWrapper:
    """Wrapper que inyecta el historial en cada pregunta y persiste en MongoDB"""
    
    def __init__(self, rag_system, memory_enabled: bool = True, mongodb_enabled: bool = True):
        self.rag_system = rag_system
        self.memory_enabled = memory_enabled
        self.mongodb_enabled = mongodb_enabled
        self.date_extractor = DateExtractor()
        self.conv_service = ConversationService()
        self.metrics_service = MetricsService()
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estima el número de tokens basado en ~4 caracteres por token.

        Args:
            text: Texto a estimar.

        Returns:
            Número estimado de tokens (mínimo 1 si hay texto).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    async def _load_history(self, session_id: str, limit: int = 10) -> str:
        """Carga el historial de la sesión desde MongoDB y lo formatea para el prompt.

        Args:
            session_id: ID de la sesión.
            limit: Número máximo de conversaciones a cargar.

        Returns:
            Historial formateado como texto, o "" si no hay historial.
        """
        try:
            conversations = await self.conv_service.get_conversations_by_session(session_id, limit=limit)
            if not conversations:
                return ""
            # Recolectar mensajes del más reciente al más antiguo (las
            # conversaciones ya vienen ordenadas por created_at desc).
            newest_first: List[str] = []
            for conv in conversations:
                for msg in reversed(conv.messages):
                    role = "usuario" if msg.role == MessageRole.USER else "asistente"
                    newest_first.append(f"- {role}: {msg.content}")

            # Recortar al presupuesto de historial (MAX_HISTORY_TOKENS) para
            # no inflar el prompt con toda la sesión acumulada.
            max_chars = langchain_config.MAX_HISTORY_TOKENS * 4
            selected: List[str] = []
            used = 0
            for line in newest_first:
                if used + len(line) > max_chars:
                    break
                selected.append(line)
                used += len(line)

            # Restaurar orden cronológico para el prompt
            return "\n".join(reversed(selected))
        except Exception as e:
            logger.debug("Historial MongoDB no disponible, usando memoria local: %s", e)
            memory = _session_memories[session_id]
            history_vars = memory.load_memory_variables({})
            if "chat_history" in history_vars:
                return "\n".join([f"- {msg.type}: {msg.content}" for msg in history_vars["chat_history"]])
            return ""

    async def _save_to_mongodb(
        self,
        question: str,
        response_text: str,
        session_id: str,
        user_id: Optional[str],
        conversation_id: str,
        is_rag: bool,
        confidence: Any,
        sources: Any,
        latency_ms: float,
    ) -> None:
        """Guarda la conversación y sus métricas en MongoDB (tarea de fondo).

        Args:
            question: Pregunta del usuario.
            response_text: Respuesta generada.
            session_id: ID de sesión.
            user_id: ID de usuario (opcional).
            conversation_id: ID de conversación.
            is_rag: Si la respuesta usó RAG.
            confidence: Confianza de la respuesta.
            sources: Fuentes usadas (lista de dicts).
            latency_ms: Latencia total de la consulta.
        """
        if not self.mongodb_enabled:
            return
        try:
            confidence_value = float(confidence) if confidence is not None else 0.0
            user_tokens = self._estimate_tokens(question)
            assistant_tokens = self._estimate_tokens(response_text)
            total_tokens = user_tokens + assistant_tokens

            messages = [
                ConversationMessage(role=MessageRole.USER, content=question, tokens=user_tokens),
                ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text,
                    tokens=assistant_tokens,
                    latency_ms=latency_ms,
                    confidence_score=confidence_value,
                    is_rag=bool(is_rag),
                    sources_used=list(sources) if isinstance(sources, list) else None,
                ),
            ]

            conv_data = ConversationCreate(
                conversation_id=conversation_id,
                session_id=session_id,
                user_id=user_id,
                messages=messages,
                sources_used=list(sources) if isinstance(sources, list) else [],
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                is_rag_response=bool(is_rag),
                confidence_score=confidence_value,
            )
            await self.conv_service.save_conversation(conv_data)

            metric = MetricCreate(
                session_id=session_id,
                endpoint="/chat",
                latency_ms=latency_ms,
                tokens_used=total_tokens,
                is_rag_response=bool(is_rag),
                confidence_score=confidence_value,
                cache_hit=False,
            )
            await self.metrics_service.record_metric(metric)

            logger.info("💾 Conversación guardada en MongoDB: %s", conversation_id)
        except Exception as e:
            logger.debug("No se pudo guardar en MongoDB (no afecta la respuesta): %s", e)

    async def _schedule_save(self, *args, **kwargs) -> None:
        """Programa el guardado en MongoDB en background sin bloquear la respuesta."""
        try:
            asyncio.create_task(self._save_to_mongodb(*args, **kwargs))
        except RuntimeError:
            # Sin event loop activo: guardar inline
            await self._save_to_mongodb(*args, **kwargs)

    async def query_with_memory(
        self,
        question: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Procesa una consulta con memoria, conciencia temporal y persistencia MongoDB.

        Args:
            question: Pregunta del usuario.
            session_id: ID de la sesión.
            user_id: ID del usuario (opcional).
            conversation_id: ID de conversación (opcional, se genera si no se da).

        Returns:
            Diccionario con la respuesta, fuentes, métricas y metadatos.
        """
        start_time = time.time()

        sanitized = InputSanitizer.sanitize(question)
        monitor = get_monitor()
        for t in sanitized.threats:
            monitor.log_incident(
                threat_type=t.threat_type,
                severity=t.severity,
                snippet=t.snippet,
                session_id=session_id,
                details={"pattern": t.pattern, "position": t.position},
            )

        question = sanitized.cleaned_text

        memory = _session_memories[session_id]

        history_text = await self._load_history(session_id)

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

        conv_id = conversation_id or str(uuid.uuid4())

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

            latency_ms = round((time.time() - start_time) * 1000, 2)
            await self._schedule_save(
                question=question,
                response_text=response_text,
                session_id=session_id,
                user_id=user_id,
                conversation_id=conv_id,
                is_rag=is_rag,
                confidence=confidence,
                sources=sources,
                latency_ms=latency_ms,
            )

            return {
                "response": response_text,
                "sources": sources,
                "is_rag_response": is_rag,
                "confidence": confidence,
                "session_id": session_id,
                "conversation_id": conv_id,
                "user_id": user_id,
                "latency_ms": latency_ms,
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

        latency_ms = round((time.time() - start_time) * 1000, 2)
        await self._schedule_save(
            question=question,
            response_text=response_text,
            session_id=session_id,
            user_id=user_id,
            conversation_id=conv_id,
            is_rag=is_rag,
            confidence=confidence,
            sources=sources,
            latency_ms=latency_ms,
        )

        return {
            "response": response_text,
            "sources": sources,
            "is_rag_response": is_rag,
            "confidence": confidence,
            "session_id": session_id,
            "conversation_id": conv_id,
            "user_id": user_id,
            "latency_ms": latency_ms,
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
