"""Modelos Pydantic para persistencia en MongoDB."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Retorna la fecha/hora actual en UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    """Roles permitidos para los mensajes de conversación."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    """Mensaje individual dentro de una conversación."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    confidence_score: Optional[float] = None
    is_rag: Optional[bool] = None
    sources_used: Optional[List[Dict[str, Any]]] = None


class ConversationCreate(BaseModel):
    """Documento para la colección de conversaciones."""

    conversation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
    context_used: Optional[str] = None
    sources_used: List[Dict[str, Any]] = Field(default_factory=list)
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    is_rag_response: bool = False
    confidence_score: Optional[float] = None
    created_at: datetime = Field(default_factory=_utcnow)


class MetricCreate(BaseModel):
    """Métrica de rendimiento para la colección de métricas."""

    session_id: Optional[str] = None
    endpoint: str
    request_timestamp: datetime = Field(default_factory=_utcnow)
    response_timestamp: Optional[datetime] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    is_rag_response: bool = False
    confidence_score: Optional[float] = None
    cache_hit: bool = False


class FeedbackCreate(BaseModel):
    """Feedback del usuario para la colección de retroalimentación."""

    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_index: Optional[int] = None
    user_rating: Optional[int] = Field(default=None, ge=1, le=5)
    user_comment: Optional[str] = None
    is_correct: Optional[bool] = None
    created_at: datetime = Field(default_factory=_utcnow)


class ConversationDocument(ConversationCreate):
    """Conversación persistida en MongoDB (incluye ID interno)."""

    id: Optional[str] = None
    updated_at: Optional[datetime] = None


class RAGCacheEntry(BaseModel):
    """Entrada de caché de respuestas RAG."""

    query_hash: str
    query: str
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    hit_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utcnow)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None


class LogEntry(BaseModel):
    """Registro de solicitud HTTP para la colección de logs."""

    timestamp: datetime = Field(default_factory=_utcnow)
    method: str
    path: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    body: Optional[str] = None
    session_id: Optional[str] = None
