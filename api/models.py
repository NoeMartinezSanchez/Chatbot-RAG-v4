"""Modelos Pydantic para los endpoints de la API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Solicitud de chat.

    Acepta ``question`` (nuevo schema) o ``message`` (compatibilidad con
    el frontend existente y test_api.py).
    """

    question: str = ""
    session_id: str = "default"
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message: Optional[str] = None  # Backward compat con frontend

    @model_validator(mode="before")
    @classmethod
    def _map_message(cls, values: Any) -> Any:
        """Mapea ``message`` a ``question`` si esta última viene vacía."""
        if isinstance(values, dict):
            if not values.get("question") and values.get("message"):
                values["question"] = values["message"]
        return values


class ChatResponse(BaseModel):
    """Respuesta de chat con metadatos."""

    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    is_rag_response: bool = True
    confidence: float = 0.0
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Solicitud de feedback.

    Compatible con el schema nuevo (``user_rating``, ``message_index``) y con
    el schema legacy (``is_helpful``, ``message_id``, ``feedback_text``).
    """

    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_index: Optional[int] = 0
    user_rating: Optional[int] = Field(default=None, ge=1, le=5)
    user_comment: Optional[str] = None
    is_correct: Optional[bool] = None

    # Legacy (test_api.py)
    message_id: Optional[str] = None
    is_helpful: Optional[bool] = None
    feedback_text: Optional[str] = None


class AnalyticsRequest(BaseModel):
    """Parámetros del endpoint de analíticas."""

    session_id: Optional[str] = None
    days: int = Field(default=7, ge=1, le=90)
