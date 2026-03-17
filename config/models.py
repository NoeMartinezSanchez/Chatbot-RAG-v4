from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    is_rag_response: bool = False
    confidence: float = 0.0

class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    is_helpful: bool
    feedback_text: Optional[str] = None

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]