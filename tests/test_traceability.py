"""Tests de trazabilidad: esquema ampliado de conversaciones y fuentes."""
import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb.models import ConversationCreate, ConversationMessage, MessageRole
from mongodb.services.collection_query_service import (
    CollectionQueryService,
    CONVERSATION_FIELDS,
)
from utils.identifiers import hash_user_id, generate_session_id


def test_conversation_schema_accepts_traceability_fields():
    """ConversationCreate debe persistir los campos nuevos de trazabilidad."""
    conv = ConversationCreate(
        conversation_id="c-1",
        session_id="ses-1",
        user_id="u-1",
        messages=[
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="Respuesta",
                message_id="m-1",
                intent="rag",
                documents_consulted=["bases convocatoria_g85.pdf"],
                chunk_ids=["abc123"],
                doc_types=["convocatoria"],
                tokens_prompt=10,
                tokens_completion=20,
            )
        ],
        documents_consulted=["bases convocatoria_g85.pdf"],
        chunk_ids=["abc123"],
        doc_types=["convocatoria"],
        intent="rag",
        model_used="openai/gpt-oss-120b",
        fecha_actual_sistema="Martes 29 de septiembre de 2026",
        tokens_prompt=10,
        tokens_completion=20,
    )
    data = conv.model_dump()
    assert data["documents_consulted"] == ["bases convocatoria_g85.pdf"]
    assert data["chunk_ids"] == ["abc123"]
    assert data["doc_types"] == ["convocatoria"]
    assert data["model_used"] == "openai/gpt-oss-120b"
    assert data["messages"][0]["message_id"] == "m-1"
    assert data["messages"][0]["tokens_prompt"] == 10


def test_flatten_conversations_includes_new_columns():
    """La vista tipo historial debe incluir las columnas de trazabilidad."""
    raw = [{
        "conversation_id": "c-1",
        "session_id": "ses-1",
        "user_id": "u-1",
        "is_rag_response": True,
        "confidence_score": 0.9,
        "intent": "rag",
        "model_used": "openai/gpt-oss-120b",
        "documents_consulted": ["bases convocatoria_g85.pdf"],
        "chunk_ids": ["abc123"],
        "doc_types": ["convocatoria"],
        "messages": [
            {"role": "user", "content": "¿Cuándo es la convocatoria?", "timestamp": datetime(2026, 8, 10, 10, 0, 0, tzinfo=timezone.utc)},
            {
                "role": "assistant",
                "content": "Del 10 al 20 de Agosto.",
                "message_id": "m-1",
                "latency_ms": 1500.0,
                "tokens": 40,
                "is_rag": True,
                "confidence_score": 0.9,
                "documents_consulted": ["bases convocatoria_g85.pdf"],
                "chunk_ids": ["abc123"],
                "doc_types": ["convocatoria"],
            },
        ],
    }]
    rows = CollectionQueryService._flatten_conversations(raw)
    assert len(rows) == 1
    row = rows[0]
    assert row["conversation_id"] == "c-1"
    assert row["user_id"] == "u-1"
    assert row["session_id"] == "ses-1"
    assert row["message_id"] == "m-1"
    assert row["is_rag_response"] is True
    assert row["documents_consulted"] == ["bases convocatoria_g85.pdf"]
    assert row["chunk_ids"] == ["abc123"]
    assert row["doc_types"] == ["convocatoria"]
    assert row["model_used"] == "openai/gpt-oss-120b"
    # Compatibilidad: las 6 columnas originales se conservan
    for col in ("fecha", "pregunta", "respuesta", "tiempo", "tokens", "rag"):
        assert col in row
    assert "confidence" in row and "intent" in row


def test_conversation_fields_include_all_traza_columns():
    """CONVERSATION_FIELDS debe declarar las columnas nuevas del esquema."""
    for col in ("conversation_id", "message_id", "user_id", "session_id",
                "documents_consulted", "chunk_ids", "doc_types", "confidence",
                "is_rag_response", "intent", "placeholders_resolved",
                "fecha_actual_sistema", "model_used", "tokens_prompt",
                "tokens_completion", "error"):
        assert col in CONVERSATION_FIELDS


def test_hash_user_id_anonymizes_pii():
    """user_id similar a PII debe hashearse (SHA-256 truncado)."""
    hashed = hash_user_id("juan.perez@correo.com")
    assert hashed is not None
    assert len(hashed) == 16
    assert hashed != "juan.perez@correo.com"
    # Determinista
    assert hash_user_id("juan.perez@correo.com") == hashed


def test_hash_user_id_leaves_anonymous_ids():
    """UUID anónimos (tipo anon-<hex>) no son PII y no deben hashearse."""
    uid = generate_session_id()
    assert uid.startswith("ses-")
    assert hash_user_id(uid) == uid


if __name__ == "__main__":
    test_conversation_schema_accepts_traceability_fields()
    test_flatten_conversations_includes_new_columns()
    test_conversation_fields_include_all_traza_columns()
    test_hash_user_id_anonymizes_pii()
    test_hash_user_id_leaves_anonymous_ids()
    print("\n✅ Todos los tests de trazabilidad pasaron!")