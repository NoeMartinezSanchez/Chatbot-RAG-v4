import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.core import RAGSystem
import pytest

def test_rag_initialization():
    """Test inicialización del sistema RAG"""
    rag = RAGSystem()
    assert rag.embedder is not None
    assert rag.vector_store is not None
    assert rag.generator is not None
    print("✓ RAG system initialization test passed")

def test_intent_matching():
    """Test matching de intents"""
    rag = RAGSystem()
    rag.load_intents("data/intents.json")
    
    # Test saludo
    response, is_rag, confidence, sources = rag.process_query("hola buenas tardes")
    assert not is_rag  # Debe ser intent, no RAG
    assert "Hola" in response or "Buen día" in response
    print("✓ Intent matching test passed")

def test_rag_response():
    """Test respuesta RAG"""
    rag = RAGSystem()
    
    # Añadir documento de prueba
    test_doc = """
    El módulo propedéutico tiene una duración de 6 semanas.
    Cada semana cubre un tema diferente: matemáticas, física, química, etc.
    Las evaluaciones son semanales y se publican los viernes.
    """
    
    rag.add_document(test_doc, {
        "title": "Información general del módulo",
        "module": "general",
        "week": "0"
    })
    
    # Consulta relacionada
    response, is_rag, confidence, sources = rag.process_query(
        "¿Cuánto dura el módulo propedéutico?"
    )
    
    assert is_rag  # Debe usar RAG
    assert "6" in response or "seis" in response
    print("✓ RAG response test passed")

def test_fallback_response():
    """Test respuesta de fallback"""
    rag = RAGSystem()
    
    # Consulta fuera de contexto
    response, is_rag, confidence, sources = rag.process_query(
        "¿Quién ganó el mundial de fútbol en 2022?"
    )
    
    # Debe dar una respuesta de fallback
    assert "No encontré" in response or "fuera del alcance" in response
    print("✓ Fallback response test passed")

if __name__ == "__main__":
    test_rag_initialization()
    test_intent_matching()
    test_rag_response()
    test_fallback_response()
    print("\n✅ Todos los tests pasaron correctamente!")