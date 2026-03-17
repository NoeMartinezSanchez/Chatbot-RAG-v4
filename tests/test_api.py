import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
import json

client = TestClient(app)

def test_root_endpoint():
    """Test endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    print("✓ Root endpoint test passed")

def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health endpoint test passed")

def test_chat_endpoint():
    """Test chat endpoint"""
    chat_data = {
        "message": "hola, tengo una duda",
        "user_id": "test_user_123"
    }
    
    response = client.post("/chat", json=chat_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0
    
    # Verificar headers
    assert "X-User-ID" in response.headers
    assert "X-Conversation-ID" in response.headers
    
    print("✓ Chat endpoint test passed")

def test_feedback_endpoint():
    """Test feedback endpoint"""
    feedback_data = {
        "conversation_id": "test_conv_123",
        "message_id": "test_msg_456",
        "is_helpful": True,
        "feedback_text": "Muy útil la respuesta"
    }
    
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    print("✓ Feedback endpoint test passed")

if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_chat_endpoint()
    test_feedback_endpoint()
    print("\n✅ Todos los tests de API pasaron correctamente!")