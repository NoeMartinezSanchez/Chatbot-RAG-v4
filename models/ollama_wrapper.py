import requests
import json
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class OllamaWrapper:
    """Wrapper for Ollama API to run Gemma 4 locally."""
    
    def __init__(
        self,
        model: str = "gemma4:4b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        logger.info(f"✅ OllamaWrapper inicializado con modelo: {model}")
        logger.info(f"   Host: {host}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_with_context(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 100,
        temperature: float = 0.2,
        on_tokens_generated: Optional[Callable[[int, float], None]] = None,
    ) -> str:
        """Generate a response given context and question (RAG mode)."""
        
        prompt = f"""Responde la pregunta usando SOLO el contextext. Responde con UNA frase corta. Empieza con "Sí" o "No" si es pregunta de sí/no.

=== CONTEXTO ===
{context}
=== FIN CONTEXTO ===

PREGUNTA: {question}

REGLAS:
1. Usa SOLO la información del contexto
2. Responde con UNA frase corta
3. Empieza con "Sí" o "No" si es pregunta de sí/no
4. Si no hay respuesta en el contexto, di: "No encontré esa información"

RESPUESTA:"""
        
        try:
            import time
            start_time = time.time()
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.85,
                        "num_predict": max_new_tokens,
                        "repeat_penalty": 1.1,
                        "stop Sequences": ["=== FIN", "=== CONTEXTO"],
                    }
                },
                timeout=60
            )
            
            elapsed = time.time() - start_time
            result = response.json()
            
            generated_text = result.get("response", "").strip()
            
            if on_tokens_generated:
                # Estimate token count (rough approximation)
                tokens = len(generated_text.split())
                on_tokens_generated(tokens, elapsed)
            
            logger.info(f"✅ Respuesta generada en {elapsed:.2f}s")
            
            return generated_text if generated_text else "No se pudo generar una respuesta."
            
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout en Ollama API")
            return "El modelo tardó demasiado en responder. Intenta de nuevo."
        except Exception as e:
            logger.error(f"❌ Error en Ollama: {e}")
            return "No se pudo generar una respuesta."
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """Generate a response from a prompt (direct mode)."""
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.85,
                        "num_predict": max_new_tokens,
                    }
                },
                timeout=60
            )
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Error en generate: {e}")
            return "No se pudo generar una respuesta."
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": self.model,
            "host": self.host,
            "type": "ollama",
        }
    
    @staticmethod
    def pull_model(model: str = "gemma4:4b", host: str = "http://localhost:11434") -> bool:
        """Pull/download a model from Ollama."""
        try:
            logger.info(f"📥 Descargando modelo: {model}")
            response = requests.post(
                f"{host}/api/pull",
                json={"name": model},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    logger.info(f"   {status}")
                    
                    if data.get("completed", False):
                        logger.info(f"✅ Modelo descargado: {model}")
                        return True
            
            return True
        except Exception as e:
            logger.error(f"❌ Error descargando modelo: {e}")
            return False