"""Ollama Wrapper for RAG-based chatbot.

This module provides a wrapper around the Ollama API to run Gemma 4 locally.
"""

import logging
import time
import requests
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class OllamaWrapper:
    """Wrapper for Ollama API to run Gemma 4 locally."""
    
    def __init__(
        self,
        model: str = "gemma4:e4b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        self._wait_for_ollama()
        logger.info(f"✅ OllamaWrapper initialized with model: {model}")
    
    def _wait_for_ollama(self, max_retries: int = 30):
        """Wait for Ollama server to be ready."""
        logger.info("⏳ Waiting for Ollama server...")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Ollama server is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            logger.info(f"   Attempt {i+1}/{max_retries}...")
            time.sleep(2)
        
        raise RuntimeError("Ollama server did not start in time")
    
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
        
        prompt = f"""Responde la pregunta usando SOLO el contexto. Responde con UNA frase corta.

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
                        "stop": ["=== FIN", "=== CONTEXTO"],
                    }
                },
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"❌ Ollama error: {response.status_code}")
                return "No se pudo generar una respuesta."
            
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            if on_tokens_generated:
                tokens = len(generated_text.split())
                on_tokens_generated(tokens, elapsed)
            
            logger.info(f"✅ Generated in {elapsed:.2f}s")
            
            return generated_text if generated_text else "No se pudo generar una respuesta."
            
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout in Ollama API")
            return "El modelo tardó demasiado en responder."
        except Exception as e:
            logger.error(f"❌ Error in Ollama: {e}")
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
            
            if response.status_code != 200:
                return "Error generating response."
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return "No se pudo generar una respuesta."
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": self.model,
            "host": self.host,
            "type": "ollama",
        }
    
    @staticmethod
    def pull_model(model: str, host: str = "http://localhost:11434") -> bool:
        """Pull/download a model from Ollama."""
        try:
            logger.info(f"📥 Pulling model: {model}")
            
            response = requests.post(
                f"{host}/api/pull",
                json={"name": model},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    logger.info(f"   {data}")
            
            logger.info(f"✅ Model pulled: {model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error pulling model: {e}")
            return False