import requests
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaWrapper:
    def __init__(self, model: str = "gemma4:e4b", host: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.host = host
        self.timeout = timeout
        self._wait_for_ollama()
        logger.info(f"✅ OllamaWrapper initialized with model: {model}")

    def _wait_for_ollama(self, max_retries: int = 30, delay: int = 1) -> None:
        """Esperar a que el servidor de Ollama esté listo"""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.host}/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Ollama server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(delay)
        logger.warning("⚠️ Could not connect to Ollama server after multiple retries")

    def generate_with_context(self, context: str, question: str) -> str:
        """Generar respuesta usando contexto RAG"""
        prompt = f"""Responde la pregunta con UNA SOLA PALABRA o UNA FRASE MUY CORTA.

Pregunta: {question}

Respuesta:"""
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.85,
                        "num_predict": 20
                    }
                },
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            result = response.json()
            answer = result.get("response", "").strip()
            logger.info(f"✅ Generated in {elapsed:.2f}s")
            return answer if answer else "No se pudo generar una respuesta."
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout after {self.timeout}s")
            return "No se pudo generar una respuesta."
        except Exception as e:
            logger.error(f"❌ Error in Ollama: {e}")
            return "No se pudo generar una respuesta."

    def generate(self, prompt: str, **kwargs) -> str:
        """Método para compatibilidad con la interfaz existente"""
        return self.generate_with_context("", prompt)
