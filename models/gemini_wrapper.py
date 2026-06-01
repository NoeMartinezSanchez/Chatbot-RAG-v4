import google.generativeai as genai
import os
import logging

logger = logging.getLogger(__name__)

class GeminiWrapper:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY no configurada. Gemini no estará disponible.")
            self.model = None
            return
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("✅ Gemini API inicializada correctamente")
    
    def generate_with_context(self, context, question, **kwargs):
        if self.model is None:
            return "Gemini no está disponible (GEMINI_API_KEY no configurada)."
        
        prompt = f"""Responde la pregunta usando SOLO el contexto. Responde con UNA frase corta.

CONTEXTO: {context}

PREGUNTA: {question}

RESPUESTA:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error en Gemini API: {e}")
            return "No se pudo generar una respuesta."
    
    def generate(self, prompt, **kwargs):
        """Método para compatibilidad con la interfaz existente.
        Acepta kwargs adicionales (como on_tokens_generated) para no romper la compatibilidad.
        """
        return self.generate_with_context("", prompt)