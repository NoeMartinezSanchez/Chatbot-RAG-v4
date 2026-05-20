# models/groq_wrapper.py
import os
from groq import Groq
import time

class GroqWrapper:
    def __init__(self, api_key=None):
        # Leer API key desde variable de entorno o parámetro
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY no encontrada. Configúrala en el archivo .env")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.max_retries = 3
        print("✅ Groq API inicializada correctamente")
    
    def generate_with_context(self, context, question, **kwargs):
        """Genera respuesta basada en el contexto recuperado por RAG"""
        
        # Construir el prompt según si hay contexto o no
        if context:
            prompt = f"""Basado en la siguiente información oficial de Prepa en Línea SEP:

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA (usa SOLO la información del contexto. Si no está en el contexto, responde: "No encontré información específica en los materiales oficiales"):"""
        else:
            prompt = f"""Eres un asistente académico de Prepa en Línea SEP.
Responde de manera clara y amigable: {question}"""
        
        # Mensajes para Groq
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente académico de Prepa en Línea SEP. Hablas en español. Das respuestas claras, precisas y útiles para estudiantes mexicanos de bachillerato."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Reintentos automáticos
        for intento in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1024,
                    top_p=1,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"❌ Error en intento {intento+1}: {e}")
                if intento < self.max_retries - 1:
                    time.sleep(2 ** intento)  # Espera: 1, 2, 4 segundos
                else:
                    return "Lo siento, tuve un problema procesando tu pregunta. Por favor intenta de nuevo."
    
    def generate(self, prompt, **kwargs):
        """Método de compatibilidad con la interfaz web"""
        return self.generate_with_context("", prompt)