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
            prompt = f"""Contexto oficial:
{context}

Pregunta: {question}

Respuesta (solo con contexto. Si no aparece, di: "No encontré información oficial"):"""
        else:
            prompt = f"""Eres asistente de Prepa en Línea SEP.
Responde clara y amigable: {question}"""
        
        # Mensajes para Groq
        messages = [
            {
                "role": "system",
                "content": "Eres asistente de Prepa en Línea SEP. Respuestas claras en español para estudiantes mexicanos."
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