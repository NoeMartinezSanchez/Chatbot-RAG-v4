# models/groq_wrapper.py
import os
import json
import time
import logging
from datetime import datetime
from groq import Groq

logger = logging.getLogger(__name__)

class GroqWrapper:
    def __init__(self, api_key=None):
        # Leer API key desde variable de entorno o parámetro
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY no encontrada. Configúrala en el archivo .env")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.max_retries = 3
        self.token_counter = 0
        self.token_file = "token_usage.json"
        
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                if data.get('date') == datetime.now().strftime('%Y-%m-%d'):
                    self.token_counter = data.get('tokens', 0)
        except:
            pass
        
        print(f"✅ Groq API inicializada correctamente. Tokens usados hoy: {self.token_counter}")
    
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
                
                # Monitoreo de tokens
                total_tokens = response.usage.total_tokens
                self.token_counter += total_tokens
                
                with open(self.token_file, 'w') as f:
                    json.dump({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'tokens': self.token_counter
                    }, f)
                
                logger.info(f"📊 Esta consulta: {total_tokens} tokens | Total hoy: {self.token_counter} / 100,000 ({self.token_counter/1000:.1f}%)")
                
                if self.token_counter > 80000:
                    logger.warning(f"⚠️ ALERTA: Cerca del límite diario! {self.token_counter}/100,000")
                
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