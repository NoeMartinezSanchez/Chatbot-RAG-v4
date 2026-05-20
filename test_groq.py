# test_groq.py
import os
from dotenv import load_dotenv
from models.groq_wrapper import GroqWrapper

# Cargar variables del archivo .env
load_dotenv()

print("=" * 50)
print("Probando Groq API...")
print("=" * 50)

# Crear instancia del wrapper
wrapper = GroqWrapper()

# Prueba 1: Pregunta simple (sin contexto)
print("\n📝 Prueba 1: Pregunta simple")
print("-" * 30)
respuesta1 = wrapper.generate("Hola, ¿cómo estás?")
print(f"Respuesta: {respuesta1}")

# Prueba 2: Pregunta con contexto (simulando RAG)
print("\n📝 Prueba 2: Pregunta con contexto")
print("-" * 30)
contexto = "El módulo propedéutico de Prepa en Línea SEP dura 10 días naturales. La calificación mínima aprobatoria es 60 puntos."
pregunta = "¿Cuánto dura el propedéutico y cuál es la calificación mínima?"
respuesta2 = wrapper.generate_with_context(contexto, pregunta)
print(f"Respuesta: {respuesta2}")

print("\n" + "=" * 50)
print("✅ Pruebas completadas")
print("=" * 50)