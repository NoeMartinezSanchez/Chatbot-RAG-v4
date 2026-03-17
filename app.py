# app.py - Punto de entrada para Hugging Face Spaces
import sys
import os

# Asegurar que podemos importar los módulos
sys.path.insert(0, os.path.dirname(__file__))

# Importar la app FastAPI
from api.main import app

# La app se exporta para que HF Spaces la detecte