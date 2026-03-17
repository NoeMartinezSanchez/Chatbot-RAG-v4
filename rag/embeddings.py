import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import logging
from config.settings import settings  # <-- SE AÑADIO ESTA LINEA

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        from config.settings import settings
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Modelos optimizados para español y CPU
        if "MiniLM" in self.model_name:
            # Muy ligero y bueno para español
            self.model = SentenceTransformer(self.model_name)
            self.dimension = 384
        else:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.dimension = 384
            
        logger.info(f"Embedding model loaded: {self.model_name}")
    
    # Cambia TU embeddings.py (línea 23):
    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, show_progress_bar=False)
    
        # SE AÑADE ESTO (NORMALIZACIÓN): ⭐⭐
        # Calcular norma
        norm = np.linalg.norm(embedding)
    
        # Normalizar solo si la norma no es cero
        if norm > 0:
            embedding = embedding / norm

        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
    
        # ⭐⭐ NORMALIZAR TODOS LOS EMBEDDINGS: ⭐⭐
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Evitar división por cero
        norms[norms == 0] = 1
        embeddings = embeddings / norms
    
        return embeddings