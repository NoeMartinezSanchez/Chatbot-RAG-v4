import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        
        # Prefijos para e5 (mejora retrieval significativamente)
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "
        
        logger.info(f"Embedding model loaded: {self.model_name}")
        logger.info(f"Using prefixes - query: '{self.query_prefix}', passage: '{self.passage_prefix}'")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with query prefix (for queries)."""
        return self.embed_query(text)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with query prefix."""
        embedding = self.model.encode(
            self.query_prefix + text, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding
    
    def embed_passage(self, text: str) -> np.ndarray:
        """Embed a passage/chunk with passage prefix."""
        embedding = self.model.encode(
            self.passage_prefix + text, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding
    
    def embed_batch(self, texts: List[str], is_passage: bool = False) -> np.ndarray:
        """Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            is_passage: If True, use passage prefix. If False, use query prefix.
        """
        prefix = self.passage_prefix if is_passage else self.query_prefix
        prefixed_texts = [prefix + text for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts, 
            show_progress_bar=False, 
            batch_size=32,
            normalize_embeddings=True
        )
        
        return embeddings
