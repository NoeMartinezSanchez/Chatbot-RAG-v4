"""RAG Generator using Gemma via Ollama.

This module provides a generator based on Gemma 4 models
for improved response quality in the RAG architecture.
"""

import logging
import time
from typing import Optional, Callable

from loguru import logger

from models.ollama_wrapper import OllamaWrapper


class GemmaGenerator:
    """Generator using Gemma 4 via Ollama for RAG-based responses."""

    def __init__(self, cache_dir: str = "models/cache", model: str = "gemma4:e4b"):
        """Initialize the Gemma generator with Ollama.

        Args:
            cache_dir: Directory to cache model files (not used with Ollama).
            model: Ollama model name to use.
        """
        logger.info("Initializing GemmaGenerator with Ollama...")
        start_time = time.time()
        
        self.wrapper = OllamaWrapper(model=model)
        self.model = model
        
        load_time = time.time() - start_time
        logger.success(f"✅ GemmaGenerator initialized in {load_time:.1f}s")

    def generate(
        self,
        query: str,
        context: str = "",
    ) -> str:
        """Generate a response for the given query.

        Args:
            query: User question/query.
            context: Retrieved context from RAG system (optional).

        Returns:
            Generated response string.
        """
        if context and context.strip():
            return self.generate_with_context(
                context=context,
                question=query,
            )
        else:
            return self.wrapper.generate(prompt=query)

    def generate_with_context(
        self,
        context: str,
        question: str,
    ) -> str:
        """Generate a response given context and a question (RAG mode).

        Args:
            context: Retrieved context from the RAG system.
            question: User question.

        Returns:
            Generated response based on the context.
        """
        try:
            # OllamaWrapper solo acepta context y question
            return self.wrapper.generate_with_context(
                context=context,
                question=question,
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lo siento, tuve un problema al generar la respuesta."
