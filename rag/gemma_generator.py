"""RAG Generator using Gemma via Ollama.

This module provides a generator based on Gemma 4 models
for improved response quality in the RAG architecture.
"""

import logging
import time
from typing import Optional, Callable

from loguru import logger

from models.gemini_wrapper import GeminiWrapper


class GemmaGenerator:
    def __init__(self, cache_dir: str = "models/cache", model: str = "gemini-2.5-flash"):
        logger.info("Initializing GemmaGenerator with Gemini API...")
        start_time = time.time()
        
        self.wrapper = GeminiWrapper()
        self.model = model
        
        load_time = time.time() - start_time
        logger.success(f"✅ GemmaGenerator initialized in {load_time:.1f}s")

    def generate(self, query: str, context: str = "", **kwargs) -> str:
        """Generate a response for the given query.

        Args:
            query: User question/query.
            context: Retrieved context from RAG system (optional).
            **kwargs: Additional arguments (like on_tokens_generated).

        Returns:
            Generated response string.
        """
        logger.info(f"📝 generate() called with kwargs: {list(kwargs.keys())}")
        logger.info(f"📝 generate() - query length: {len(query)}, context length: {len(context)}")
        
        if context and context.strip():
            return self.generate_with_context(
                context=context,
                question=query,
                **kwargs
            )
        else:
            return self.wrapper.generate(prompt=query, **kwargs)

    def generate_with_context(self, context: str, question: str, **kwargs) -> str:
        """Generate a response given context and a question (RAG mode).

        Args:
            context: Retrieved context from the RAG system.
            question: User question.
            **kwargs: Additional arguments ignored for compatibility.

        Returns:
            Generated response based on the context.
        """
        logger.info(f"📝 generate_with_context() - Context length: {len(context)}, Question: {question[:50]}...")
        if kwargs:
            logger.warning(f"   Additional kwargs ignored: {list(kwargs.keys())}")
        
        try:
            response = self.wrapper.generate_with_context(context, question)
            logger.info(f"✅ Response generated: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"❌ Error in generate_with_context: {e}", exc_info=True)
            return "Lo siento, tuve un problema al generar la respuesta."
