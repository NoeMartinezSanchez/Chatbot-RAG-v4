"""RAG Generator using Gemma (via Ollama or transformers).

This module provides a generator based on Gemma models
for improved response quality in the RAG architecture.
"""

import logging
import random
import re
import time
from typing import Optional, Callable, Tuple

from loguru import logger


# Try to use Ollama, fall back to transformers
try:
    from models.ollama_wrapper import OllamaWrapper
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from models.gemma_wrapper import GemmaWrapper
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class GemmaGenerator:
    """Generator using Gemma for RAG-based responses."""

    def __init__(self, cache_dir: str = "models/cache", use_ollama: bool = False):
        """Initialize the Gemma generator.

        Args:
            cache_dir: Directory to cache model files.
            use_ollama: If True, try to use Ollama first.
        """
        logger.info("Initializing GemmaGenerator...")
        start_time = time.time()
        
        self.wrapper = None
        self.use_ollama = use_ollama
        
        # Try Ollama first if requested
        if use_ollama and OLLAMA_AVAILABLE:
            try:
                self.wrapper = OllamaWrapper(model="gemma4:4b")
                if self.wrapper.is_available():
                    logger.success(f"✅ GemmaGenerator initialized with Ollama in {time.time() - start_time:.1f}s")
                    return
                else:
                    logger.warning("⚠️ Ollama not available, trying transformers...")
            except Exception as e:
                logger.warning(f"⚠️ Ollama not available: {e}")
        
        # Fall back to transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                self.wrapper = GemmaWrapper(cache_dir=cache_dir)
                logger.success(f"✅ GemmaGenerator initialized with Transformers in {time.time() - start_time:.1f}s")
            except Exception as e:
                logger.error(f"Failed to initialize: {e}")
        else:
            logger.error("❌ No generator available (Ollama and Transformers failed)")
            raise RuntimeError(f"Generator initialization failed: {e}") from e

    def generate(
        self,
        query: str,
        context: str = "",
        max_length: int = 256,
        on_tokens_generated: Optional[Callable[[int, float], None]] = None,
    ) -> str:
        """Generate a response for the given query.

        Args:
            query: User question/query.
            context: Retrieved context from RAG system (optional).
            max_length: Maximum tokens to generate.
            on_tokens_generated: Callback(token_count, elapsed_seconds).

        Returns:
            Generated response string.
        """
        start_time = time.time()

        try:
            logger.info(f"Generating response for query (length: {len(query)})")

            if not context or context.strip() == "":
                logger.info("No context provided, using direct generation")
                response = self.wrapper.generate(
                    prompt=query,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    on_tokens_generated=on_tokens_generated,
                )
            else:
                logger.info(f"Using RAG with context (length: {len(context)})")
                response = self.generate_with_context(
                    context=context,
                    question=query,
                    max_new_tokens=max_length,
                    on_tokens_generated=on_tokens_generated,
                )

            elapsed = time.time() - start_time
            logger.info(f"Response generated in {elapsed:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lo siento, tuve un problema al generar la respuesta. Por favor, intenta de nuevo."

    def generate_with_context(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 256,
        on_tokens_generated: Optional[Callable[[int, float], None]] = None,
    ) -> str:
        """Genera respuesta basada en contexto para Prepa en Línea SEP usando Gemma."""
        lines = context.split('\n')
        clean_lines = []
        for line in lines:
            if re.match(r'^\[.*?\]$', line):
                continue
            if re.match(r'^#{2,}', line):
                continue
            if re.match(r'^📄', line):
                continue
            if re.match(r'^Fila:', line):
                continue
            if re.match(r'^Hoja:', line):
                continue
            if line.strip() and len(line.strip()) > 10:
                clean_lines.append(line.strip())

        clean_context = ' '.join(clean_lines)

        if len(clean_context) > 1500:
            clean_context = clean_context[:1500] + "..."

        if not clean_context or len(clean_context) < 50:
            return "Lo siento, no encontré información específica sobre eso en los materiales de Prepa en Línea SEP."

        logger.info(f"RAG generation - Context: {len(clean_context)} chars, Question: {question[:50]}...")

        try:
            return self.wrapper.generate_with_context(
                context=clean_context,
                question=question,
                max_new_tokens=max_new_tokens,
                on_tokens_generated=on_tokens_generated,
            )
        except Exception as e:
            logger.error(f"Error in generate_with_context: {e}")
            return "Lo siento, no encontré información específica sobre eso en los materiales de Prepa en Línea SEP."

    def generate_fallback(self, query: str) -> str:
        """Generate a fallback response when no relevant information is found.

        Args:
            query: The user's query.

        Returns:
            Fallback response string.
        """
        fallback_responses = [
            f"No encontré información específica sobre '{query}' en los materiales disponibles.",
            f"Esa pregunta está fuera del alcance de mi conocimiento actual. ¿Hay algo más en lo que pueda ayudarte?",
            "No tengo información suficiente para responder eso. ¿Podrías reformular tu pregunta?",
        ]
        return random.choice(fallback_responses)


class ResponseGenerator(GemmaGenerator):
    """Backward compatibility wrapper.

    This class maintains compatibility with existing code that uses
    ResponseGenerator while internally using Gemma.
    """
    pass