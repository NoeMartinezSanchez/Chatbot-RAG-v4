"""RAG Generator using TinyLlama.

This module replaces the previous BERT-based generator with TinyLlama
for improved response quality in the RAG architecture.
"""
import logging
import time
from typing import List, Optional

from loguru import logger

from models.tinyllama_wrapper import TinyLlamaWrapper


class TinyLlamaGenerator:
    """Generator using TinyLlama for RAG-based responses.

    This class wraps TinyLlamaWrapper to provide a simple interface
    for generating responses with or without context.
    """

    def __init__(self, use_quantization: bool = False, cache_dir: str = "models/cache"):
        """Initialize the TinyLlama generator.

        Args:
            use_quantization: Whether to use 4-bit quantization.
            cache_dir: Directory to cache model files.
        """
        logger.info("Initializing TinyLlamaGenerator...")
        start_time = time.time()

        try:
            self.wrapper = TinyLlamaWrapper(
                use_quantization=use_quantization,
                cache_dir=cache_dir,
            )
            load_time = time.time() - start_time
            logger.success(f"TinyLlamaGenerator initialized in {load_time:.1f}s")

        except Exception as e:
            logger.error(f"Failed to initialize TinyLlamaGenerator: {e}")
            raise RuntimeError(f"Generator initialization failed: {e}") from e

    def generate(
        self,
        query: str,
        context: str = "",
        max_length: int = 256,
    ) -> str:
        """Generate a response for the given query.

        Args:
            query: User question/query.
            context: Retrieved context from RAG system (optional).
            max_length: Maximum tokens to generate.

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
                    temperature=0.2,
                    top_p=0.9,
                )
            else:
                logger.info(f"Using RAG with context (length: {len(context)})")
                response = self.generate_with_context(
                    context=context,
                    question=query,
                    max_new_tokens=max_length,
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
        max_new_tokens: int = 150,
    ) -> str:
        """Genera respuesta basada en contexto para Prepa en Línea SEP."""
        import re

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

        if len(clean_context) > 1200:
            clean_context = clean_context[:1200] + "..."

        if not clean_context or len(clean_context) < 50:
            return "Lo siento, no encontré información específica sobre eso en los materiales de Prepa en Línea SEP."

        prompt = f"""Eres un asesor académico de Prepa en Línea SEP. Solo conoces información sobre este programa educativo.

CONTEXTO OFICIAL:
{clean_context}

PREGUNTA:
{question}

Responde usando SOLO la información del contexto. Si no hay información suficiente, dilo claramente. Sé breve y directo."""

        logger.info(f"RAG generation - Context: {len(clean_context)} chars, Question: {question[:50]}...")

        try:
            return self.wrapper.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.5,
                top_p=0.95,
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
        import random
        return random.choice(fallback_responses)


class ResponseGenerator(TinyLlamaGenerator):
    """Backward compatibility wrapper.

    This class maintains compatibility with existing code that uses
    ResponseGenerator while internally using TinyLlama.
    """

    pass