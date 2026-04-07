"""Gemma-2-2b-it Wrapper for RAG-based chatbot.

This module provides a wrapper around the google/gemma-2-2b-it model for generating
responses in a RAG architecture, optimized for CPU-only inference without quantization.
"""

import gc
import os
import time
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaWrapper:
    """Wrapper for Gemma-2-2b-it model with RAG integration support.

    This class provides an interface to the Gemma-2-2b-it model optimized for
    CPU-only inference without quantization (float32).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        cache_dir: str = "models/cache",
    ) -> None:
        """Initialize the Gemma wrapper.

        Args:
            model_name: Hugging Face model identifier.
            cache_dir: Directory to cache the model files.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cpu"
        self.model = None
        self.tokenizer = None

        self._setup_logger()
        self._load_model()

    def _setup_logger(self) -> None:
        """Configure logging for the wrapper."""
        logger.add(
            "logs/gemma_wrapper.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
        )

    def _load_model(self) -> None:
        """Load the Gemma model and tokenizer."""
        try:
            logger.info(f"Initializing Gemma model: {self.model_name}")
            logger.info(f"Loading on CPU with float32 (no quantization)")

            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                logger.info("HF_TOKEN found in environment variables")
            else:
                logger.warning("HF_TOKEN not found in environment variables")

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=hf_token,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token = eos_token")

            logger.info("Loading model with device_map='cpu' and torch.float32...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                cache_dir=self.cache_dir,
                token=hf_token,
            )

            self.model.eval()
            logger.info("Model loaded successfully on CPU with float32")
            logger.info(f"Model memory footprint: ~5-6 GB RAM")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: The input prompt string in Gemma chat format.
            max_new_tokens: Maximum number of tokens to generate.
            min_new_tokens: Minimum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
            early_stopping: Whether to stop when reaching end of sentence.
            no_repeat_ngram_size: Prevents repeating n-grams of this size.

        Returns:
            Generated response string (without the prompt).
        """
        start_time = time.time()

        try:
            logger.info(f"Generating response for prompt (length: {len(prompt)})")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=early_stopping,
                )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            response = generated_text[len(prompt):].strip()

            if len(response) < 10:
                logger.warning(f"Response very short ({len(response)} chars)")
                return response.strip() if response.strip() else "No se pudo generar una respuesta."

            elapsed = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            logger.info(
                f"Generated {tokens_generated} tokens in {elapsed:.2f}s "
                f"({tokens_generated/elapsed:.1f} tokens/s)"
            )

            self._clear_cache()

            return response

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            self._clear_cache()
            return "Lo siento, hubo un problema al generar la respuesta. Por favor, intenta de nuevo."

    def generate_with_context(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a response given context and a question (RAG mode).

        Args:
            context: Retrieved context from the RAG system.
            question: User question.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response based on the context.
        """
        if len(context) > 800:
            context = context[:800] + "..."

        prompt = self._build_gemma_prompt(context, question)

        logger.info(f"RAG generation - Context length: {len(context)}, Question: {question[:50]}...")
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            min_new_tokens=30,
        )

    def _build_gemma_prompt(self, context: str, question: str) -> str:
        """Build the prompt in Gemma chat format.

        Args:
            context: Retrieved context from RAG.
            question: User question.

        Returns:
            Formatted prompt string.
        """
        user_message = f"""Información de contexto:
{context}

Pregunta del estudiante:
{question}"""

        prompt = f"""<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
"""

        return prompt

    def _clear_cache(self) -> None:
        """Clear Python and PyTorch garbage and cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Cleared memory cache")

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": "float32",
            "parameters": "2B",
            "quantization": "none",
        }