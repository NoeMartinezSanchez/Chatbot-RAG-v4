"""TinyLlama Wrapper for RAG-based chatbot.

This module provides a wrapper around TinyLlama model for generating
responses in a RAG architecture, replacing the previous BERT-based approach.
"""

import time
from typing import Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class TinyLlamaWrapper:
    """Wrapper for TinyLlama model with RAG integration support.

    This class provides an interface to the TinyLlama-1.1B-Chat model with
    support for 4-bit quantization for memory-efficient inference.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_quantization: bool = True,
        cache_dir: str = "models/cache",
    ) -> None:
        """Initialize the TinyLlama wrapper.

        Args:
            model_name: Hugging Face model identifier.
            use_quantization: Whether to use 4-bit quantization.
            cache_dir: Directory to cache the model files.
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.cache_dir = cache_dir
        self.device: Optional[str] = None
        self.model = None
        self.tokenizer = None

        self._setup_logger()
        self._load_model()

    def _setup_logger(self) -> None:
        """Configure logging for the wrapper."""
        logger.add(
            "logs/tinyllama_wrapper.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
        )

    def _load_model(self) -> None:
        """Load the TinyLlama model and tokenizer."""
        try:
            has_gpu = torch.cuda.is_available()
            self.device = "cuda" if has_gpu else "cpu"

            logger.info(f"Initializing TinyLlama model: {self.model_name}")
            logger.info(f"GPU available: {has_gpu}, Quantization: {self.use_quantization}")

            quantization_config = None
            device_map = "auto" if has_gpu else None

            if has_gpu and self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("4-bit quantization enabled for GPU inference")
            elif not has_gpu:
                logger.info("Loading model on CPU with float32")
                self.use_quantization = False

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token = eos_token")

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16 if has_gpu else torch.float32,
                cache_dir=self.cache_dir,
            )

            logger.info("Model loaded successfully on cpu")
            if quantization_config:
                logger.info("Model loaded with 4-bit quantization")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: The input prompt string.
            max_new_tokens: Maximum number of tokens to generate.
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

            input_device = next(self.model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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

            elapsed = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            logger.info(
                f"Generated {tokens_generated} tokens in {elapsed:.2f}s"
            )

            return response

        except Exception as e:
            self._log_error(f"Generation failed: {str(e)}")
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
        if len(context) > 1500:
            context = context[:1500] + "..."

        prompt = f"""Eres un asistente virtual amable de Prepa en Línea SEP. 
Debes responder SIEMPRE en español, de forma clara y útil.

Contexto relevante:
{context}

Pregunta del estudiante:
{question}

Instrucciones:
- Responde ÚNICAMENTE con la información del contexto proporcionado
- Si la información no está en el contexto, di: "Lo siento, no encontré información específica sobre eso en los materiales disponibles."
- Mantén un tono amable y profesional
- Responde SIEMPRE en español, NUNCA en inglés

Respuesta en español:"""

        logger.info(f"RAG generation - Context length: {len(context)}, Question: {question[:50]}...")
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
        )

    def _log_error(self, error_msg: str) -> None:
        """Log an error message.

        Args:
            error_msg: The error message to log.
        """
        logger.error(error_msg)