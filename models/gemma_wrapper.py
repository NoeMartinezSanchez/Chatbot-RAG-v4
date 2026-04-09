"""Gemma-2-2b-it Wrapper for RAG-based chatbot.

This module provides a wrapper around the google/gemma-2-2b-it model for generating
responses in a RAG architecture, optimized for CPU-only inference without quantization.
"""

import gc
import os
import sys
import time
from typing import Optional

import torch
import transformers
from loguru import logger
from requests.adapters import HTTPAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib3.util.retry import Retry


MODEL_VARIANTS = [
    "google/gemma-2-2b-it",
    "google/gemma-1.1-2b-it",
    "google/gemma-2b-it",
]

MIN_TRANSFORMERS_VERSION = "4.40.0"


def _check_transformers_version():
    """Verify transformers version meets minimum requirement."""
    installed = transformers.__version__
    major, minor, _ = installed.split(".")[:3]
    required_major, required_minor = MIN_TRANSFORMERS_VERSION.split(".")[:2]
    
    if (int(major) < int(required_major) or 
        (int(major) == int(required_major) and int(minor) < int(required_minor))):
        logger.warning(
            f"transformers {installed} may be incompatible. "
            f"Recommended: >= {MIN_TRANSFORMERS_VERSION}"
        )
    else:
        logger.info(f"transformers version: {installed} (OK)")


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
        _check_transformers_version()
        
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
        """Load the Gemma model and tokenizer with fallback strategy."""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("HF_TOKEN found in environment variables")
        else:
            logger.warning("HF_TOKEN not found in environment variables")

        loaded = False
        last_error = None
        
        for i, model_variant in enumerate(MODEL_VARIANTS):
            if i > 0:
                logger.info(f"Trying fallback model: {model_variant}")
            
            try:
                logger.info(f"Loading tokenizer from {model_variant}...")
                download_start = time.time()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_variant,
                    cache_dir=self.cache_dir,
                    token=hf_token,
                    trust_remote_code=True,
                )
                logger.info(f"Tokenizer downloaded in {time.time() - download_start:.1f}s")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("Set pad_token = eos_token")

                logger.info(f"Loading model from {model_variant}...")
                logger.info("Model size: ~4-5 GB, may take several minutes...")
                model_start = time.time()
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_variant,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                    token=hf_token,
                    trust_remote_code=True,
                )
                
                model_time = time.time() - model_start
                logger.info(f"Model loaded successfully in {model_time:.1f}s")
                
                self.model_name = model_variant
                loaded = True
                break
                
            except KeyError as e:
                logger.error(f"KeyError loading {model_variant}: {e}")
                last_error = e
                continue
            except TypeError as e:
                if "timeout" in str(e):
                    logger.error(f"timeout parameter not supported in {model_variant}: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"Error loading {model_variant}: {str(e)}")
                last_error = e
                continue

        if not loaded:
            error_msg = (
                f"Failed to load any Gemma variant. "
                f"Last error: {last_error}. "
                f"Tried: {MODEL_VARIANTS}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.model.eval()
        logger.info(f"Model loaded on CPU with float32: {self.model_name}")
        logger.info(f"Model memory footprint: ~5-6 GB RAM")

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

            response = self._clean_response(response)

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
        prompt = self._build_simple_prompt(context, question)

        logger.info(f"RAG generation - Context length: {len(context)}, Question: {question[:50]}...")
        return self.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.0,
        )

    def _build_simple_prompt(self, context: str, question: str) -> str:
        """Build simple prompt - direct and minimal.

        Args:
            context: Retrieved context from RAG.
            question: User question.

        Returns:
            Formatted prompt string.
        """
        question_lower = question.lower().strip()
        
        saludos = ["hola", "buenos días", "buenas tardes", "buenas", "holi", "hello", "hey", "qué tal", "cómo estás"]
        despedidas = ["adiós", "chao", "bye", "hasta luego", "me voy", "nos vemos"]
        gracias = ["gracias", "thank", "agradezco", "muchas gracias"]
        
        if any(s in question_lower for s in saludos):
            user_message = """¡Hola! Soy el asistente virtual de Prepa en Línea SEP. Estoy aquí para ayudarte con tus dudas sobre el programa. ¿Qué necesitas saber?"""
        
        elif any(s in question_lower for s in despedidas):
            user_message = """¡Hasta luego! Éxito en tus estudios. Cuando tengas dudas, vuelve a escribirme."""
        
        elif any(s in question_lower for s in gracias):
            user_message = """¡De nada! Si tienes más dudas sobre Prepa en Línea, con gusto te ayudo."""
        
        else:
            user_message = f"""Eres un asistente de Prepa en Línea SEP. Responde preguntas de estudiantes usando la información del contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Responde de forma clara y útil en español."""

        prompt = f"""<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
"""

        return prompt

    def _clean_response(self, response: str) -> str:
        """Clean and post-process generated response.

        Args:
            response: Raw response from the model.

        Returns:
            Cleaned response.
        """
        import re
        
        lines = response.strip().split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if re.match(r'^---+$', line):
                continue
            clean_lines.append(line)
        
        response = ' '.join(clean_lines)
        
        response = re.sub(r'#\w+', '', response)
        
        response = re.sub(r'\*+', '', response)
        
        response = re.sub(r'[{}\[\]()]', '', response)
        
        response = re.sub(r'\s+', ' ', response).strip()
        
        if response.startswith('Pre ') or response.startswith('El '):
            pass
        elif response.startswith('Res ') or response.startswith('Te '):
            pass
        elif len(response) > 0 and not response[0].isalpha():
            words = response.split()
            if words:
                response = ' '.join(words)
        
        return response

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