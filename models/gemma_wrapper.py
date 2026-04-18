"""Phi-2 Wrapper for RAG-based chatbot.

This module provides a wrapper around the microsoft/phi-2 model for generating
responses in a RAG architecture, optimized for CPU-only inference.
"""

import gc
import os
import sys
import time
from typing import Optional, Callable

import torch
import transformers
from loguru import logger
from requests.adapters import HTTPAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib3.util.retry import Retry


MODEL_VARIANTS = [
    "google/gemma-2-2b-it",
    "google/gemma-1.1-2b-it",
]

MIN_TRANSFORMERS_VERSION = "4.42.0"


def _check_transformers_version():
    """Verify transformers version meets minimum requirement for Gemma 2."""
    installed = transformers.__version__
    major, minor, _ = installed.split(".")[:3]
    required_major, required_minor = MIN_TRANSFORMERS_VERSION.split(".")[:2]
    
    if (int(major) < int(required_major) or 
        (int(major) == int(required_major) and int(minor) < int(required_minor))):
        logger.warning(
            f"⚠️ transformers {installed} incompatible con gemma-2. "
            f"Requiere >= {MIN_TRANSFORMERS_VERSION}. "
            f"Se usará fallback gemma-1.1-2b-it"
        )
    else:
        logger.info(f"✅ transformers version: {installed} (compatible con gemma-2)")


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
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO CARGA DEL MODELO GEMMA")
        logger.info("=" * 60)
        
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("✅ HF_TOKEN encontrado en variables de entorno")
        else:
            logger.warning("⚠️ HF_TOKEN no encontrado en variables de entorno")
            logger.info("   (El modelo debe ser público o tener HF_TOKEN configurado)")

        loaded = False
        last_error = None
        
        for i, model_variant in enumerate(MODEL_VARIANTS):
            if i > 0:
                logger.info(f"🔄 Intentando fallback: {model_variant}")
            
            try:
                logger.info(f"📥 [1/4] Descargando tokenizer de: {model_variant}")
                download_start = time.time()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_variant,
                    cache_dir=self.cache_dir,
                    token=hf_token,
                    trust_remote_code=True,
                )
                logger.info(f"   ✅ Tokenizer descargado en {time.time() - download_start:.1f}s")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("   ✅ pad_token = eos_token (configurado)")
                else:
                    logger.info(f"   ✅ pad_token ya configurado: {self.tokenizer.pad_token}")

                logger.info(f"📥 [2/4] Descargando modelo: {model_variant}")
                logger.info("   ℹ️ Tamaño: ~4-5 GB, puede tomar varios minutos...")
                model_start = time.time()
                
                logger.info("   ℹ️ Configuración: device_map=cpu, torch_dtype=float32")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_variant,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                    token=hf_token,
                    trust_remote_code=True,
                )
                
                model_time = time.time() - model_start
                logger.info(f"   ✅ Modelo descargado en {model_time:.1f}s")
                
                logger.info("   ℹ️ Ejecutando model.eval()...")
                self.model.eval()
                
                self.model_name = model_variant
                loaded = True
                
                logger.info("=" * 60)
                logger.info(f"✅ MODELO CARGADO EXITOSAMENTE: {self.model_name}")
                logger.info(f"   📍 Device: CPU (float32)")
                logger.info(f"   💾 Memoria aproximada: ~5-6 GB RAM")
                logger.info(f"   ⏱️ Tiempo total: {model_time:.1f}s")
                logger.info("=" * 60)
                break
                
            except KeyError as e:
                logger.error(f"❌ KeyError con {model_variant}: {e}")
                logger.info("   → Intentando siguiente variante...")
                last_error = e
                continue
            except TypeError as e:
                if "timeout" in str(e):
                    logger.error(f"❌ Timeout con {model_variant}: {e}")
                else:
                    logger.error(f"❌ TypeError con {model_variant}: {e}")
                logger.info("   → Intentando siguiente variante...")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"❌ Error cargando {model_variant}: {str(e)}")
                logger.info("   → Intentando siguiente variante...")
                last_error = e
                continue

        if not loaded:
            error_msg = (
                f"❌ Falló la carga de todas las variantes de Gemma. "
                f"Último error: {last_error}. "
                f"Modelos intentados: {MODEL_VARIANTS}. "
                f"Asegúrate de tener transformers>={MIN_TRANSFORMERS_VERSION}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        min_new_tokens: int = 10,
        temperature: float = 0.3,
        top_p: float = 0.85,
        repetition_penalty: float = 1.1,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
        on_tokens_generated: Optional[Callable[[int, float], None]] = None,
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
            on_tokens_generated: Callback to report tokens and elapsed time.

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
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=40,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
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
            response = self.fix_common_errors(response)

            if len(response) < 10:
                logger.warning(f"Response very short ({len(response)} chars)")
                return response.strip() if response.strip() else "No se pudo generar una respuesta."

            elapsed = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            logger.info(
                f"Generated {tokens_generated} tokens in {elapsed:.2f}s "
                f"({tokens_generated/elapsed:.1f} tokens/s)"
            )
            
            if on_tokens_generated:
                on_tokens_generated(tokens_generated, elapsed)
            
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
        max_new_tokens: int = 60,
        on_tokens_generated: Optional[Callable[[int, float], None]] = None,
    ) -> str:
        """Generate a response given context and a question (RAG mode).

        Args:
            context: Retrieved context from the RAG system.
            question: User question.
            max_new_tokens: Maximum tokens to generate.
            on_tokens_generated: Callback(token_count, elapsed_seconds).

        Returns:
            Generated response based on the context.
        """
        prompt = self._build_simple_prompt(context, question)

        logger.info(f"RAG generation - Context length: {len(context)}, Question: {question[:50]}...")
        return self.generate(
            prompt=prompt,
            max_new_tokens=60,
            min_new_tokens=10,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            on_tokens_generated=on_tokens_generated,
        )

    def _build_simple_prompt(self, context: str, question: str) -> str:
        """Build a prompt for Gemma optimized for RAG with short responses."""
        
        prompt = f"""Responde la pregunta con UNA frase corta usando SOLO el contexto.

Contexto:
{context}

Pregunta: {question}

Respuesta (máximo 15 palabras):"""
        
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _clean_response(self, text: str) -> str:
        if not text:
            return text

        import re

        text = re.sub(r'^[^a-zA-ZáéíóúÁÉÍÓÚ¿¡]+', '', text)

        words = text.split()
        if len(words) > 1 and len(words[0]) <= 2:
            text = ' '.join(words[1:])

        text = re.sub(r'\s+', ' ', text).strip()

        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def fix_common_errors(self, text: str) -> str:
        replacements = {
            "constatancia": "constancia",
            "constatancoa": "constancia",
            "secondary": "secundaria",
            "otografía": "fotografía",
            "credenciación": "credencial",
            "cartascompromiso": "carta compromiso",
            "carta compromiso ": "carta compromiso ",
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

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