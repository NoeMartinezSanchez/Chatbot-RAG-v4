"""Detección de mensajes emocionales / de riesgo (crisis).

Muy conservador: solo activa ante frases que indican riesgo o desesperanza
clara, para no reemplazar respuestas normales por el mensaje de ayuda.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Frases que indican crisis o riesgo. Se comparan sin acentos ni mayúsculas.
FRASES_CRISIS = [
    "ya no quiero estar",
    "no quiero estar aquí",
    "no quiero estar aqui",
    "no quiero seguir viviendo",
    "no quiero vivir",
    "quiero desaparecer",
    "quiero morir",
    "me quiero morir",
    "quitarme la vida",
    "acabar con mi vida",
    "no vale la pena vivir",
    "nadie me comprende",
    "nadie me entiende",
    "no le importo a nadie",
    "me siento deprimid",
    "tengo depresion",
    "suicid",
    "no me he sentido bien emocionalmente",
]


def _normalizar(texto: str) -> str:
    """Minúsculas y sin acentos para comparar."""
    texto = str(texto).lower()
    for origen, destino in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ü", "u")):
        texto = texto.replace(origen, destino)
    return re.sub(r"\s+", " ", texto).strip()


def detectar_crisis(texto: str) -> bool:
    """Indica si el texto contiene señales de crisis/riesgo.

    Args:
        texto: Mensaje del usuario.

    Returns:
        ``True`` si se detecta al menos una frase de crisis.
    """
    return motivo_crisis(texto) is not None


def motivo_crisis(texto: str) -> Optional[str]:
    """Devuelve la frase de crisis detectada, o ``None`` si no hay señal."""
    normalizado = _normalizar(texto)
    for frase in FRASES_CRISIS:
        if frase in normalizado:
            logger.warning("🚨 Señal de crisis detectada: '%s'", frase)
            return frase
    return None
