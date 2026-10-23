"""Mensajes estándar y limpieza de respuestas del chatbot.

Centraliza los textos que NO deben depender del modelo generativo:

- ``MENSAJE_FUERA_DOMINIO``: respuesta cordial cuando la consulta no pertenece
  a los documentos oficiales de Prepa en Línea-SEP.
- ``MENSAJE_CRISIS``: respuesta empática + derivación a ayuda ante mensajes
  emocionales o de riesgo.
- ``limpiar_artefactos_respuesta``: retira marcadores de plantilla/citas que no
  deben llegar al usuario (p. ej. ``【CONTROL_ESCOLAR】``, ``(Fuente: ...)``).
"""
import re
from typing import List

MENSAJE_FUERA_DOMINIO = (
    "No encontré esa información en los documentos oficiales de Prepa en Línea-SEP. "
    "Puedo ayudarte con temas de registro y convocatoria, módulos y control escolar, "
    "normas de convivencia y reglas de comunicación virtual. "
    "Si tu consulta es de otro tipo, te recomiendo contactar a la Mesa de Servicio: "
    "https://prepaenlinea.sep.gob.mx/mesadeservicio"
)

MENSAJE_CRISIS = (
    "Lamento mucho que te sientas así, y quiero que sepas que tu bienestar es lo más importante. "
    "No estás solo(a). Te animo a hablar con una persona de tu confianza o a llamar a la "
    "Línea de la Vida (800 911 2000), gratuita y disponible las 24 horas. "
    "Si quieres, también puedo orientarte para encontrar apoyo dentro de la comunidad educativa."
)

# Marcadores de cita/plantilla que no deben mostrarse al usuario.
_PATRONES_ARTEFACTO: List[re.Pattern] = [
    re.compile(r"【[^】]*】"),                                   # 【CONTROL_ESCOLAR】
    re.compile(r"\s*[（(]\s*(?:fuente|secci[oó]n|ver)\b[^)）]*[)）]", re.IGNORECASE),
    re.compile(r"\s*[–-]\s*(?:control_escolar|convocatoria|gu[ií]a|normativa)\b", re.IGNORECASE),
    re.compile(r"[\U0001F4C4\U0001F4CE\u2b50\u2728]+"),          # emojis de cita sueltos
    re.compile(r"\*\*\s*\*\*"),                                  # énfasis vacío
]


def limpiar_artefactos_respuesta(text: str) -> str:
    """Retira marcadores de cita/plantilla de una respuesta generada.

    Args:
        text: Respuesta final del chatbot.

    Returns:
        Texto limpio, con espacios y puntuación normalizados.
    """
    if not text:
        return text
    result = text
    for patron in _PATRONES_ARTEFACTO:
        result = patron.sub("", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:])", r"\1", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
