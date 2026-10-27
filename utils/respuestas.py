"""Mensajes estándar y limpieza de respuestas del chatbot.

Centraliza los textos que NO deben depender del modelo generativo. Incluye los
mensajes propuestos por el equipo (evaluación de Ma. Fernanda):

- Variantes de ``MENSAJE_FUERA_DOMINIO`` con lenguaje más empático.
- ``MENSAJE_RIESGO_SISTEMA``: rechazo amable a intentos de jailbreak.
- ``MENSAJE_AREA_OPORTUNIDAD``: no se encontró información; invita a reformular.
- ``MENSAJE_REQUIERE_OTRA_AREA``: temas que corresponden a otra área (carreras).
- ``MENSAJE_CRISIS``: respuesta empática + derivación (Línea de la Vida / WhatsApp).
- ``limpiar_artefactos_respuesta``: retira marcadores internos/citas.
"""
import random
import re
from typing import List

# --- Fuera de dominio: variantes con tono empático (se rotan) ---
MENSAJE_FUERA_DOMINIO_VARIANTES = [
    "¡Hola! Por el momento todavía estoy aprendiendo y solo puedo ayudarte con dudas sobre la "
    "convocatoria, el registro, el curso propedéutico o las inscripciones para nuevos aspirantes. "
    "No logré encontrar información exacta sobre lo que me preguntas, pero te invito a revisar "
    "nuestro menú de opciones o a intentar con otra pregunta relacionada con los temas anteriores. "
    "¡Estoy aquí para guiarte en tu ingreso!",
    "¡Ups! Esa pregunta se sale un poco de mi área de conocimiento actual. Por ahora, mi especialidad "
    "es guiarte en todo el proceso de admisión e inscripción (convocatoria, registro, propedéutico y "
    "dudas generales del servicio). Te sugiero elegir alguna opción del menú principal o escribir una "
    "nueva pregunta sobre estos temas para poder ayudarte mejor.",
    "Agradezco tu consulta. Actualmente este canal está habilitado para resolver dudas generales del "
    "servicio y guiar a los nuevos aspirantes en sus etapas de convocatoria, registro, curso "
    "propedéutico e inscripción. Lamento no poder responder a tu solicitud específica en este momento; "
    "te invito a consultar el menú de temas disponibles para continuar con tu proceso.",
    "¡Me encantaría ayudarte con eso, pero aún no tengo esa información en mi sistema! Como soy un "
    "asistente en entrenamiento, solo conozco los detalles de los procesos de admisión, registro, "
    "propedéutico e inscripción. Si tu duda tiene que ver con alguno de estos pasos, intenta "
    "reescribirla con otras palabras o navega con los botones del menú.",
]

# Se mantiene por compatibilidad (se usa en el prompt del generador).
MENSAJE_FUERA_DOMINIO = MENSAJE_FUERA_DOMINIO_VARIANTES[0]


def obtener_mensaje_fuera_dominio() -> str:
    """Devuelve una variante aleatoria del mensaje de fuera de dominio."""
    return random.choice(MENSAJE_FUERA_DOMINIO_VARIANTES)


MENSAJE_RIESGO_SISTEMA = (
    "No puedo compartir detalles sobre el funcionamiento interno, código o arquitectura de este "
    "sistema. Mi única función es orientarte sobre trámites, convocatorias y dudas académicas de "
    "Prepa en Línea-SEP. Por favor, escribe tu duda sobre tu proceso o estudios."
)

MENSAJE_AREA_OPORTUNIDAD = (
    "No logré encontrar información sobre este tema. Para ayudarte mejor, intenta hacer tu pregunta "
    "con otras palabras o de forma más directa. ¡Gracias!"
)

MENSAJE_REQUIERE_OTRA_AREA = (
    "Prepa en Línea-SEP solo imparte el nivel bachillerato y no ofrece carreras universitarias. "
    "Para dudas sobre tu egreso, certificado o canalización con áreas institucionales, te pido "
    "generar un ticket en la Mesa de Servicio (apartado Estudiantes, tema Control Escolar): "
    "https://prepaenlinea.sep.gob.mx/mesadeservicio  "
    "Además, puedes conocer la oferta educativa en las Ferias Vocacionales que se difunden por los "
    "canales oficiales de la SEP."
)

NOMBRE_ASISTENTE = "Avi"

IDENTIDAD_AVI = (
    "¡Hola! Soy **Avi**, el asistente virtual de Prepa en Línea-SEP. Puedo ayudarte con dudas sobre "
    "la convocatoria, el registro, el curso propedéutico, la inscripción, los módulos y trámites "
    "académicos. ¿En qué te ayudo?"
)

CAPACIDADES_AVI = (
    "Soy **Avi** y puedo ayudarte con:\n"
    "- Convocatoria y registro\n"
    "- Curso propedéutico\n"
    "- Inscripción y documentación\n"
    "- Módulos y control escolar\n"
    "- Normas de convivencia y reglas de comunicación virtual\n"
    "También puedo definirte términos (por ejemplo, PL = Prepa en Línea). ¿Qué necesitas?"
)

SMALLTALK_AVI = (
    "¡Hola! Estoy muy bien y listo para ayudarte. Cuéntame, ¿tienes alguna duda sobre tu registro, "
    "el curso propedéutico o tu inscripción en Prepa en Línea-SEP?"
)

MENSAJE_CRISIS = (
    "Lamento mucho que te sientas así, y quiero que sepas que tu bienestar es lo más importante. "
    "No estás solo(a). Para recibir apoyo emocional gratuito, confidencial y especializado las 24 "
    "horas, puedes llamar a la Línea de la Vida al 800 911 2000 o enviar un mensaje por WhatsApp al "
    "55 7055 5555. Si quieres, también puedo orientarte para encontrar apoyo dentro de la comunidad "
    "educativa."
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
