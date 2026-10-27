"""Glosario de términos y abreviaturas de Prepa en Línea-SEP.

Dos usos:

- ``expandir_abreviaturas``: reemplaza abreviaturas comunes (PL, PL-SEP, G85,
  etc.) por su forma completa ANTES del retrieval, para mejorar la recuperación.
- ``definir_termino``: responde preguntas del tipo "¿qué significa PL?" o
  "¿qué es el propedéutico?" con una definición breve.

No requiere reindexar FAISS.
"""
import re
from typing import Optional

# Abreviatura (normalizada, sin acentos) -> forma completa.
ABREVIATURAS = {
    "pl": "Prepa en Línea",
    "pls": "Prepa en Línea",
    "plsep": "Prepa en Línea SEP",
    "pl-sep": "Prepa en Línea SEP",
    "prepa en linea sep": "Prepa en Línea SEP",
    "sep": "Secretaría de Educación Pública (SEP)",
    "g85": "Generación 85",
    "g86": "Generación 86",
    "prope": "módulo propedéutico",
    "prop": "módulo propedéutico",
    "mod": "módulo",
    "asesor": "Asesor Virtual",
    "asesor virtual": "Asesor Virtual",
    "ava": "Asesor Virtual",
    "sems": "Subsecretaría de Educación Media Superior (SEMS)",
    "sigaprep": "SIGAPREP (Sistema Integral de Gestión Académica)",
    "same": "SAME (Sistema de Acompañamiento y Mejora Educativa)",
    "dgair": "DGAIR (Dirección General de Acreditación, Incorporación y Revalidación)",
    "cct": "Clave del Centro de Trabajo (CCT)",
    "curp": "CURP (Clave Única de Registro de Población)",
    "ine": "credencial del INE",
    "pdf": "PDF",
    "faq": "preguntas frecuentes",
}

# Término (clave normalizada) -> definición breve.
GLOSARIO = {
    "pl": "PL es la abreviatura de Prepa en Línea-SEP, el servicio educativo gratuito y en línea de nivel medio superior de la SEP.",
    "pl-sep": "PL-SEP significa Prepa en Línea-SEP: bachillerato gratuito, 100% en línea y con validez oficial, ofrecido por la SEP.",
    "prepa en linea": "Prepa en Línea-SEP es un servicio educativo federal gratuito de nivel medio superior, 100% en línea y con validez oficial en todo México.",
    "propedeutico": "El módulo propedéutico es un curso obligatorio de 10 días naturales que te prepara para usar el aula virtual y conocer el modelo educativo. Debes obtener al menos 60 puntos.",
    "modulo": "Un módulo es cada una de las 23 etapas del plan de estudios de Prepa en Línea-SEP; cada uno dura 4 semanas.",
    "asesor virtual": "El Asesor Virtual es la figura docente que te acompaña y califica tus actividades en cada módulo.",
    "tutor": "El tutor es la figura de apoyo que te orienta durante el módulo propedéutico y tus estudios.",
    "sigaprep": "SIGAPREP es el Sistema Integral de Gestión Académica, donde puedes consultar tu estatus escolar y tus créditos.",
    "same": "El SAME (Sistema de Acompañamiento y Mejora Educativa) apoya cuando el Asesor Virtual no está disponible.",
    "dgair": "La DGAIR (Dirección General de Acreditación, Incorporación y Revalidación) valida revalidaciones y duplicados de certificados.",
    "mesa de servicio": "La Mesa de Servicio es el canal oficial para resolver dudas técnicas o administrativas mediante tickets (no sirve para registrarse).",
    "generacion": "La generación es el grupo de aspirantes que ingresa en una misma convocatoria (por ejemplo, Generación 85).",
    "netiqueta": "La netiqueta son las reglas de comunicación virtual (respeto, claridad, no escribir en MAYÚSCULAS, citar fuentes).",
}


def _normalizar(texto: str) -> str:
    t = str(texto).lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ü", "u"), ("ñ", "n")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def expandir_abreviaturas(texto: str) -> str:
    """Expande abreviaturas conocidas en un texto de consulta.

    Sustituye palabras sueltas por su forma completa (p. ej. "PL" -> "Prepa en
    Línea"). Se aplica antes del retrieval para mejorar la recuperación.
    """
    if not texto:
        return texto
    resultado = texto
    # Reemplazo por palabra completa, insensible a mayúsculas/acentos en la clave.
    patrones = [
        ("PL-SEP", "Prepa en Línea SEP"),
        ("PLSEP", "Prepa en Línea SEP"),
        ("PLS", "Prepa en Línea"),
        ("PL", "Prepa en Línea"),
        ("G85", "Generación 85"),
        ("G86", "Generación 86"),
    ]
    for clave, valor in patrones:
        resultado = re.sub(rf"\b{re.escape(clave)}\b", valor, resultado, flags=re.IGNORECASE)
    return resultado


def definir_termino(pregunta: str) -> Optional[str]:
    """Si la pregunta pide el significado de un término del glosario, lo define.

    Detecta patrones como "¿qué significa PL?", "¿qué es el propedéutico?".
    """
    q = _normalizar(pregunta)
    if not re.search(r"(que (es|significa|quiere decir)|a que se refiere|definicion de|defineme)", q):
        return None
    for clave, definicion in GLOSARIO.items():
        if re.search(rf"\b{re.escape(clave)}\b", q):
            return definicion
    # Caso especial: "servicio educativo federal"
    if "federal" in q:
        return ("Se refiere a que el Servicio Nacional de Bachillerato en Línea (Prepa en Línea-SEP) "
                "es un servicio educativo ofrecido por el Gobierno Federal a través de la SEP, "
                "gratuito, con validez oficial y cobertura nacional.")
    return None
