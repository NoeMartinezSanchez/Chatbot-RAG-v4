"""Stopwords en español y utilidades de normalización de palabras clave.

Módulo reutilizable para filtrado semántico de palabras clave en el
dashboard y cualquier análisis de texto. Las comparaciones se hacen SIEMPRE
con texto normalizado (minúsculas, sin acentos, sin puntuación) para evitar
que variantes como ``cuándo``/``cuando`` no se filtren (bug conocido del
Top 5 original).
"""
import re

from typing import List, Set


# Acentos ignorados al normalizar (comparación insensible a tildes)
_ACCENTS = str.maketrans(
    "áàäâãéèëêíìïîóòöôõúùüûñç",
    "aaaaaeeeeiiiiooooouuuunc",
)


def _no_accent(text: str) -> str:
    """Elimina tildes y diacríticos de un texto.

    Args:
        text: Texto de entrada.

    Returns:
        Texto sin acentos (lowercase duty del llamador).
    """
    return text.translate(_ACCENTS)


def normalize_keyword(word: str) -> str:
    """Normaliza una palabra para comparación/frecuencia.

    Aplica lowercase, eliminación de acentos y de puntuación. Retorna la
    palabra limpia o cadena vacía si no es válida.

    Args:
        word: Palabra cruda (con mayúsculas, acentos, puntuación).

    Returns:
        Palabra normalizada (minúsculas, sin acentos, sin puntuación).
    """
    if not word:
        return ""
    clean = word.lower()
    clean = _no_accent(clean)
    # Conservar espacios para permitir tokenización con split() en oraciones.
    clean = re.sub(r"[^a-z0-9ñ\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_stopword(word: str) -> str:
    """Normaliza un stopword de la misma forma que ``normalize_keyword``.

    Se mantienen ambas funciones por claridad de propósito, pero comparten
    el mismo pipeline de normalización.

    Args:
        word: Stopword crudo.

    Returns:
        Stopword normalizado.
    """
    return normalize_keyword(word)


# Set base de stopwords en español (frecuentes, sin acentos — ya normalizado).
_BASE_ES: Set[str] = {
    # Artículos, pronombres, preposiciones, conjunciones
    "de", "la", "el", "en", "y", "a", "que", "es", "por", "con", "los",
    "las", "un", "una", "se", "su", "para", "mi", "me", "como", "mas",
    "no", "pero", "del", "al", "le", "les", "esto", "esta", "este", "estos",
    "si", "ya", "o", "u", "ni", "lo", "te", "tu", "sus", "nos", "os",
    "cual", "cuales", "que", "quien", "quienes", "donde", "cuando", "cuanto",
    "como", "cuales", "debe", "deben", "puede", "pueden", "puedo", "poder",
    "hacer", "tener", "ser", "estar", "hay", "haber", "tiene", "tienen",
    "sea", "sean", "son", "era", "era", "fue", "fue", "ha", "han", "he",
    "hemos", "habia", "habian", "hubo", "soy", "eres", "es", "somos", "son",
    "estoy", "estas", "esta", "estamos", "estan", "estaba", "estaban",
    "del", "al", "desde", "hasta", "entre", "sobre", "contra", "antes",
    "despues", "durante", "mediante", "segun", "sin", "con", "por", "para",
    "todo", "toda", "todos", "todas", "estos", "estas", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "muy", "mucho", "muchos",
    "mucha", "muchas", "poco", "pocos", "poca", "pocas", "varios", "varias",
    "algun", "algunos", "alguna", "algunas", "ningun", "ninguno", "ninguna",
    "cada", "cualquier", "cualesquiera", "otro", "otros", "otra", "otras",
    "mismo", "mismos", "misma", "mismas", "nuevo", "nuevos", "nueva",
    "nuevas", "bien", "mal", "solo", "solo", "tambien", "tampoco", "ademas",
    "entonces", "luego", "despues", "ahora", "allí", "aqui", "ahi",
    "asi", "tal", "tales", "cuan", "cuanto", "cualquier", "quien", "quienes",
    "cosa", "cosas", "tipo", "forma", "manera", "razon", "parte", "caso",
    "vez", "veces", "dia", "dias", "hora", "horas", "mes", "año", "tiempo",
    "algo", "alguien", "nada", "nadie", "algun", "ningun", "todo", "nada",
}

# Términos conversacionales y del dominio del chatbot (ruido semántico).
# Incluye saludos, cortesías, fórmulas y palabras casi vacías que NO aportan
# contenido de negocio en el Top N de palabras clave.
_DOMAIN_ES: Set[str] = {
    "hola", "buenas", "buenos", "dias", "tardes", "noches", "saludos",
    "gracias", "agradezco", "agradecer", "muchas", "favor", "porfavor",
    "disculpa", "disculpe", "perdon", "ayuda", "ayudar", "ayudame",
    "consulta", "pregunta", "duda", "quiero", "necesito", "podrias",
    "podria", "puedo", "quisiera", "estas", "estas", "eres", "llamas",
    "llamar", "nombre", "cual", "cuales", "como", "cuando", "donde",
    "porque", "que", "cual", "semana", "quiero", "me", "mi", "mis",
    "responder", "respuesta", "contestar", "decir", "dime", "digan",
    "explicame", "explica", "informacion", "informar", "inf", "n",
    "prepa", "linea", "en", "linea", "pls", "sep", "mexico", "mex",
}

# Stopwords del dominio particular que ya estaban en el Top 5 y deben
# excluirse (ruido conversacional, ver AGENTS.md).
_CONVERSATION_NOISE: Set[str] = {
    "cuando", "tengo", "puedo", "estas", "eres", "quien", "como", "cual",
    "hola", "gracias", "favor", "buenas", "dias", "tardes", "noches",
    "quiero", "necesito", "pregunta", "duda",
}


def _build_stopwords() -> Set[str]:
    """Compone el set completo de stopwords ya normalizado.

    Returns:
        Set con todas las stopwords en su forma normalizada.
    """
    combined: Set[str] = set()
    for source in (_BASE_ES, _DOMAIN_ES, _CONVERSATION_NOISE):
        for w in source:
            norm = normalize_stopword(w)
            if norm:
                combined.add(norm)
    return combined


# Set global de stopwords del proyecto.
STOPWORDS_ES: Set[str] = _build_stopwords()


def filter_keywords(words: List[str], min_length: int = 3) -> List[str]:
    """Filtra una lista de palabras crudas contra los stopwords.

    Cada palabra se normaliza y se descarta si: (1) su forma normalizada
    está en los stopwords, o (2) su longitud es menor a ``min_length``.

    Args:
        words: Lista de palabras crudas (tal como salen del split).
        min_length: Longitud mínima de la forma normalizada (default: 3).

    Returns:
        Lista de palabras NORMALIZADAS que pasaron el filtro.
    """
    result: List[str] = []
    for w in words:
        norm = normalize_keyword(w)
        if not norm:
            continue
        if len(norm) < min_length:
            continue
        if norm in STOPWORDS_ES:
            continue
        result.append(norm)
    return result