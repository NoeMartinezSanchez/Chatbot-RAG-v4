"""Tests de filtrado de stopwords y normalización para palabras clave."""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stopwords import (
    normalize_keyword,
    filter_keywords,
    STOPWORDS_ES,
)


def test_normalize_keyword_lowercase_and_accents():
    """Debe poner en minúsculas y quitar acentos."""
    assert normalize_keyword("Convocatoria") == "convocatoria"
    assert normalize_keyword("¿Cuándo?") == "cuando"
    assert normalize_keyword("MÓDULO") == "modulo"


def test_normalize_preserves_spaces_for_tokenization():
    """Una oración debe conservar espacios para poder usar split()."""
    assert normalize_keyword("¿Qué documentos necesito?") == "que documentos necesito"


def test_cuando_is_stopword():
    """'cuando' (antes en el Top 5) debe ser stopword normalizada."""
    assert "cuando" in STOPWORDS_ES


def test_tengo_is_stopword():
    """'tengo' (antes en el Top 5) debe ser stopword."""
    assert "tengo" in STOPWORDS_ES


def test_conversational_terms_filtered():
    """Términos conversacionales deben filtrarse del Top."""
    words = ["hola", "gracias", "favor", "buenas", "dias", "tardes", "noches"]
    assert filter_keywords(words) == []


def test_filter_keywords_drops_short_and_stop():
    """Debe descartar palabras cortas y stopwords, conservando las de negocio."""
    result = filter_keywords(
        ["cuando", "tengo", "convocatoria", "documentos", "registro", "certificado"]
    )
    assert "cuando" not in result
    assert "tengo" not in result
    assert result == ["convocatoria", "documentos", "registro", "certificado"]


def test_filter_keywords_min_length():
    """Debe descartar palabras menores a min_length (default 3)."""
    assert filter_keywords(["de", "la", "el", "a"]) == []
    assert filter_keywords(["si", "no", "es"]) == []


if __name__ == "__main__":
    test_normalize_keyword_lowercase_and_accents()
    test_normalize_preserves_spaces_for_tokenization()
    test_cuando_is_stopword()
    test_tengo_is_stopword()
    test_conversational_terms_filtered()
    test_filter_keywords_drops_short_and_stop()
    test_filter_keywords_min_length()
    print("\n✅ Todos los tests de stopwords pasaron!")