"""Utilidades de identidad y trazabilidad de usuarios/sesiones.

Permiten generar identificadores anónimos estables (UUID por navegador) y
proteger datos personales (PII) con hash SHA-256 truncado antes de persistir
``user_id`` no anónimo.
"""
import hashlib
import uuid

from typing import Optional


def generate_anonymous_id(prefix: str = "anon") -> str:
    """Genera un identificador anónimo estable (UUID v4).

    Args:
        prefix: Prefijo legible (default: ``anon``).

    Returns:
        String tipo ``prefijo-<uuid hex>``.
    """
    return f"{prefix}-{uuid.uuid4().hex}"


def generate_session_id() -> str:
    """Genera un ``session_id`` único para un visitante.

    Returns:
        String tipo ``ses-<uuid hex>``.
    """
    return generate_anonymous_id("ses")


def _looks_like_pii(value: str) -> bool:
    """Detecta heurísticamente si un ``user_id`` parece dato personal.

    Algunos clientes envían correos, CURPs o nombres como ``user_id``; solo
    esos casos se consideran PII. Los UUID anónimos generados por el
    frontend no lo son.

    Args:
        value: ``user_id`` tal como llegó.

    Returns:
        ``True`` si contiene signos típicos de PII.
    """
    if not value:
        return False
    v = value.strip().lower()
    # Correo o CURP (nombre fijo de 18 caracteres) o cadenas con espacios
    if "@" in v:
        return True
    if len(v) == 18 and v.replace("ñ", "n").isalnum():
        return True
    if " " in v and any(c.isalpha() for c in v):
        return True
    return False


def hash_user_id(user_id: Optional[str], force: bool = False) -> Optional[str]:
    """Hashea un ``user_id`` para proteger PII antes de persistir.

    Si ``force`` es ``True`` se hashea siempre; si no, solo cuando el valor
    parece contener PII (ver ``_looks_like_pii``). Devuelve un hash SHA-256
    truncado a 16 caracteres hex para mantener privacidad.

    Args:
        user_id: Identificador del usuario (opcional).
        force: Hash obligatorio aunque no parezca PII.

    Returns:
        Hash truncado o ``None`` si el input es ``None``/vacío.
    """
    if not user_id:
        return None
    if not force and not _looks_like_pii(user_id):
        return user_id
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def get_or_generate_user_id(user_id: Optional[str]) -> str:
    """Retorna el ``user_id`` recibido o genera uno anónimo.

    Args:
        user_id: Identificador provisto por el cliente (opcional).

    Returns:
        El ``user_id`` limpio, o un ID anónimo si estaba vacío.
    """
    if user_id and user_id.strip():
        return user_id.strip()
    return generate_anonymous_id("user")