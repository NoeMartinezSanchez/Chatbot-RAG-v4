"""Verifica que las variables de entorno necesarias están configuradas.

Diseñado para ejecutarse en Hugging Face Spaces (o cualquier despliegue)
para confirmar que todas las variables requeridas por el sistema están
presentes antes de iniciar la aplicación.

Uso:
    python scripts/verify_hf_spaces.py

Exit codes:
    0  — Todas las variables obligatorias están configuradas
    1  — Falta al menos una variable obligatoria
"""
import os
import sys
from typing import Dict, List

# Windows usa cp1252 por defecto y no puede imprimir emojis (✅, ❌, etc.).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Obligatorias: sin estas el sistema no puede funcionar.
REQUIRED_VARS: Dict[str, str] = {
    "GROQ_API_KEY": "API key de Groq (GPT OSS 120B) para generación de respuestas",
    "MONGODB_URI": "URI de conexión a MongoDB Atlas (mongodb+srv://...)",
    "MONGODB_DB_NAME": "Nombre de la base de datos (default: chatbot_rag_db)",
    "TIMEZONE": "Zona horaria para fechas (ej: America/Mexico_City)",
}

# Recomendadas: deberían estar configuradas, pero el sistema degrada con grace.
RECOMMENDED_VARS: Dict[str, str] = {
    "ADMIN_API_KEY": "Protege los endpoints /admin/cleanup y /admin/dashboard-report",
    "RAG_CACHE_ENABLED": "Habilita/deshabilita el caché de respuestas RAG",
    "RAG_CACHE_TTL_HOURS": "TTL del caché de respuestas en horas",
    "RETENTION_CONVERSATIONS_DAYS": "Días a conservar conversaciones",
    "RETENTION_METRICS_DAYS": "Días a conservar métricas",
    "RETENTION_LOGS_DAYS": "Días a conservar logs",
    "RETENTION_FEEDBACK_DAYS": "Días a conservar feedback",
}

# Opcionales: solo se usan si el despliegue las necesita.
OPTIONAL_VARS: Dict[str, str] = {
    "ENVIRONMENT": "development, staging o production",
    "LOG_LEVEL": "Nivel de logging (default: INFO)",
    "TELEGRAM_BOT_TOKEN": "Token del bot de Telegram para alertas",
    "TELEGRAM_CHAT_ID": "Chat ID de Telegram para recibir alertas",
    "MONGODB_ENABLE_LOGGING": "Habilita el LoggingMiddleware (default: true)",
    "MONGODB_RETRY_WRITES": "Retries en escrituras MongoDB (default: true)",
    "MONGODB_MAX_POOL_SIZE": "Tamaño del pool de conexiones (default: 50)",
    "MONGODB_TIMEOUT_MS": "Timeout de conexión en ms (default: 5000)",
}


def _check_vars(vars_map: Dict[str, str]) -> List[str]:
    """Retorna la lista de variables presentes del mapa dado.

    Args:
        vars_map: Diccionario variable → descripción.

    Returns:
        Lista con los nombres de las variables encontradas.
    """
    found = []
    for name in vars_map:
        value = os.getenv(name, "").strip()
        if value:
            found.append(name)
    return found


def _mask(value: str) -> str:
    """Enmascara un valor sensible mostrando solo sus últimos 4 caracteres.

    Args:
        value: Valor de la variable a enmascarar.

    Returns:
        Valor enmascarado o "(vacío)" si el valor no existe.
    """
    if not value:
        return "(vacío)"
    return f"****{value[-4:]}"


def main() -> int:
    """Ejecuta la verificación de variables de entorno.

    Returns:
        Código de salida: 0 si todas las obligatorias están presentes, 1 si no.
    """
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE VARIABLES DE ENTORNO (HF Spaces)")
    print("=" * 60)

    # Verificar obligatorias
    found_required = _check_vars(REQUIRED_VARS)
    missing_required = [k for k in REQUIRED_VARS if k not in found_required]

    print("\n📌 OBLIGATORIAS:")
    for name, desc in REQUIRED_VARS.items():
        status = "✅" if name in found_required else "❌"
        value = _mask(os.getenv(name, ""))
        print(f"  {status} {name} = {value}   ({desc})")

    # Verificar recomendadas
    found_recommended = _check_vars(RECOMMENDED_VARS)
    missing_recommended = [k for k in RECOMMENDED_VARS if k not in found_recommended]

    print("\n⭐ RECOMENDADAS:")
    for name, desc in RECOMMENDED_VARS.items():
        status = "✅" if name in found_recommended else "⚠️"
        print(f"  {status} {name}   ({desc})")

    # Verificar opcionales
    found_optional = _check_vars(OPTIONAL_VARS)

    print("\n🔹 OPCIONALES:")
    for name, desc in OPTIONAL_VARS.items():
        status = "✅" if name in found_optional else "➖"
        print(f"  {status} {name}   ({desc})")

    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print(f"  Encontradas  : {len(found_required)}/{len(REQUIRED_VARS)} obligatorias, "
          f"{len(found_recommended)}/{len(RECOMMENDED_VARS)} recomendadas, "
          f"{len(found_optional)}/{len(OPTIONAL_VARS)} opcionales")

    if missing_required:
        print(f"  ❌ FALTAN OBLIGATORIAS: {', '.join(missing_required)}")
        print("  🔧 Configura estas variables en Settings > Variables and secrets del Space")
        print("     y vuelve a ejecutar este script.")
        return 1

    if missing_recommended:
        print(f"  ⚠️ Faltan recomendadas (el sistema degradará): {', '.join(missing_recommended)}")

    print("  ✅ Todas las variables obligatorias están configuradas.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
