import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class ThreatInfo:
    threat_type: str
    pattern: str
    severity: str
    position: int
    snippet: str

@dataclass
class SanitizationResult:
    is_safe: bool
    cleaned_text: str
    threats: List[ThreatInfo] = field(default_factory=list)
    severity: str = "none"

class InputSanitizer:
    INJECTION_PATTERNS = [
        (r"<script[\s>]", "html_script_tag", "critical"),
        (r"javascript\s*:", "js_protocol", "critical"),
        (r"on\w+\s*=\s*['\"]", "html_event_handler", "high"),
        (r"<iframe[\s>]", "html_iframe", "critical"),
        (r"<object[\s>]", "html_object", "critical"),
        (r"<embed[\s>]", "html_embed", "critical"),
        (r"<svg[\s/>]", "html_svg", "high"),
        (r"ALTER\s+TABLE", "sql_alter", "critical"),
        (r"ALTER\s+DATABASE", "sql_alter_db", "critical"),
        (r"DROP\s+TABLE", "sql_drop_table", "critical"),
        (r"DROP\s+DATABASE", "sql_drop_db", "critical"),
        (r"TRUNCATE\s+TABLE", "sql_truncate", "critical"),
        (r"EXEC\s*\(?", "sql_exec", "critical"),
        (r"xp_cmdshell", "sql_xp_cmdshell", "critical"),
        (r"__import__\s*\(", "python_import", "critical"),
        (r"eval\s*\(", "python_eval", "critical"),
        (r"exec\s*\(", "python_exec", "critical"),
        (r"os\.system\s*\(", "os_system", "critical"),
        (r"subprocess\.", "subprocess_call", "critical"),
        (r"base64", "base64_ref", "low"),
    ]

    ESCAPE_PATTERNS = [
        (r"ignora\s*(instrucciones|prompt|todo)", "ignore_instructions", "high"),
        (r"ignora\s*(las|todas|tus)\s*(instrucciones|reglas)", "ignore_instructions", "high"),
        (r"ignora\s*(las|todas)\s*tus\s*(instrucciones|reglas)", "ignore_instructions", "high"),
        (r"olvida\s*(instrucciones|prompt|reglas|todo)", "forget_instructions", "high"),
        (r"olvida\s*(las|todas|tus)\s*(instrucciones|reglas)", "forget_instructions", "high"),
        (r"olvida\s*(las|todas)\s*tus\s*(instrucciones|reglas)", "forget_instructions", "high"),
        (r"no\s*sigas\s*(instrucciones|reglas)", "dont_follow", "high"),
        (r"no\s*sigas\s*(las|tus)\s*(instrucciones|reglas)", "dont_follow", "high"),
        (r"desvíate", "deviate", "high"),
        (r"actúa\s*como\s*(si\s*no\s*)?(fueras|fueses)", "act_as", "medium"),
        (r"eres\s*libre", "you_are_free", "medium"),
        (r"no\s*tienes\s*restricciones", "no_restrictions", "high"),
        (r"sin\s*restricciones", "without_restrictions", "high"),
        (r"sistema\s*prompt", "system_prompt_ref", "high"),
        (r"prompt\s*original", "original_prompt_ref", "high"),
        (r"muéstrame\s*(tu\s*)?prompt", "show_prompt", "critical"),
        (r"dime\s*(tu\s*)?prompt", "tell_prompt", "critical"),
        (r"reveal\s*(your\s*)?prompt", "reveal_prompt", "critical"),
        (r"override\s*(security|protocols|system)", "override_security", "high"),
        (r"bypass\s*(security|restrictions|rules)", "bypass_security", "high"),
        (r"token\s*de\s*sistema", "system_token", "high"),
        (r"eres\s*un\s*asistente\s*(anterior|diferente)", "different_assistant", "medium"),
    ]

    OBJFUSCATION_PATTERNS = [
        (r"[A-Za-z0-9+/]{40,}={0,2}", "base64_long", "high"),
        (r"(?:\\x[0-9a-fA-F]{2}){10,}", "hex_encoding", "high"),
        (r"(?:\\u[0-9a-fA-F]{4}){5,}", "unicode_escape", "high"),
        (r"(?:&#\d{2,};){5,}", "html_entity", "medium"),
        (r"(?:%[0-9a-fA-F]{2}){10,}", "url_encoding", "high"),
        (r"[^\x20-\x7E\n\r\t]{5,}", "non_ascii_excess", "medium"),
        (r"(.)\1{20,}", "repeated_char", "low"),
        (r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", "unicode_superscript", "low"),
        (r"[₀₁₂₃₄₅₆₇₈₉]", "unicode_subscript", "low"),
    ]

    MAX_LENGTH = 5000

    @classmethod
    def sanitize(cls, text: str) -> SanitizationResult:
        if not text or not isinstance(text, str):
            return SanitizationResult(is_safe=True, cleaned_text=text or "", severity="none")

        if len(text) > cls.MAX_LENGTH:
            logger.warning(f"🔒 Texto excede longitud máxima: {len(text)} > {cls.MAX_LENGTH}")
            return SanitizationResult(
                is_safe=False,
                cleaned_text=text[:cls.MAX_LENGTH],
                threats=[ThreatInfo("length", "max_length_exceeded", "medium", 0, text[:50])],
                severity="medium"
            )

        threats = []
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        max_sev = "none"

        for pattern, threat_type, severity in cls.INJECTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                snippet = text[max(0, pos - 20):pos + len(match.group()) + 20]
                threats.append(ThreatInfo(threat_type, pattern, severity, pos, snippet))
                if severity_order.get(severity, 0) > severity_order.get(max_sev, 0):
                    max_sev = severity

        for pattern, threat_type, severity in cls.ESCAPE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                snippet = text[max(0, pos - 20):pos + len(match.group()) + 20]
                threats.append(ThreatInfo(threat_type, pattern, severity, pos, snippet))
                if severity_order.get(severity, 0) > severity_order.get(max_sev, 0):
                    max_sev = severity

        for pattern, threat_type, severity in cls.OBJFUSCATION_PATTERNS:
            for match in re.finditer(pattern, text):
                pos = match.start()
                snippet = text[max(0, pos - 20):pos + len(match.group()) + 20]
                threats.append(ThreatInfo(threat_type, pattern, severity, pos, snippet))
                if severity_order.get(severity, 0) > severity_order.get(max_sev, 0):
                    max_sev = severity

        is_safe = max_sev in ("none", "low")

        if threats:
            level = logging.WARNING if is_safe else logging.ERROR
            logger.log(level, f"🚨 Sanitizer detectó {len(threats)} amenaza(s), severidad máxima: {max_sev}")
            for t in threats:
                logger.log(level, f"🚨   [{t.severity}] {t.threat_type} en posición {t.position}: {t.snippet[:60]}")

        return SanitizationResult(
            is_safe=is_safe,
            cleaned_text=text,
            threats=threats,
            severity=max_sev
        )

    @classmethod
    def is_safe(cls, text: str) -> bool:
        return cls.sanitize(text).is_safe
