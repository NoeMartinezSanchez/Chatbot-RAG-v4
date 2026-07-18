import os
import time
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class SecurityIncident:
    timestamp: float
    threat_type: str
    severity: str
    snippet: str
    session_id: str
    source_ip: str
    details: Dict[str, Any] = field(default_factory=dict)

class SecurityMonitor:
    MAX_INCIDENTS = 1000

    def __init__(self):
        self._incidents: deque = deque(maxlen=self.MAX_INCIDENTS)
        self._stats_cache: Optional[Dict] = None
        self._stats_time: float = 0
        self._stats_ttl: float = 2.0
        logger.info("🔒 SecurityMonitor inicializado (memoria, máx 1000 incidentes)")

    def log_incident(
        self,
        threat_type: str,
        severity: str,
        snippet: str,
        session_id: str = "unknown",
        source_ip: str = "0.0.0.0",
        details: Optional[Dict] = None,
    ) -> SecurityIncident:
        incident = SecurityIncident(
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            snippet=snippet[:120],
            session_id=session_id,
            source_ip=source_ip,
            details=details or {},
        )
        self._incidents.append(incident)
        self._stats_cache = None

        log_msg = f"🚨 [{severity.upper()}] {threat_type} | sesión={session_id} | {snippet[:80]}"
        if severity in ("high", "critical"):
            logger.error(log_msg)
            self._try_telegram_alert(incident)
        else:
            logger.warning(log_msg)

        return incident

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        if self._stats_cache and (now - self._stats_time) < self._stats_ttl:
            return self._stats_cache

        total = len(self._incidents)
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for inc in self._incidents:
            by_severity[inc.severity] = by_severity.get(inc.severity, 0) + 1
            by_type[inc.threat_type] = by_type.get(inc.threat_type, 0) + 1

        window = 3600
        cutoff = time.time() - window
        recent_count = sum(1 for inc in self._incidents if inc.timestamp >= cutoff)

        result = {
            "total_incidents": total,
            "by_severity": dict(sorted(by_severity.items())),
            "by_type": dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True)),
            "incidents_last_hour": recent_count,
            "max_capacity": self.MAX_INCIDENTS,
            "usage_pct": round(total / self.MAX_INCIDENTS * 100, 1) if total else 0.0,
        }

        self._stats_cache = result
        self._stats_time = now
        return result

    def get_recent_incidents(self, limit: int = 50, min_severity: str = "low") -> List[Dict]:
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_order = severity_order.get(min_severity, 0)

        result = []
        for inc in reversed(self._incidents):
            if severity_order.get(inc.severity, 0) >= min_order:
                result.append({
                    "timestamp": datetime.fromtimestamp(inc.timestamp).isoformat(),
                    "threat_type": inc.threat_type,
                    "severity": inc.severity,
                    "snippet": inc.snippet,
                    "session_id": inc.session_id,
                    "source_ip": inc.source_ip,
                    "details": inc.details,
                })
                if len(result) >= limit:
                    break

        return result

    def _try_telegram_alert(self, incident: SecurityIncident) -> None:
        try:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not bot_token or not chat_id:
                logger.debug("🔒 Telegram no configurado, omitiendo alerta")
                return

            import requests

            fecha = datetime.fromtimestamp(incident.timestamp).strftime("%d/%m/%Y %H:%M:%S")
            message = (
                f"🚨 *ALERTA DE SEGURIDAD*\n"
                f"• Tipo: `{incident.threat_type}`\n"
                f"• Severidad: *{incident.severity.upper()}*\n"
                f"• Sesión: `{incident.session_id}`\n"
                f"• IP: `{incident.source_ip}`\n"
                f"• Fragmento: `{incident.snippet[:80]}`\n"
                f"• Fecha: {fecha}\n"
                f"• Tokens consumidos: 0"
            )

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }, timeout=15)

            if resp.status_code == 200:
                logger.debug("🔒 Alerta de seguridad enviada a Telegram")
            else:
                logger.debug(f"🔒 Telegram respondió {resp.status_code}")

        except Exception as e:
            logger.debug(f"🔒 Error enviando alerta Telegram: {e}")


_monitor: Optional[SecurityMonitor] = None

def get_monitor() -> SecurityMonitor:
    global _monitor
    if _monitor is None:
        _monitor = SecurityMonitor()
    return _monitor
