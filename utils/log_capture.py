import logging
import json
from datetime import datetime
from pathlib import Path

class DashboardLogHandler(logging.Handler):
    def __init__(self, log_file="data/system_logs.jsonl"):
        super().__init__()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
                "line": record.lineno if hasattr(record, "lineno") else None,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)
