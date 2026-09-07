from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any


# ------------------------------
# Logging helper (JSONL)
# ------------------------------
class JsonlLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(path):
            with open(self.path, "w", encoding="utf-8"):
                pass

    def log(self, event: str, **payload: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        line =json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
