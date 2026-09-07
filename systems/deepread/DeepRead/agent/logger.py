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
        self._event_sequence = 0
        if not os.path.exists(path):
            with open(self.path, "w", encoding="utf-8"):
                pass
        else:
            with open(self.path, "r", encoding="utf-8") as stream:
                self._event_sequence = sum(1 for line in stream if line.strip())

    def log(self, event: str, **payload: Any) -> None:
        with self._lock:
            self._event_sequence += 1
            rec = {
                "schema_version": 1,
                "event_id": f"event_{self._event_sequence:06d}",
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
