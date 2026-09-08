"""Append-only JSONL trace storage owned by the evolution framework."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class JsonlTraceWriter:
    """Persist self-contained raw events and assign storage-level event ids."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._event_sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as stream:
                self._event_sequence = sum(1 for line in stream if line.strip())
        else:
            self.path.touch()

    def log(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._event_sequence += 1
            record = {
                "schema_version": 1,
                "event_id": f"event_{self._event_sequence:06d}",
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **fields,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class ScopedTraceWriter:
    """Attach run/task identity without exposing scoping logic to a target."""

    def __init__(self, writer: JsonlTraceWriter, scope: Mapping[str, Any]) -> None:
        self._writer = writer
        self._scope = {key: value for key, value in scope.items() if value is not None}

    def log(self, event: str, **fields: Any) -> None:
        self._writer.log(event, **self._scope, **fields)
