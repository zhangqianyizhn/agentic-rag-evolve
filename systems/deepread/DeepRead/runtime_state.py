"""Runtime state formerly imported from the ruc-ov evaluation package.

This local tracker preserves the behavior of ruc-ov-eval's
``ThreadLocalTokenTracker`` while removing the target system's import-time
dependency on ``src.core.token_tracer_util``.
"""

from __future__ import annotations

import threading


class ThreadLocalTokenTracker:
    def __init__(self) -> None:
        self._local = threading.local()

    def _ensure(self) -> None:
        if not hasattr(self._local, "input_tokens"):
            self._local.input_tokens = 0
            self._local.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self._ensure()
        self._local.input_tokens += int(input_tokens)
        self._local.output_tokens += int(output_tokens)

    def reset(self) -> None:
        self._local.input_tokens = 0
        self._local.output_tokens = 0

    def get(self) -> dict[str, int]:
        self._ensure()
        return {
            "input_tokens": self._local.input_tokens,
            "output_tokens": self._local.output_tokens,
        }


token_tracker = ThreadLocalTokenTracker()
