"""Task-local access to the active raw trace writer."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Protocol


class TraceSink(Protocol):
    def log(self, event: str, **fields: Any) -> None: ...


_ACTIVE_TRACE: ContextVar[TraceSink | None] = ContextVar("active_trace", default=None)


@contextmanager
def trace_context(sink: TraceSink) -> Iterator[None]:
    token = _ACTIVE_TRACE.set(sink)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


def emit_trace(event: str, **fields: Any) -> None:
    sink = _ACTIVE_TRACE.get()
    if sink is not None:
        sink.log(event, **fields)
