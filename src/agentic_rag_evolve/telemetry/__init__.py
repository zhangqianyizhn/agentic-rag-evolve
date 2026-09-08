"""Framework-owned telemetry primitives."""

from .jsonl import JsonlTraceWriter, ScopedTraceWriter

__all__ = ["JsonlTraceWriter", "ScopedTraceWriter"]
