"""Tracing decorator for target-owned tool executors."""

from __future__ import annotations

from typing import Any, Mapping

from .context import emit_trace


class TracingToolExecutor:
    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> Mapping[str, Any]:
        emit_trace(
            "tool_call",
            tool=name,
            args=dict(arguments),
            tool_call_id=call_id,
        )
        try:
            result = self._executor.execute(name, arguments)
        except Exception as exc:
            emit_trace(
                "tool_result",
                tool=name,
                tool_call_id=call_id,
                ok=False,
                error=str(exc),
                result={"ok": False, "error": str(exc)},
            )
            raise
        emit_trace(
            "tool_result",
            tool=name,
            tool_call_id=call_id,
            ok=bool(result.get("ok", True)),
            result=result,
        )
        return result
