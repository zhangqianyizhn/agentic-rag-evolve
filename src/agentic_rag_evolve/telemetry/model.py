"""Provider-neutral tracing decorator for chat models."""

from __future__ import annotations

from typing import Any, Mapping

from .context import emit_trace


def _tool_call_summary(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    return [
        {
            "id": call.get("id"),
            "name": (call.get("function") or {}).get("name"),
        }
        for call in tool_calls
    ]


class TracingChatModel:
    """Record one request/response pair per model decision.

    The wrapped provider remains responsible only for transport. A fresh wrapper
    is created per query, so its monotonically increasing call number is the
    DeepRead round number without requiring the agent loop to calculate trace ids.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._call_number = 0

    @property
    def model_name(self) -> str:
        return str(self._model.model_name)

    @property
    def base_url(self) -> str:
        return str(self._model.base_url)

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._call_number += 1
        round_number = self._call_number
        emit_trace(
            "llm_request",
            round=round_number,
            model=self.model_name,
        )
        try:
            response = self._model.complete(payload)
        except Exception as exc:
            emit_trace(
                "llm_call_error",
                round=round_number,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        message = (response.get("choices") or [{}])[0].get("message") or {}
        reasoning = next(
            (
                message.get(field)
                for field in ("reasoning", "reasoning_content", "thinking", "internal_monologue")
                if message.get(field) is not None
            ),
            None,
        )
        emit_trace(
            "llm_response",
            round=round_number,
            content=message.get("content"),
            reasoning_content=reasoning,
            tool_calls=_tool_call_summary(message.get("tool_calls")),
        )
        return response
