"""Translate typed agent observations into framework trace events."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .context import emit_trace


def _tool_call_summary(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.get("id"),
            "name": (call.get("function") or {}).get("name"),
        }
        for call in tool_calls
    ]


class TraceAgentObserver:
    def tool_calls_recovered(
        self,
        *,
        round_number: int,
        tool_calls: Sequence[Mapping[str, Any]],
        recovery_kind: str | None,
    ) -> None:
        emit_trace(
            "tool_calls_recovered_from_text",
            round=round_number,
            recovered=_tool_call_summary(tool_calls),
            recovered_kind=recovery_kind,
        )

    def tool_arguments_invalid(
        self,
        *,
        round_number: int,
        tool_name: str | None,
        call_id: str | None,
        raw_arguments: str,
        error: str,
    ) -> None:
        emit_trace(
            "tool_args_parse_error",
            round=round_number,
            tool=tool_name,
            tool_call_id=call_id,
            raw=raw_arguments,
            error=error,
        )

    def model_response_empty(
        self,
        *,
        round_number: int,
        reasoning_preview: str | None,
    ) -> None:
        if reasoning_preview:
            emit_trace(
                "llm_thinking_only",
                round=round_number,
                reasoning_preview=reasoning_preview,
            )
        else:
            emit_trace("llm_empty_message", round=round_number)
