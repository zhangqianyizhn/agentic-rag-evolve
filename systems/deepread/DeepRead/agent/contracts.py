"""Small behavior contracts emitted by the DeepRead agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    answer: str
    termination_reason: str
    rounds_completed: int


class AgentObserver(Protocol):
    def tool_calls_recovered(
        self,
        *,
        round_number: int,
        tool_calls: Sequence[Mapping[str, Any]],
        recovery_kind: str | None,
    ) -> None: ...

    def tool_arguments_invalid(
        self,
        *,
        round_number: int,
        tool_name: str | None,
        call_id: str | None,
        raw_arguments: str,
        error: str,
    ) -> None: ...

    def model_response_empty(
        self,
        *,
        round_number: int,
        reasoning_preview: str | None,
    ) -> None: ...
