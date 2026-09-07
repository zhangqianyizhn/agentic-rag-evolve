"""Stable data contracts shared by benchmark adapters, runners, and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class DocumentInput:
    """One source document visible to the RAG system."""

    document_id: str
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _require_text(self.document_id, "document_id"))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True, slots=True)
class DocumentQATask:
    """Runner input. Gold answers and evidence are deliberately excluded."""

    task_id: str
    question: str
    documents: tuple[DocumentInput, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        object.__setattr__(self, "question", _require_text(self.question, "question"))
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.documents:
            raise ValueError("documents must contain at least one document")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document_id values must be unique within a task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "documents": [document.to_dict() for document in self.documents],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReference:
    """Evaluator-only labels kept outside the agent-visible task contract."""

    task_id: str
    gold_answers: tuple[str, ...]
    gold_evidence: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        answers = tuple(_require_text(answer, "gold_answer") for answer in self.gold_answers)
        evidence = tuple(text.strip() for text in self.gold_evidence if text.strip())
        object.__setattr__(self, "gold_answers", answers)
        object.__setattr__(self, "gold_evidence", evidence)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not answers:
            raise ValueError("gold_answers must contain at least one answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "gold_answers": list(self.gold_answers),
            "gold_evidence": list(self.gold_evidence),
            "metadata": dict(self.metadata),
        }
