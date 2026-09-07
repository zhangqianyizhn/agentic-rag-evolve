"""Provider-neutral capabilities required by the DeepRead target."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class EventLogger(Protocol):
    def log(self, event: str, **fields: Any) -> None: ...


class ChatModel(Protocol):
    """A chat-completion capability injected by the stable runner."""

    model_name: str
    base_url: str

    def complete(
        self,
        payload: Mapping[str, Any],
        *,
        logger: EventLogger | None = None,
        query_id: str = "",
    ) -> Mapping[str, Any]: ...


class EmbeddingModel(Protocol):
    """A text-embedding capability shared by ingestion and query retrieval."""

    model_name: str
    base_url: str
    normalized: bool

    def embed(self, text: str) -> Sequence[float]: ...


class Reranker(Protocol):
    """A document reranking capability used by semantic retrieval."""

    model_name: str

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int = -1,
    ) -> Mapping[str, Any]: ...
