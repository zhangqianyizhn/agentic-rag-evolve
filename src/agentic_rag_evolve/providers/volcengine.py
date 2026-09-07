"""Volcengine Ark embedding adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


def truncate_and_normalize(
    embedding: Sequence[float], dimension: int | None
) -> list[float]:
    """Preserve the vector transformation used by ruc-ov-eval ``fb8a301c``."""

    vector = list(embedding)
    if not dimension or len(vector) <= dimension:
        return vector
    vector = vector[:dimension]
    norm = math.sqrt(sum(value**2 for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


@dataclass(slots=True)
class VolcengineMultimodalEmbeddingModel:
    model_name: str
    api_key: str
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    dimension: int | None = 2048
    client: Any | None = field(default=None, repr=False)
    normalized: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.model_name.strip():
            raise ValueError("embedding model_name must not be empty")
        if self.client is None and not self.api_key:
            raise ValueError("embedding api_key is required")
        if self.client is None:
            import volcenginesdkarkruntime

            self.client = volcenginesdkarkruntime.Ark(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            if self.dimension is None:
                raise ValueError("dimension is required to embed empty text")
            return [0.0] * self.dimension
        response = self.client.multimodal_embeddings.create(
            input=[{"type": "text", "text": text}],
            model=self.model_name,
        )
        return truncate_and_normalize(response.data.embedding, self.dimension)
