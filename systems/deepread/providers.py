"""Provider adapters used by the DeepRead baseline."""

from __future__ import annotations

import math
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


class VolcengineMultimodalEmbeddingProvider:
    """Adapter for the Ark multimodal text embedding endpoint.

    ``client`` is injectable so ingestion tests do not require network access.
    """

    normalized = True

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        dimension: int | None = 2048,
        *,
        client: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if client is None and not api_key:
            raise ValueError("api_key is required when client is not provided")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        if client is None:
            import volcenginesdkarkruntime

            client = volcenginesdkarkruntime.Ark(
                api_key=api_key,
                base_url=self.base_url,
            )
        self._client = client

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            if self.dimension is None:
                raise ValueError("dimension is required to embed empty text")
            return [0.0] * self.dimension

        response = self._client.multimodal_embeddings.create(
            input=[{"type": "text", "text": text}],
            model=self.model_name,
        )
        return truncate_and_normalize(response.data.embedding, self.dimension)
