import math
import unittest
from types import SimpleNamespace

from systems.deepread.providers import (
    VolcengineMultimodalEmbeddingProvider,
    truncate_and_normalize,
)


class FakeMultimodalEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=SimpleNamespace(embedding=[3.0, 4.0, 12.0]))


class EmbeddingProviderTest(unittest.TestCase):
    def test_truncates_and_normalizes_only_when_vector_is_too_long(self) -> None:
        self.assertEqual(truncate_and_normalize([3.0, 4.0], 2), [3.0, 4.0])
        truncated = truncate_and_normalize([3.0, 4.0, 12.0], 2)
        self.assertTrue(math.isclose(truncated[0], 0.6))
        self.assertTrue(math.isclose(truncated[1], 0.8))

    def test_uses_multimodal_text_payload(self) -> None:
        endpoint = FakeMultimodalEmbeddings()
        client = SimpleNamespace(multimodal_embeddings=endpoint)
        provider = VolcengineMultimodalEmbeddingProvider(
            "embedding-model",
            api_key="",
            base_url="https://example.invalid/v3/",
            dimension=2,
            client=client,
        )

        vector = provider.embed("Revenue was 42.")

        self.assertEqual(endpoint.calls, [{
            "input": [{"type": "text", "text": "Revenue was 42."}],
            "model": "embedding-model",
        }])
        self.assertEqual(provider.base_url, "https://example.invalid/v3")
        self.assertTrue(math.isclose(vector[0], 0.6))
        self.assertTrue(math.isclose(vector[1], 0.8))


if __name__ == "__main__":
    unittest.main()
