import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from systems.deepread.DeepRead.agent.logger import JsonlLogger
from systems.deepread.DeepRead.agent.runner import run_agent
from systems.deepread.DeepRead.tool.retrieval import DocIndex


class FakeChatModel:
    model_name = "fake-chat"
    base_url = "local://fake"

    def __init__(self) -> None:
        self.payloads: list[Mapping[str, Any]] = []

    def complete(self, payload, *, logger=None, query_id=""):
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"content": "answer", "tool_calls": None}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }


class FakeEmbeddingModel:
    model_name = "fake-embedding"
    base_url = "local://fake"
    normalized = True

    def embed(self, text: str):
        return [1.0, 0.0]


class ProviderBoundaryTest(unittest.TestCase):
    def test_agent_receives_chat_capability_without_credentials(self) -> None:
        index = DocIndex(
            [{"id": "n1", "doc_id": "1", "title": "Title", "paragraphs": ["text"], "children": []}],
            neighbor_window=None,
        )
        model = FakeChatModel()
        with tempfile.TemporaryDirectory() as directory:
            answer = run_agent(
                chat_model=model,
                doc_index=index,
                user_question="question",
                logger=JsonlLogger(str(Path(directory) / "trace.jsonl")),
                max_rounds=1,
            )
        self.assertEqual(answer, "answer")
        self.assertEqual(model.payloads[0]["model"], "fake-chat")

    def test_vector_search_uses_injected_embedding_capability(self) -> None:
        index = DocIndex(
            [{"id": "n1", "doc_id": "1", "title": "Title", "paragraphs": ["text"], "children": []}],
            neighbor_window=None,
        )
        index._vec_matrix = np.asarray([[1.0, 0.0]], dtype=np.float32)
        index._vec_idmap = [{"doc_id": "1", "node_id": "n1", "paragraph_index": 0}]
        index._vec_normalized = True

        result = index.vector_search("question", embedding_model=FakeEmbeddingModel())

        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["text"], "text")

    def test_vector_search_reports_dimension_mismatch(self) -> None:
        index = DocIndex([], neighbor_window=None)
        index._vec_matrix = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        index._vec_idmap = [{"doc_id": "1", "node_id": "n1", "paragraph_index": 0}]

        result = index.vector_search("question", embedding_model=FakeEmbeddingModel())

        self.assertEqual(result["error"], "embedding_dimension_mismatch")
        self.assertEqual(result["expected"], 3)


if __name__ == "__main__":
    unittest.main()
