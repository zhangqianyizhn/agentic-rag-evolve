import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agentic_rag_evolve.telemetry import (
    JsonlTraceWriter,
    TraceAgentObserver,
    TracingChatModel,
    TracingToolExecutor,
    trace_context,
)
from systems.deepread.DeepRead.agent.runner import run_agent
from systems.deepread.DeepRead.tool import DeepReadToolExecutor
from systems.deepread.DeepRead.tool.retrieval import DocIndex


class FakeChatModel:
    model_name = "fake-chat"
    base_url = "local://fake"

    def __init__(self) -> None:
        self.payloads: list[Mapping[str, Any]] = []

    def complete(self, payload):
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


class ToolCallingChatModel:
    model_name = "fake-chat"
    base_url = "local://fake"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, payload):
        self.call_count += 1
        if self.call_count == 1:
            message = {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bm25_search", "arguments": '{"query":"text","scope":"full"}'},
                }],
            }
        else:
            message = {"content": "answer", "tool_calls": None}
        return {"choices": [{"message": message}], "usage": {}}


class ProviderBoundaryTest(unittest.TestCase):
    def test_agent_receives_chat_capability_without_credentials(self) -> None:
        index = DocIndex(
            [{"id": "n1", "doc_id": "1", "title": "Title", "paragraphs": ["text"], "children": []}],
            neighbor_window=None,
        )
        model = FakeChatModel()
        outcome = run_agent(
            chat_model=model,
            doc_index=index,
            tool_executor=TracingToolExecutor(DeepReadToolExecutor(index)),
            user_question="question",
            observer=TraceAgentObserver(),
            max_rounds=1,
        )
        self.assertEqual(outcome.answer, "answer")
        self.assertEqual(outcome.termination_reason, "final_answer")
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

    def test_tool_result_repeats_explicit_call_id_in_raw_trace(self) -> None:
        index = DocIndex(
            [{"id": "n1", "doc_id": "1", "title": "Title", "paragraphs": ["text"], "children": []}],
            neighbor_window=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            writer = JsonlTraceWriter(trace_path)
            with trace_context(writer):
                outcome = run_agent(
                    chat_model=TracingChatModel(ToolCallingChatModel()),
                    doc_index=index,
                    tool_executor=TracingToolExecutor(DeepReadToolExecutor(index)),
                    user_question="question",
                    observer=TraceAgentObserver(),
                    max_rounds=2,
                )
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        result_event = next(event for event in events if event["event"] == "tool_result")
        self.assertEqual(result_event["tool_call_id"], "call-1")
        self.assertEqual(
            [event["event"] for event in events],
            [
                "llm_request",
                "llm_response",
                "tool_call",
                "tool_result",
                "llm_request",
                "llm_response",
            ],
        )
        self.assertEqual(outcome.answer, "answer")


if __name__ == "__main__":
    unittest.main()
