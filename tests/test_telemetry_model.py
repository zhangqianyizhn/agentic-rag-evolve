import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.telemetry import (
    JsonlTraceWriter,
    ScopedTraceWriter,
    TracingChatModel,
    trace_context,
)


class FakeModel:
    model_name = "fake-model"
    base_url = "local://fake"

    def complete(self, payload):
        return {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "search first",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "bm25_search", "arguments": "{}"},
                    }],
                }
            }]
        }


class TelemetryModelTest(unittest.TestCase):
    def test_model_wrapper_owns_round_and_model_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            writer = JsonlTraceWriter(path)
            scoped = ScopedTraceWriter(writer, {"run_id": "r1", "task_id": "q1"})
            model = TracingChatModel(FakeModel())

            with trace_context(scoped):
                model.complete({"messages": []})
                model.complete({"messages": []})

            events = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([event["event"] for event in events], [
            "llm_request", "llm_response", "llm_request", "llm_response"
        ])
        self.assertEqual([events[0]["round"], events[2]["round"]], [1, 2])
        self.assertEqual(events[1]["tool_calls"], [{"id": "c1", "name": "bm25_search"}])
        self.assertTrue(all(event["task_id"] == "q1" for event in events))

    def test_model_wrapper_is_silent_without_trace_context(self) -> None:
        model = TracingChatModel(FakeModel())
        response = model.complete({"messages": []})
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "search first")


if __name__ == "__main__":
    unittest.main()
