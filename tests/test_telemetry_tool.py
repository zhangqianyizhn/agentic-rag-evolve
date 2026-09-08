import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.telemetry import JsonlTraceWriter, TracingToolExecutor, trace_context


class SuccessfulExecutor:
    def execute(self, name, arguments):
        return {"ok": True, "echo": arguments["query"]}


class FailingExecutor:
    def execute(self, name, arguments):
        raise ValueError("invalid section")


class TelemetryToolTest(unittest.TestCase):
    def _run(self, executor):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        writer = JsonlTraceWriter(path)
        with trace_context(writer):
            result = executor.execute("bm25_search", {"query": "revenue"}, call_id="c1")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        return result, events

    def test_merges_successful_call_with_explicit_correlation_id(self) -> None:
        result, events = self._run(TracingToolExecutor(SuccessfulExecutor()))

        self.assertEqual(result, {"ok": True, "echo": "revenue"})
        self.assertEqual([event["event"] for event in events], ["tool_call", "tool_result"])
        self.assertTrue(all(event["tool_call_id"] == "c1" for event in events))
        self.assertEqual(events[1]["result"], result)

    def test_records_failure_once_and_preserves_exception(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.jsonl"
        writer = JsonlTraceWriter(path)
        executor = TracingToolExecutor(FailingExecutor())

        with self.assertRaisesRegex(ValueError, "invalid section"):
            with trace_context(writer):
                executor.execute("read_section", {}, call_id="c2")

        events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["tool_call", "tool_result"])
        self.assertFalse(events[1]["ok"])
        self.assertEqual(events[1]["error"], "invalid section")


if __name__ == "__main__":
    unittest.main()
