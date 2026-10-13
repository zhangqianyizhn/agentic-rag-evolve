import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.telemetry.context import emit_trace
from agentic_rag_evolve.telemetry.evolution import (
    record_agent_tool, reconstruct_requests, traced_agent, traced_model,
)
from agentic_rag_evolve.evaluation.evaluator import evaluate_financebench


class Model:
    model_name = "test-model"
    api_key = "SECRET-NOT-TO-LOG"
    default_headers = {"Authorization": "SECRET-NOT-TO-LOG"}

    def complete(self, payload):
        emit_trace("llm_http_attempt", attempt=1)
        return {"choices": [{"message": {"content": "full answer",
                    "reasoning_content": "full returned reasoning"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7}}


class EvolutionTraceTest(unittest.TestCase):
    def test_judge_keeps_full_prompt_response_and_final_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps({"financebench_id": "q1", "doc_name": "report",
                                          "question": "Q", "answer": "A"}) + "\n")
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"task_id": "q1", "answer": "A"}) + "\n")
            class Judge(Model):
                def complete(self, payload):
                    return {"choices": [{"message": {"content": '{"score":4,"reasoning":"matches"}',
                                      "reasoning_content": "complete returned thought"}}]}
            path = root / "judge.trace.json"
            _, summary = evaluate_financebench(dataset_path=dataset, prediction_path=predictions,
                                               judge_model=Judge(), trace_path=path)
            trace = json.loads(path.read_text())
            self.assertEqual(summary.judge_errors, 0)
            self.assertIn("Gold Answer", trace["events"][0]["messages_delta"][1]["content"])
            self.assertEqual(trace["events"][0]["response"]["choices"][0]["message"]["reasoning_content"],
                             "complete returned thought")
            self.assertEqual(trace["outcome"], summary.to_dict())

    def test_lossless_history_including_replacement_and_shrinking_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.trace.json"
            payloads = [
                {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}], "tools": [{"name": "read"}]},
                {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}, {"role": "tool", "content": "R"}]},
                {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "changed"}]},
                {"messages": [{"role": "system", "content": "S"}]},
            ]
            with traced_model(path, "test", Model()) as trace:
                for payload in payloads:
                    trace.complete(payload)
                result = "x" * 30000
                record_agent_tool(name="replace_text", arguments={"old_text": "one", "new_text": "two"}, result=result)
                trace.finish({"status": "ok"})
            value = json.loads(path.read_text())
            self.assertEqual(list(reconstruct_requests(value)), payloads)
            calls = [event for event in value["events"] if event["type"] == "model"]
            self.assertEqual([call["messages_prefix"] for call in calls], [0, 2, 1, 1])
            self.assertEqual(calls[0]["response"]["choices"][0]["message"]["reasoning_content"], "full returned reasoning")
            self.assertEqual(len(calls[0]["transport"]), 1)
            self.assertEqual(value["events"][-1]["result"], result)
            self.assertEqual(value["events"][-1]["arguments"]["old_text"], "one")
            self.assertNotIn("SECRET-NOT-TO-LOG", path.read_text())
            old = path.read_bytes()
            record_agent_tool(name="outside", arguments={}, result={})
            self.assertEqual(path.read_bytes(), old)

    def test_pending_request_visible_during_call_and_error_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.trace.json"
            class Failure(Model):
                def complete(self, payload):
                    pending = json.loads(path.read_text())["events"][0]
                    assert pending["status"] == "pending"
                    assert pending["messages_delta"][0]["content"] == "full input"
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                with traced_model(path, "test", Failure()) as trace:
                    trace.complete({"messages": [{"role": "user", "content": "full input"}]})
            value = json.loads(path.read_text())
            self.assertEqual(len(value["events"]), 1)
            self.assertEqual(value["status"], "error")
            self.assertEqual(value["events"][0]["status"], "error")

    def test_non_overwrite_guard_and_skipped_agent(self):
        @traced_agent("test", directory_output=True)
        def agent(*, model, output_path):
            if output_path.exists() and any(output_path.iterdir()):
                raise FileExistsError("non-empty")
            output_path.mkdir()
            (output_path / "audit.json").write_text('{"status":"skipped"}')
            return {"status": "skipped"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            agent(model=Model(), output_path=output)
            trace = json.loads((output / "agent_trace.json").read_text())
            self.assertEqual(trace["events"], [])
            self.assertEqual(trace["outcome"]["status"], "skipped")
            old = (output / "agent_trace.json").read_bytes()
            with self.assertRaises(FileExistsError):
                agent(model=Model(), output_path=output)
            self.assertEqual(old, (output / "agent_trace.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
