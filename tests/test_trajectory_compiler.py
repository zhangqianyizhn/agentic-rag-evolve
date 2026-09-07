import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.trajectory import compile_trajectories


class TrajectoryCompilerTest(unittest.TestCase):
    def test_compiles_legacy_trace_and_links_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = "What was revenue?"
            query_id = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
            prediction = {
                "task_id": "q1",
                "question": question,
                "status": "ok",
                "answer": "12 million",
                "token_usage": {"input_tokens": 10, "output_tokens": 2},
            }
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            raw = [
                {"ts": "2026-01-01T00:00:00+00:00", "event": "llm_request", "query_id": query_id, "round": 1, "context_delta_preview": "large duplicate"},
                {"ts": "2026-01-01T00:00:01+00:00", "event": "llm_response", "query_id": query_id, "round": 1, "tool_calls": [{"id": "c1", "name": "bm25_search"}]},
                {"ts": "2026-01-01T00:00:02+00:00", "event": "tool_call", "query_id": query_id, "tool": "bm25_search", "tool_call_id": "c1", "args": {"query": "revenue"}},
                {"ts": "2026-01-01T00:00:03+00:00", "event": "tool_result", "query_id": query_id, "tool": "bm25_search", "tool_call_id": "c1", "result": {"results": [{"ref": {"doc_id": "3"}, "text": "Revenue"}]}},
                {"ts": "2026-01-01T00:00:04+00:00", "event": "final_answer", "query_id": query_id, "answer": "12 million"},
            ]
            trace = root / "trace.jsonl"
            trace.write_text("".join(json.dumps(item) + "\n" for item in raw), encoding="utf-8")

            report = compile_trajectories(
                trace_path=trace,
                prediction_path=predictions,
                output_path=root / "compiled",
            )
            trajectory = json.loads(
                (root / "compiled" / "q1.trajectory.json").read_text(encoding="utf-8")
            )

            self.assertEqual(report.unassigned_event_count, 0)
            self.assertEqual(trajectory["schema_version"], "deepread-trajectory-v1")
            self.assertEqual(trajectory["events"][3]["parent_id"], "step_0003")
            self.assertEqual(trajectory["events"][0]["payload"]["omitted_fields"], ["context_delta_preview"])
            self.assertNotIn("large duplicate", json.dumps(trajectory))
            self.assertEqual(trajectory["summary"]["retrieved_doc_ids"], ["3"])
            self.assertEqual(trajectory["summary"]["terminal_event"], "answer.final")

    def test_reports_unassigned_events_and_missing_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"task_id": "q1", "question": "Q", "status": "ok", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({"event": "llm_request", "query_id": "other"}) + "\n", encoding="utf-8")

            report = compile_trajectories(
                trace_path=trace,
                prediction_path=predictions,
                output_path=root / "compiled",
            )
            trajectory = json.loads((root / "compiled" / "q1.trajectory.json").read_text())

            self.assertEqual(report.unassigned_event_count, 1)
            self.assertIn("missing_terminal_event", trajectory["summary"]["warnings"])

    def test_infers_missing_tool_result_id_for_legacy_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = "Q"
            query_id = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
            (root / "predictions.jsonl").write_text(
                json.dumps({"task_id": "q1", "question": question, "status": "ok", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            events = [
                {"event": "llm_request", "query_id": query_id, "round": 1},
                {"event": "llm_response", "query_id": query_id, "round": 1},
                {"event": "tool_call", "query_id": query_id, "tool_call_id": "c1", "tool": "bm25_search"},
                {"event": "tool_result", "query_id": query_id, "tool": "bm25_search", "result": {"results": []}},
                {"event": "final_answer", "query_id": query_id, "answer": "A"},
            ]
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            compile_trajectories(
                trace_path=root / "trace.jsonl",
                prediction_path=root / "predictions.jsonl",
                output_path=root / "compiled",
            )
            trajectory = json.loads((root / "compiled" / "q1.trajectory.json").read_text())
            self.assertEqual(trajectory["events"][3]["parent_id"], "step_0003")
            self.assertIn(
                "legacy_inferred_tool_parent:legacy_000004",
                trajectory["summary"]["warnings"],
            )

    def test_refuses_to_overwrite_compiled_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "compiled"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                compile_trajectories(
                    trace_path=root / "missing-trace.jsonl",
                    prediction_path=root / "missing-predictions.jsonl",
                    output_path=output,
                )


if __name__ == "__main__":
    unittest.main()
