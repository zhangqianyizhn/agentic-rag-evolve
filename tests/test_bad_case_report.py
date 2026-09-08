import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.diagnostics import build_bad_case_report


class BadCaseReportTest(unittest.TestCase):
    def _bundle(self, root: Path, task_id: str, triage: str, is_bad_case: bool) -> Path:
        path = root / task_id
        path.mkdir()
        bundle = {
            "schema_version": "deepread-diagnostic-input-v1",
            "task": {
                "task_id": task_id,
                "sample_id": "report",
                "question": f"Question {task_id}?",
            },
            "evaluation": {
                "gold_answers": ["gold"],
                "prediction": {"answer": "generated"},
                "metrics": {"f1": 0.0, "recall": 0.0},
                "judge": {"status": "ok", "score": 0, "reasoning": "wrong"},
            },
            "evidence_coverage": {"evidence": []},
            "failure_signals": {
                "schema_version": "deepread-failure-signals-v1",
                "triage": triage,
                "confidence": "high",
                "is_bad_case": is_bad_case,
                "evidence_path": {
                    "primary_signal": "candidate",
                    "layer_recall": {"corpus": 1.0, "candidate": 0.0},
                },
                "signals": [
                    {
                        "code": "retrieval_candidate_miss",
                        "severity": "warning",
                        "summary": "Evidence was not retrieved.",
                    }
                ],
            },
        }
        (path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_builds_compact_json_and_markdown_for_bad_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = self._bundle(root, "q2", "incorrect_answer", True)
            passed = self._bundle(root, "q1", "pass", False)
            output = root / "report"
            result = build_bad_case_report(
                bundle_paths=[bad, passed], output_path=output
            )
            summary = json.loads((output / "summary.json").read_text())
            records = json.loads((output / "bad_cases.json").read_text())
            markdown = (output / "bad_cases.md").read_text()

        self.assertEqual(result.total, 2)
        self.assertEqual(result.bad_cases, 1)
        self.assertEqual(summary["triage_counts"], {"incorrect_answer": 1, "pass": 1})
        self.assertEqual([record["task_id"] for record in records], ["q2"])
        self.assertIn("q2 — `incorrect_answer`", markdown)
        self.assertNotIn("q1 — `pass`", markdown)

    def test_rejects_duplicate_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._bundle(root, "q1", "incorrect_answer", True)
            duplicate = root / "duplicate.json"
            duplicate.write_text((first / "bundle.json").read_text())

            with self.assertRaisesRegex(ValueError, "duplicate task IDs"):
                build_bad_case_report(
                    bundle_paths=[first, duplicate], output_path=root / "report"
                )


if __name__ == "__main__":
    unittest.main()
