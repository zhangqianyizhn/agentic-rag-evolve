import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.reporting import build_no_candidate_iteration_report


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class NoCandidateReportTest(unittest.TestCase):
    def test_builds_compact_terminal_report_for_empty_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = _write(
                root / "cohort.json",
                {
                    "schema_version": "deepread-hypothesis-cohort-v1",
                    "cohort_id": "cohort-1",
                    "counts": {"input": 1, "eligible": 0, "excluded": 1},
                    "eligible_diagnoses": [],
                    "excluded_diagnoses": [{"task_id": "secret-task"}],
                },
            )
            hypotheses = _write(
                root / "hypotheses.json",
                {
                    "schema_version": "deepread-improvement-hypotheses-v1",
                    "cohort_id": "cohort-1",
                    "status": "no_hypotheses",
                    "hypotheses": [],
                },
            )
            plan = _write(
                root / "plan.json",
                {
                    "schema_version": "deepread-modification-plan-v1",
                    "cohort_id": "cohort-1",
                    "hypothesis_set_status": "no_hypotheses",
                    "plans": [],
                },
            )
            report = build_no_candidate_iteration_report(
                cohort_path=cohort,
                hypotheses_path=hypotheses,
                plan_path=plan,
                output_path=root / "report.json",
            )

        self.assertEqual(report["decision"]["outcome"], "no_candidate")
        self.assertEqual(report["decision"]["reason"], "no_repair_eligible_diagnoses")
        self.assertIsNone(report["candidate"])
        self.assertNotIn("secret-task", json.dumps(report))

    def test_rejects_proceeding_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = _write(root / "c.json", {"schema_version": "deepread-hypothesis-cohort-v1", "cohort_id": "c", "eligible_diagnoses": []})
            hypotheses = _write(root / "h.json", {"schema_version": "deepread-improvement-hypotheses-v1", "cohort_id": "c", "hypotheses": []})
            plan = _write(root / "p.json", {"schema_version": "deepread-modification-plan-v1", "cohort_id": "c", "plans": [{"decision": "proceed"}]})
            with self.assertRaisesRegex(ValueError, "proceeding plan"):
                build_no_candidate_iteration_report(
                    cohort_path=cohort,
                    hypotheses_path=hypotheses,
                    plan_path=plan,
                    output_path=root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
