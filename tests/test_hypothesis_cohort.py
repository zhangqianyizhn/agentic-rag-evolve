import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.planning import (
    build_hypothesis_cohort,
    build_hypothesis_cohort_with_audits,
    write_hypothesis_cohort,
)


def _diagnosis(task_id: str, status: str) -> dict:
    return {
        "schema_version": "deepread-diagnosis-v1",
        "task_id": task_id,
        "status": status,
        "failure_manifestation": "Observed failure.",
        "earliest_intervention": None,
        "root_cause_hypothesis": f"Root cause for {task_id}.",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "counterfactual": {},
        "affected_sources": [],
        "uncertainties": ["One case only."],
    }


class HypothesisCohortTest(unittest.TestCase):
    def _write(self, root: Path, task_id: str, status: str) -> Path:
        path = root / f"{task_id}.json"
        path.write_text(json.dumps(_diagnosis(task_id, status)))
        return path

    def test_only_diagnosed_cases_enter_repair_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnosed = self._write(root, "q2", "diagnosed")
            mismatch = self._write(root, "q1", "not_agent_failure")
            uncertain = self._write(root, "q3", "insufficient_evidence")

            cohort = build_hypothesis_cohort([diagnosed, mismatch, uncertain])

        self.assertEqual(cohort["counts"], {"input": 3, "eligible": 1, "excluded": 2})
        self.assertEqual(
            [item["task_id"] for item in cohort["eligible_diagnoses"]], ["q2"]
        )
        self.assertEqual(
            [item["route"] for item in cohort["excluded_diagnoses"]],
            ["evaluation_review", "evidence_review"],
        )

    def test_cohort_id_is_stable_across_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root, "q1", "diagnosed")
            second = self._write(root, "q2", "diagnosed")

            forward = build_hypothesis_cohort([first, second])
            reverse = build_hypothesis_cohort([second, first])

        self.assertEqual(forward["cohort_id"], reverse["cohort_id"])

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(_diagnosis("q1", "diagnosed")))
            second.write_text(json.dumps(_diagnosis("q1", "diagnosed")))

            with self.assertRaisesRegex(ValueError, "duplicate diagnosis task_id"):
                build_hypothesis_cohort([first, second])

    def test_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write(root, "q1", "diagnosed")
            output = root / "cohort.json"
            write_hypothesis_cohort([source], output)

            with self.assertRaises(FileExistsError):
                write_hypothesis_cohort([source], output)

    def test_skipped_diagnosis_audit_is_explicitly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema_version": "deepread-diagnosis-audit-v1",
                        "task_id": "q1",
                        "status": "skipped",
                        "route": {
                            "eligible": False,
                            "target": "answer_judge",
                            "reason": "Judge first.",
                        },
                        "reason": "Judge first.",
                    }
                )
            )
            cohort = build_hypothesis_cohort_with_audits([], [audit])

        self.assertEqual(cohort["counts"], {"input": 1, "eligible": 0, "excluded": 1})
        self.assertEqual(cohort["excluded_diagnoses"][0]["status"], "diagnosis_skipped")
        self.assertEqual(cohort["excluded_diagnoses"][0]["route"], "answer_judge")


if __name__ == "__main__":
    unittest.main()
