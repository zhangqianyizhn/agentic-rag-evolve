import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.validation import evaluate_validation_gate


def _item(task_id: str, score: float, *, tokens: int = 100, status: str = "ok") -> dict:
    return {
        "task_id": task_id,
        "sample_id": f"doc-{task_id}",
        "question": f"Question {task_id}?",
        "prediction": {"status": status},
        "token_usage": {"input_tokens": tokens - 10, "output_tokens": 10},
        "metrics": {"f1": score, "recall": score, "accuracy_normalized": score},
    }


class ValidationGateTest(unittest.TestCase):
    def _write_inputs(self, root: Path, *, costly: bool = False, regress: bool = False):
        evaluations = {
            "dev-baseline.json": [_item("q1", 0.25)],
            "dev-candidate.json": [_item("q1", 0.75, tokens=110)],
            "hold-baseline.json": [_item("q2", 0.75)],
            "hold-candidate.json": [
                _item("q2", 0.5 if regress else 0.75, tokens=200 if costly else 110)
            ],
        }
        for name, value in evaluations.items():
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        suite = {
            "schema_version": "deepread-validation-suite-v1",
            "candidate_id": "candidate-1",
            "plan_id": "plan-1",
            "candidate_snapshot_sha256": "c" * 64,
            "gate_level": "development",
            "comparison_epsilon": 1e-9,
            "cohorts": [
                {
                    "name": "financebench-development",
                    "dataset": "financebench",
                    "role": "development",
                    "baseline_evaluation": "dev-baseline.json",
                    "candidate_evaluation": "dev-candidate.json",
                    "task_ids": ["q1"],
                    "primary_metric": "accuracy_normalized",
                    "min_mean_delta": 0.1,
                    "min_improved": 1,
                    "max_regressions": 0,
                    "max_token_cost_ratio": 1.25,
                },
                {
                    "name": "financebench-holdout",
                    "dataset": "financebench",
                    "role": "holdout",
                    "baseline_evaluation": "hold-baseline.json",
                    "candidate_evaluation": "hold-candidate.json",
                    "task_ids": ["q2"],
                    "primary_metric": "accuracy_normalized",
                    "min_mean_delta": 0.0,
                    "min_improved": 0,
                    "max_regressions": 0,
                    "max_token_cost_ratio": 1.25,
                },
            ],
        }
        suite_path = root / "suite.json"
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        plan_path = root / "plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-modification-plan-v1",
                    "plans": [{"plan_id": "plan-1", "decision": "proceed"}],
                }
            ),
            encoding="utf-8",
        )
        audit_path = root / "candidate-audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-candidate-audit-v1",
                    "candidate_id": "candidate-1",
                    "plan_id": "plan-1",
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "candidate_snapshot_sha256": "c" * 64,
                    "test_policy_id": "unit-v1",
                    "test_policy_sha256": "d" * 64,
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )
        test_audit_path = root / "candidate-test-audit.json"
        test_audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-candidate-test-audit-v1",
                    "candidate_id": "candidate-1",
                    "plan_id": "plan-1",
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "candidate_snapshot_sha256": "c" * 64,
                    "test_policy_id": "unit-v1",
                    "test_policy_sha256": "d" * 64,
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )
        return suite_path, audit_path, plan_path, test_audit_path

    def _evaluate(self, paths):
        return evaluate_validation_gate(
            suite_path=paths[0],
            candidate_audit_path=paths[1],
            plan_path=paths[2],
            candidate_test_audit_path=paths[3],
        )

    def test_gate_passes_improved_development_and_stable_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(Path(directory))
            result = self._evaluate(paths)

        self.assertTrue(result["passed"])
        development = result["cohorts"][0]
        self.assertEqual(development["improved_task_ids"], ["q1"])
        self.assertAlmostEqual(development["mean_delta"], 0.5)

    def test_gate_fails_holdout_regression_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(Path(directory), costly=True, regress=True)
            result = self._evaluate(paths)

        self.assertFalse(result["passed"])
        holdout = result["cohorts"][1]
        self.assertIn("mean_delta", holdout["failure_reasons"])
        self.assertIn("regression_count", holdout["failure_reasons"])
        self.assertIn("token_cost", holdout["failure_reasons"])

    def test_development_and_holdout_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            suite = json.loads(paths[0].read_text())
            suite["cohorts"][1]["task_ids"] = ["q1"]
            suite["cohorts"][1]["baseline_evaluation"] = "dev-baseline.json"
            suite["cohorts"][1]["candidate_evaluation"] = "dev-candidate.json"
            paths[0].write_text(json.dumps(suite))

            with self.assertRaisesRegex(ValueError, "leak"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_test_role_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            suite = json.loads(paths[0].read_text())
            suite["cohorts"][1]["role"] = "test"
            paths[0].write_text(json.dumps(suite))

            with self.assertRaisesRegex(ValueError, "unsupported role"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_promotion_gate_requires_cross_dataset_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            suite = json.loads(paths[0].read_text())
            suite["gate_level"] = "promotion"
            paths[0].write_text(json.dumps(suite))

            with self.assertRaisesRegex(ValueError, "cross_dataset"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_promotion_gate_accepts_disjoint_cross_dataset_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            (root / "cross-baseline.json").write_text(
                json.dumps([_item("cross-q1", 0.5)]), encoding="utf-8"
            )
            (root / "cross-candidate.json").write_text(
                json.dumps([_item("cross-q1", 0.5, tokens=110)]), encoding="utf-8"
            )
            suite = json.loads(paths[0].read_text())
            suite["gate_level"] = "promotion"
            suite["cohorts"].append(
                {
                    "name": "other-dataset-cross",
                    "dataset": "other-dataset",
                    "role": "cross_dataset",
                    "baseline_evaluation": "cross-baseline.json",
                    "candidate_evaluation": "cross-candidate.json",
                    "task_ids": ["cross-q1"],
                    "primary_metric": "accuracy_normalized",
                    "min_mean_delta": 0.0,
                    "min_improved": 0,
                    "max_regressions": 0,
                    "max_token_cost_ratio": 1.25,
                }
            )
            paths[0].write_text(json.dumps(suite))

            result = self._evaluate(paths)

        self.assertTrue(result["passed"])
        self.assertEqual(result["gate_level"], "promotion")

    def test_plan_bytes_must_match_static_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            plan = json.loads(paths[2].read_text())
            plan["unreviewed_change"] = True
            paths[2].write_text(json.dumps(plan))

            with self.assertRaisesRegex(ValueError, "plan hash"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_metric_must_be_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            candidate = json.loads((root / "dev-candidate.json").read_text())
            candidate[0]["metrics"]["accuracy_normalized"] = 1.1
            (root / "dev-candidate.json").write_text(json.dumps(candidate))

            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_missing_primary_metric_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            candidate = json.loads((root / "dev-candidate.json").read_text())
            candidate[0]["metrics"]["accuracy_normalized"] = None
            (root / "dev-candidate.json").write_text(json.dumps(candidate))

            with self.assertRaisesRegex(ValueError, "no comparable accuracy_normalized"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_static_audit_must_pass_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            audit = json.loads(paths[1].read_text())
            audit["passed"] = False
            paths[1].write_text(json.dumps(audit))

            with self.assertRaisesRegex(ValueError, "static audit must pass"):
                evaluate_validation_gate(
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    plan_path=paths[2],
                    candidate_test_audit_path=paths[3],
                )

    def test_fixed_tests_must_pass_before_behavior_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            test_audit = json.loads(paths[3].read_text())
            test_audit["passed"] = False
            paths[3].write_text(json.dumps(test_audit))

            with self.assertRaisesRegex(ValueError, "fixed tests must pass"):
                self._evaluate(paths)

    def test_fixed_test_snapshot_must_match_static_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            test_audit = json.loads(paths[3].read_text())
            test_audit["candidate_snapshot_sha256"] = "e" * 64
            paths[3].write_text(json.dumps(test_audit))

            with self.assertRaisesRegex(ValueError, "snapshot"):
                self._evaluate(paths)


if __name__ == "__main__":
    unittest.main()
