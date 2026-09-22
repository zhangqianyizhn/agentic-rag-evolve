import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.validation import (
    build_regression_feedback,
    evaluate_validation_gate,
)


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
    def _write_inputs(
        self,
        root: Path,
        *,
        costly: bool = False,
        regress: bool = False,
        dev_regress: bool = False,
    ):
        evaluations = {
            "dev-baseline.json": [_item("q1", 0.25)],
            "dev-candidate.json": [
                _item("q1", 0.0 if dev_regress else 0.75, tokens=110)
            ],
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
        candidate_path = root / "candidate"
        candidate_path.mkdir()
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
                    "candidate_path": str(candidate_path),
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )
        test_audit_path = root / "candidate-test-audit.json"
        audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        test_audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-candidate-test-audit-v1",
                    "candidate_audit_sha256": audit_sha256,
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

    def _write_gate(self, root: Path, paths) -> Path:
        gate_path = root / "gate.json"
        gate_path.write_text(
            json.dumps(self._evaluate(paths)), encoding="utf-8"
        )
        return gate_path

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

    def test_fixed_test_audit_must_bind_exact_static_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            test_audit = json.loads(paths[3].read_text())
            test_audit["candidate_audit_sha256"] = "f" * 64
            paths[3].write_text(json.dumps(test_audit))

            with self.assertRaisesRegex(ValueError, "not bound"):
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

    def test_ingestion_plan_requires_run_bound_to_rebuilt_candidate_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            plan = json.loads(paths[2].read_text())
            plan["plans"][0]["edit_scope"] = {"requires_store_rebuild": True}
            paths[2].write_text(json.dumps(plan), encoding="utf-8")
            audit = json.loads(paths[1].read_text())
            audit["plan_sha256"] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
            paths[1].write_text(json.dumps(audit), encoding="utf-8")
            test_audit = json.loads(paths[3].read_text())
            test_audit["plan_sha256"] = audit["plan_sha256"]
            test_audit["candidate_audit_sha256"] = hashlib.sha256(
                paths[1].read_bytes()
            ).hexdigest()
            paths[3].write_text(json.dumps(test_audit), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires candidate_run_manifest"):
                self._evaluate(paths)

            store_manifest = root / "STORE_MANIFEST.json"
            store_manifest.write_text(
                json.dumps({"schema_version": "deepread-store-build-v1"}),
                encoding="utf-8",
            )
            run_manifest = root / "candidate-run.json"
            candidate_audit_sha256 = hashlib.sha256(paths[1].read_bytes()).hexdigest()
            candidate_evaluation = json.loads((root / "dev-candidate.json").read_text())
            candidate_evaluation[0]["run_id"] = "candidate-run-1"
            (root / "dev-candidate.json").write_text(json.dumps(candidate_evaluation))
            holdout_evaluation = json.loads((root / "hold-candidate.json").read_text())
            holdout_evaluation[0]["run_id"] = "candidate-run-1"
            (root / "hold-candidate.json").write_text(json.dumps(holdout_evaluation))
            run_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "candidate-run-1",
                        "task_ids": ["q1", "q2"],
                        "candidate_source": {
                            "candidate_snapshot_sha256": "c" * 64,
                        },
                        "store_build": {
                            "path": str(store_manifest),
                            "sha256": hashlib.sha256(store_manifest.read_bytes()).hexdigest(),
                            "candidate": {
                                "candidate_snapshot_sha256": "c" * 64,
                                "candidate_audit_sha256": candidate_audit_sha256,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            suite = json.loads(paths[0].read_text())
            for cohort in suite["cohorts"]:
                cohort["candidate_run_manifest"] = "candidate-run.json"
            paths[0].write_text(json.dumps(suite), encoding="utf-8")

            result = self._evaluate(paths)
            self.assertTrue(result["passed"])
            self.assertTrue(result["requires_store_rebuild"])

    def test_regression_feedback_exposes_development_tasks_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root, regress=True, dev_regress=True)
            gate_path = self._write_gate(root, paths)
            feedback = build_regression_feedback(
                gate_path=gate_path,
                suite_path=paths[0],
                candidate_audit_path=paths[1],
                output_path=root / "feedback.json",
            )

            self.assertEqual(feedback["status"], "ready")
            self.assertEqual(feedback["diagnosis_scope"]["task_ids"], ["q1"])
            self.assertEqual(
                feedback["sealed_cohort_summaries"][0]["regressed_task_count"], 1
            )
            self.assertNotIn(
                "regressed_task_ids", feedback["sealed_cohort_summaries"][0]
            )
            self.assertNotIn("baseline_evaluation", feedback["sealed_cohort_summaries"][0])
            self.assertFalse(feedback["final_test_accessed"])

    def test_feedback_does_not_promote_holdout_regression_to_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root, regress=True)
            gate_path = self._write_gate(root, paths)
            feedback = build_regression_feedback(
                gate_path=gate_path,
                suite_path=paths[0],
                candidate_audit_path=paths[1],
                output_path=root / "feedback.json",
            )

            self.assertEqual(feedback["status"], "no_development_regressions")
            self.assertEqual(feedback["diagnosis_scope"]["task_ids"], [])
            self.assertNotIn("q2", json.dumps(feedback))

    def test_feedback_rejects_tampered_validation_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root, dev_regress=True)
            gate_path = self._write_gate(root, paths)
            suite = json.loads(paths[0].read_text())
            suite["comparison_epsilon"] = 0.5
            paths[0].write_text(json.dumps(suite), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "suite hash"):
                build_regression_feedback(
                    gate_path=gate_path,
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    output_path=root / "feedback.json",
                )

    def test_feedback_rejects_evaluation_changed_after_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root, dev_regress=True)
            gate_path = self._write_gate(root, paths)
            evaluation_path = root / "dev-candidate.json"
            evaluation = json.loads(evaluation_path.read_text())
            evaluation[0]["metrics"]["accuracy_normalized"] = 0.1
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "evaluation hash"):
                build_regression_feedback(
                    gate_path=gate_path,
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    output_path=root / "feedback.json",
                )

    def test_feedback_is_deterministic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root, dev_regress=True)
            gate_path = self._write_gate(root, paths)
            first_path = root / "feedback-1.json"
            first = build_regression_feedback(
                gate_path=gate_path,
                suite_path=paths[0],
                candidate_audit_path=paths[1],
                output_path=first_path,
            )
            second = build_regression_feedback(
                gate_path=gate_path,
                suite_path=paths[0],
                candidate_audit_path=paths[1],
                output_path=root / "feedback-2.json",
            )

            self.assertEqual(first["feedback_id"], second["feedback_id"])
            with self.assertRaises(FileExistsError):
                build_regression_feedback(
                    gate_path=gate_path,
                    suite_path=paths[0],
                    candidate_audit_path=paths[1],
                    output_path=first_path,
                )


if __name__ == "__main__":
    unittest.main()
