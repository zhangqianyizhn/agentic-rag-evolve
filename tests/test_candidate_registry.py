import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import record_candidate_outcome


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class CandidateRegistryTest(unittest.TestCase):
    def _fixture(self, root: Path, *, passed: bool = True, level: str = "promotion"):
        plan_path = _write(
            root / "plan.json",
            {
                "schema_version": "deepread-modification-plan-v1",
                "plans": [{"plan_id": "plan-1", "decision": "proceed"}],
            },
        )
        manifest_path = _write(
            root / "manifest.json",
            {
                "schema_version": "deepread-candidate-manifest-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan_path),
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "base_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
            },
        )
        modification_path = _write(
            root / "modification.json",
            {
                "schema_version": "deepread-candidate-modification-v1",
                "candidate_manifest_sha256": _sha(manifest_path),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan_path),
                "base_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
                "candidate_snapshot_sha256": "c" * 64,
                "status": "modified",
            },
        )
        audit_path = _write(
            root / "audit.json",
            {
                "schema_version": "deepread-candidate-audit-v1",
                "candidate_manifest_sha256": _sha(manifest_path),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan_path),
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "base_commit": "a" * 40,
                "head_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
                "candidate_snapshot_sha256": "c" * 64,
                "modification_sha256": _sha(modification_path),
                "passed": True,
            },
        )
        test_audit_path = _write(
            root / "test-audit.json",
            {
                "schema_version": "deepread-candidate-test-audit-v1",
                "candidate_audit_sha256": _sha(audit_path),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan_path),
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "candidate_snapshot_sha256": "c" * 64,
                "passed": True,
            },
        )
        suite_path = _write(
            root / "suite.json",
            {
                "schema_version": "deepread-validation-suite-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "candidate_snapshot_sha256": "c" * 64,
                "gate_level": level,
            },
        )
        failure_reasons = [] if passed else ["regression_count"]
        holdout = {
            "name": "financebench-holdout",
            "dataset": "financebench",
            "role": "holdout",
            "passed": passed,
            "mean_delta": 0.0 if passed else -0.2,
            "regressed_task_ids": [] if passed else ["q2"],
            "token_cost_ratio": 1.1,
            "failure_reasons": failure_reasons,
        }
        cohorts = [
            {
                "name": "financebench-development",
                "dataset": "financebench",
                "role": "development",
                "passed": True,
                "mean_delta": 0.2,
                "regressed_task_ids": [],
                "token_cost_ratio": 1.1,
                "failure_reasons": [],
            },
            holdout,
            {
                "name": "other-cross",
                "dataset": "other",
                "role": "cross_dataset",
                "passed": True,
                "mean_delta": 0.0,
                "regressed_task_ids": [],
                "token_cost_ratio": 1.0,
                "failure_reasons": [],
            },
        ]
        gate_path = _write(
            root / "gate.json",
            {
                "schema_version": "deepread-validation-gate-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "gate_level": level,
                "candidate_snapshot_sha256": "c" * 64,
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "plan_sha256": _sha(plan_path),
                "validation_suite_sha256": _sha(suite_path),
                "candidate_audit_sha256": _sha(audit_path),
                "candidate_test_audit_sha256": _sha(test_audit_path),
                "passed": passed,
                "failed_cohorts": [] if passed else ["financebench-holdout"],
                "cohorts": cohorts,
            },
        )
        return {
            "registry_root": root / "registry",
            "candidate_manifest_path": manifest_path,
            "plan_path": plan_path,
            "candidate_modification_path": modification_path,
            "candidate_audit_path": audit_path,
            "candidate_test_audit_path": test_audit_path,
            "validation_suite_path": suite_path,
            "validation_gate_path": gate_path,
        }

    def test_records_accepted_candidate_without_git_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            record, path = record_candidate_outcome(**inputs)

            self.assertEqual(path.parent.name, "accepted")
            self.assertTrue(path.exists())

        self.assertEqual(record["outcome"], "accepted")
        self.assertEqual(record["promotion_status"], "eligible_for_materialization")
        self.assertFalse(record["git_mutation_performed"])
        self.assertEqual(record["decision_reasons"], ["promotion_gate_passed"])

    def test_records_rejected_candidate_with_gate_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory), passed=False)
            record, path = record_candidate_outcome(**inputs)

        self.assertEqual(path.parent.name, "rejected")
        self.assertEqual(record["promotion_status"], "not_eligible")
        self.assertEqual(
            record["decision_reasons"],
            ["cohort:financebench-holdout:regression_count"],
        )
        self.assertEqual(record["next_action"], "retain_for_failure_memory")

    def test_development_gate_is_not_a_terminal_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory), level="development")
            with self.assertRaisesRegex(ValueError, "promotion-level"):
                record_candidate_outcome(**inputs)

    def test_candidate_snapshot_can_only_have_one_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            record_candidate_outcome(**inputs)
            with self.assertRaisesRegex(FileExistsError, "already has an outcome"):
                record_candidate_outcome(**inputs)

    def test_rejects_tampered_upstream_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._fixture(root)
            plan = json.loads(inputs["plan_path"].read_text())
            plan["tampered"] = True
            inputs["plan_path"].write_text(json.dumps(plan))

            with self.assertRaisesRegex(ValueError, "plan hash"):
                record_candidate_outcome(**inputs)

    def test_rejects_internally_inconsistent_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory), passed=False)
            gate = json.loads(inputs["validation_gate_path"].read_text())
            gate["failed_cohorts"] = []
            inputs["validation_gate_path"].write_text(json.dumps(gate))

            with self.assertRaisesRegex(ValueError, "internally inconsistent"):
                record_candidate_outcome(**inputs)


if __name__ == "__main__":
    unittest.main()
