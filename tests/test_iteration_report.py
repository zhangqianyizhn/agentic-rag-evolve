import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import record_candidate_outcome
from agentic_rag_evolve.planning import (
    build_preservation_memory,
    build_repair_memory,
)
from agentic_rag_evolve.reporting import build_iteration_report


SOURCE = "systems/deepread/DeepRead/agent/runner.py"


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class IterationReportTest(unittest.TestCase):
    def _fixture(self, root: Path, *, accepted: bool):
        cohort = _write(
            root / "cohort.json",
            {
                "schema_version": "deepread-hypothesis-cohort-v1",
                "cohort_id": "cohort-1",
                "eligible_diagnoses": [
                    {"task_id": "dev-secret", "affected_sources": []},
                    {"task_id": "dev-other", "affected_sources": []},
                ],
                "excluded_diagnoses": [
                    {"task_id": "excluded-secret", "route": "evidence_review"}
                ],
            },
        )
        hypotheses = _write(
            root / "hypotheses.json",
            {
                "schema_version": "deepread-improvement-hypotheses-v1",
                "cohort_id": "cohort-1",
                "status": "ready",
                "hypotheses": [
                    {
                        "hypothesis_id": "hyp-1",
                        "maturity": "recurring",
                        "task_ids": ["dev-secret", "dev-other"],
                    }
                ],
            },
        )
        plan = _write(
            root / "plan.json",
            {
                "schema_version": "deepread-modification-plan-v1",
                "cohort_id": "cohort-1",
                "hypothesis_set_status": "ready",
                "plans": [
                    {
                        "plan_id": "plan-1",
                        "hypothesis_id": "hyp-1",
                        "decision": "proceed",
                        "edit_scope": {"allowed_paths": [SOURCE]},
                        "change_contract": {
                            "current_behavior": "The answer is emitted directly.",
                            "required_behavior_delta": "Verify values against read evidence.",
                            "must_preserve": ["Existing retrieval behavior."],
                            "non_goals": ["Do not change providers."],
                        },
                        "risk": {
                            "level": "medium",
                            "regression_scenarios": ["Correct answers may change."],
                        },
                    }
                ],
            },
        )
        manifest = _write(
            root / "manifest.json",
            {
                "schema_version": "deepread-candidate-manifest-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan),
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "base_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
            },
        )
        modification = _write(
            root / "modification.json",
            {
                "schema_version": "deepread-candidate-modification-v1",
                "candidate_manifest_sha256": _sha(manifest),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan),
                "base_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
                "candidate_snapshot_sha256": "c" * 64,
                "status": "modified",
            },
        )
        audit = _write(
            root / "audit.json",
            {
                "schema_version": "deepread-candidate-audit-v1",
                "candidate_manifest_sha256": _sha(manifest),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan),
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "base_commit": "a" * 40,
                "head_commit": "a" * 40,
                "candidate_path": str(root / "candidate"),
                "candidate_snapshot_sha256": "c" * 64,
                "modification_sha256": _sha(modification),
                "changed_paths": [SOURCE],
                "changed_file_count": 1,
                "passed": True,
            },
        )
        test_audit = _write(
            root / "test-audit.json",
            {
                "schema_version": "deepread-candidate-test-audit-v1",
                "candidate_audit_sha256": _sha(audit),
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "plan_sha256": _sha(plan),
                "candidate_snapshot_sha256": "c" * 64,
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "passed": True,
                "checks": [{"tests_run": 154, "passed": True}],
            },
        )
        suite = _write(
            root / "suite.json",
            {
                "schema_version": "deepread-validation-suite-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "candidate_snapshot_sha256": "c" * 64,
                "gate_level": "promotion",
            },
        )
        holdout_passed = accepted
        cohorts = [
            {
                "name": "financebench-development",
                "dataset": "financebench",
                "role": "development",
                "passed": True,
                "task_count": 2,
                "mean_delta": 0.2,
                "improved_task_ids": ["dev-secret"],
                "regressed_task_ids": [],
                "token_cost_ratio": 1.1,
                "failure_reasons": [],
            },
            {
                "name": "financebench-holdout",
                "dataset": "financebench",
                "role": "holdout",
                "passed": holdout_passed,
                "task_count": 1,
                "mean_delta": 0.0 if accepted else -0.2,
                "improved_task_ids": [],
                "regressed_task_ids": [] if accepted else ["holdout-secret"],
                "token_cost_ratio": 1.0,
                "failure_reasons": [] if accepted else ["regression_count"],
            },
            {
                "name": "other-cross",
                "dataset": "other",
                "role": "cross_dataset",
                "passed": True,
                "task_count": 1,
                "mean_delta": 0.0,
                "improved_task_ids": ["cross-secret"],
                "regressed_task_ids": [],
                "token_cost_ratio": 1.0,
                "failure_reasons": [],
            },
        ]
        gate = _write(
            root / "gate.json",
            {
                "schema_version": "deepread-validation-gate-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "gate_level": "promotion",
                "candidate_snapshot_sha256": "c" * 64,
                "test_policy_id": "unit-v1",
                "test_policy_sha256": "d" * 64,
                "plan_sha256": _sha(plan),
                "validation_suite_sha256": _sha(suite),
                "candidate_audit_sha256": _sha(audit),
                "candidate_test_audit_sha256": _sha(test_audit),
                "passed": accepted,
                "failed_cohorts": [] if accepted else ["financebench-holdout"],
                "cohorts": cohorts,
            },
        )
        outcome, outcome_path = record_candidate_outcome(
            registry_root=root / "outcomes",
            candidate_manifest_path=manifest,
            plan_path=plan,
            candidate_modification_path=modification,
            candidate_audit_path=audit,
            candidate_test_audit_path=test_audit,
            validation_suite_path=suite,
            validation_gate_path=gate,
        )
        return {
            "cohort": cohort,
            "hypotheses": hypotheses,
            "gate": gate,
            "outcome": outcome,
            "outcome_path": outcome_path,
        }

    def _accepted_memory(self, root: Path, fixture: dict) -> Path:
        outcome_path = fixture["outcome_path"]
        outcome = fixture["outcome"]
        outcome_sha = _sha(outcome_path)
        materialized_commit = "b" * 40
        materialization_id = "materialization-" + _canonical(
            {
                "decision_id": outcome["decision_id"],
                "outcome_record_sha256": outcome_sha,
                "materialized_commit": materialized_commit,
            }
        )[:20]
        materialization = _write(
            root / "materialization.json",
            {
                "schema_version": "deepread-candidate-materialization-v1",
                "materialization_id": materialization_id,
                "decision_id": outcome["decision_id"],
                "outcome_record": str(outcome_path),
                "outcome_record_sha256": outcome_sha,
                "materialized_commit": materialized_commit,
                "status": "materialized_detached",
            },
        )
        baseline = {
            "schema_version": "deepread-baseline-entry-v1",
            "generation": 1,
            "commit": materialized_commit,
            "tree": "e" * 40,
            "parent_baseline_id": "baseline-0000-parent",
            "parent_commit": "a" * 40,
            "materialization_id": materialization_id,
            "materialization_path": str(materialization),
            "materialization_sha256": _sha(materialization),
            "source_repo": str(root / "candidate"),
            "status": "registered",
        }
        basis = {
            key: value for key, value in baseline.items() if key != "schema_version"
        }
        digest = _canonical(basis)
        baseline["entry_basis_sha256"] = digest
        baseline["baseline_id"] = f"baseline-0001-{digest[:16]}"
        baseline["durable_ref"] = (
            f"refs/agentic-rag-evolve/baselines/{baseline['baseline_id']}"
        )
        baseline_path = _write(root / "baseline.json", baseline)
        _, memory_path = build_preservation_memory(
            outcome_path=outcome_path,
            materialization_path=materialization,
            baseline_entry_path=baseline_path,
            memory_root=root / "memory",
        )
        return memory_path

    def _feedback(self, root: Path, fixture: dict) -> Path:
        outcome = fixture["outcome"]
        task_ids = ["dev-secret"]
        gate_sha = _sha(fixture["gate"])
        basis = {
            "validation_gate_sha256": gate_sha,
            "candidate_snapshot_sha256": outcome["candidate_snapshot_sha256"],
            "development_regressed_task_ids": task_ids,
        }
        return _write(
            root / "feedback.json",
            {
                "schema_version": "deepread-regression-feedback-v1",
                "feedback_id": "feedback-" + _canonical(basis)[:20],
                "candidate_id": outcome["candidate_id"],
                "plan_id": outcome["plan_id"],
                "candidate_snapshot_sha256": outcome["candidate_snapshot_sha256"],
                "status": "ready",
                "diagnosis_scope": {
                    "allowed_roles": ["development"],
                    "task_ids": task_ids,
                },
                "development_cohorts": [{"role": "development"}],
                "sealed_cohort_summaries": [
                    {
                        "role": "holdout",
                        "regressed_task_count": 1,
                        "task_id_visibility": "sealed",
                    }
                ],
                "artifacts": {
                    "validation_gate": {
                        "path": str(fixture["gate"]),
                        "sha256": gate_sha,
                    }
                },
                "final_test_accessed": False,
            },
        )

    def test_rejected_report_is_compact_and_hides_all_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, accepted=False)
            _, memory = build_repair_memory(
                outcome_path=fixture["outcome_path"], memory_root=root / "memory"
            )
            report = build_iteration_report(
                outcome_path=fixture["outcome_path"],
                terminal_memory_path=memory,
                cohort_path=fixture["cohort"],
                hypotheses_path=fixture["hypotheses"],
                regression_feedback_path=self._feedback(root, fixture),
                output_path=root / "report.json",
            )

        serialized = json.dumps(report)
        self.assertEqual(report["terminal"]["status"], "rejected_retained")
        self.assertFalse(report["terminal"]["baseline_changed"])
        self.assertEqual(report["candidate"]["fixed_test_count"], 154)
        self.assertEqual(report["validation"]["cohorts"][1]["regressed_task_count"], 1)
        self.assertEqual(
            report["regression_feedback"]["development_regression_count"], 1
        )
        for secret in (
            "dev-secret", "dev-other", "excluded-secret",
            "holdout-secret", "cross-secret",
        ):
            self.assertNotIn(secret, serialized)
        for copied_payload in ("trajectory", "gold_evidence", "stdout_tail"):
            self.assertNotIn(copied_payload, serialized)
        self.assertFalse(report["large_payloads_embedded"])
        self.assertFalse(report["final_test_accessed"])

    def test_accepted_report_requires_registered_preservation_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, accepted=True)
            memory = self._accepted_memory(root, fixture)
            report = build_iteration_report(
                outcome_path=fixture["outcome_path"],
                terminal_memory_path=memory,
                cohort_path=fixture["cohort"],
                hypotheses_path=fixture["hypotheses"],
                output_path=root / "report.json",
            )

        self.assertEqual(report["terminal"]["status"], "accepted_registered")
        self.assertTrue(report["terminal"]["baseline_changed"])
        self.assertEqual(report["terminal"]["baseline"]["commit"], "b" * 40)
        self.assertEqual(report["decision"]["outcome"], "accepted")

    def test_report_rejects_upstream_artifact_changed_after_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, accepted=False)
            _, memory = build_repair_memory(
                outcome_path=fixture["outcome_path"], memory_root=root / "memory"
            )
            gate = json.loads(fixture["gate"].read_text())
            gate["tampered"] = True
            fixture["gate"].write_text(json.dumps(gate), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "gate artifact hash"):
                build_iteration_report(
                    outcome_path=fixture["outcome_path"],
                    terminal_memory_path=memory,
                    cohort_path=fixture["cohort"],
                    hypotheses_path=fixture["hypotheses"],
                    output_path=root / "report.json",
                )

    def test_report_identity_is_deterministic_and_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root, accepted=False)
            _, memory = build_repair_memory(
                outcome_path=fixture["outcome_path"], memory_root=root / "memory"
            )
            arguments = {
                "outcome_path": fixture["outcome_path"],
                "terminal_memory_path": memory,
                "cohort_path": fixture["cohort"],
                "hypotheses_path": fixture["hypotheses"],
            }
            first = build_iteration_report(
                **arguments, output_path=root / "report-1.json"
            )
            second = build_iteration_report(
                **arguments, output_path=root / "report-2.json"
            )

            self.assertEqual(first["iteration_id"], second["iteration_id"])
            with self.assertRaises(FileExistsError):
                build_iteration_report(
                    **arguments, output_path=root / "report-1.json"
                )


if __name__ == "__main__":
    unittest.main()
