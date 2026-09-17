import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.planning import (
    ModificationPlanValidationError,
    build_planning_memory_context,
    build_preservation_memory,
    build_repair_memory,
    validate_modification_plan,
    validate_planning_memory_context,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(path: str = "systems/deepread/DeepRead/agent/runner.py") -> dict:
    return {
        "schema_version": "deepread-modification-plan-v1",
        "cohort_id": "old-cohort",
        "hypothesis_set_status": "ready",
        "plans": [
            {
                "plan_id": "plan-old",
                "hypothesis_id": "hyp-old",
                "decision": "proceed",
                "edit_scope": {"allowed_paths": [path]},
                "change_contract": {
                    "current_behavior": "The answer is emitted directly.",
                    "required_behavior_delta": "Verify numeric claims against read evidence.",
                    "must_preserve": ["Existing retrieval behavior."],
                    "non_goals": ["Do not change providers."],
                },
            }
        ],
    }


def _outcome(root: Path, *, outcome: str = "rejected") -> Path:
    plan_path = _write(root / "old-plan.json", _plan())
    value = {
        "schema_version": "deepread-candidate-outcome-v1",
        "decision_id": "outcome-old",
        "candidate_id": "candidate-old",
        "plan_id": "plan-old",
        "outcome": outcome,
        "promotion_status": "not_eligible" if outcome == "rejected" else "eligible_for_materialization",
        "candidate_snapshot_sha256": "a" * 64,
        "cohort_summary": [
            {
                "name": "financebench-development",
                "dataset": "financebench",
                "role": "development",
                "passed": False,
                "mean_delta": -0.1,
                "regressed_task_ids": ["q1"],
                "token_cost_ratio": 1.2,
                "failure_reasons": ["mean_delta"],
            },
            {
                "name": "financebench-holdout",
                "dataset": "financebench",
                "role": "holdout",
                "passed": False,
                "mean_delta": -0.2,
                "regressed_task_ids": ["q9"],
                "token_cost_ratio": 1.4,
                "failure_reasons": ["regression_count"],
            }
        ],
        "artifacts": {
            "modification_plan": {"path": str(plan_path), "sha256": _sha(plan_path)}
        },
    }
    return _write(root / "outcome.json", value)


def _accepted_lineage(root: Path) -> tuple[Path, Path, Path]:
    plan_path = _write(root / "accepted-plan.json", _plan())
    gate_path = _write(
        root / "accepted-gate.json",
        {
            "schema_version": "deepread-validation-gate-v1",
            "passed": True,
            "cohorts": [
                {
                    "name": "financebench-development",
                    "dataset": "financebench",
                    "role": "development",
                    "passed": True,
                    "mean_delta": 0.2,
                    "improved_task_ids": ["q1"],
                    "regressed_task_ids": [],
                    "token_cost_ratio": 1.1,
                },
                {
                    "name": "other-cross",
                    "dataset": "other",
                    "role": "cross_dataset",
                    "passed": True,
                    "mean_delta": 0.0,
                    "improved_task_ids": ["secret-cross-task"],
                    "regressed_task_ids": [],
                    "token_cost_ratio": 1.0,
                },
            ],
        },
    )
    outcome_path = _write(
        root / "accepted-outcome.json",
        {
            "schema_version": "deepread-candidate-outcome-v1",
            "decision_id": "outcome-accepted",
            "candidate_id": "candidate-accepted",
            "plan_id": "plan-old",
            "outcome": "accepted",
            "promotion_status": "eligible_for_materialization",
            "artifacts": {
                "modification_plan": {
                    "path": str(plan_path),
                    "sha256": _sha(plan_path),
                },
                "validation_gate": {
                    "path": str(gate_path),
                    "sha256": _sha(gate_path),
                },
            },
        },
    )
    materialization_id = "materialization-" + hashlib.sha256(
        json.dumps(
            {
                "decision_id": "outcome-accepted",
                "outcome_record_sha256": _sha(outcome_path),
                "materialized_commit": "b" * 40,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    materialization_path = _write(
        root / "materialization.json",
        {
            "schema_version": "deepread-candidate-materialization-v1",
            "materialization_id": materialization_id,
            "decision_id": "outcome-accepted",
            "outcome_record": str(outcome_path),
            "outcome_record_sha256": _sha(outcome_path),
            "materialized_commit": "b" * 40,
            "status": "materialized_detached",
        },
    )
    baseline = {
        "schema_version": "deepread-baseline-entry-v1",
        "generation": 1,
        "commit": "b" * 40,
        "tree": "c" * 40,
        "parent_baseline_id": "baseline-0000-parent",
        "parent_commit": "a" * 40,
        "materialization_id": materialization_id,
        "materialization_path": str(materialization_path),
        "materialization_sha256": _sha(materialization_path),
        "source_repo": str(root / "candidate"),
        "status": "registered",
    }
    basis = {key: value for key, value in baseline.items() if key != "schema_version"}
    basis_sha = hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    baseline["entry_basis_sha256"] = basis_sha
    baseline["baseline_id"] = f"baseline-0001-{basis_sha[:16]}"
    baseline["durable_ref"] = (
        f"refs/agentic-rag-evolve/baselines/{baseline['baseline_id']}"
    )
    baseline_path = _write(root / "baseline.json", baseline)
    return outcome_path, materialization_path, baseline_path


def _cohort() -> dict:
    return {
        "schema_version": "deepread-hypothesis-cohort-v1",
        "cohort_id": "cohort-new",
        "eligible_diagnoses": [
            {
                "task_id": "q1",
                "affected_sources": [
                    {
                        "path": "systems/deepread/DeepRead/agent/runner.py",
                        "symbol": "run_agent",
                    }
                ],
            },
            {
                "task_id": "q2",
                "affected_sources": [
                    {
                        "path": "systems/deepread/DeepRead/agent/runner.py",
                        "symbol": "run_agent",
                    }
                ],
            },
        ],
    }


def _hypotheses() -> dict:
    return {
        "schema_version": "deepread-improvement-hypotheses-v1",
        "cohort_id": "cohort-new",
        "status": "ready",
        "hypotheses": [
            {
                "hypothesis_id": "hyp-new",
                "maturity": "recurring",
                "task_ids": ["q1", "q2"],
                "affected_source_refs": [
                    {"task_id": "q1", "index": 0, "rationale": "Agent loop."}
                ],
            }
        ],
    }


def _candidate_plan(delta: str = "Verify numeric claims against read evidence.") -> dict:
    return {
        "schema_version": "deepread-modification-plan-v1",
        "cohort_id": "cohort-new",
        "hypothesis_set_status": "ready",
        "plans": [
            {
                "hypothesis_id": "hyp-new",
                "decision": "proceed",
                "rationale": "Recurring evidence supports a narrow change.",
                "allowed_source_refs": [
                    {"task_id": "q1", "index": 0, "rationale": "Narrow source."}
                ],
                "change_contract": {
                    "current_behavior": "The answer is emitted directly.",
                    "required_behavior_delta": delta,
                    "must_preserve": ["Existing retrieval behavior."],
                    "non_goals": ["Do not change providers."],
                },
                "validation_plan": {
                    "development_task_ids": ["q1", "q2"],
                    "holdout_selection_rules": ["Use unseen numeric questions."],
                    "expected_observations": ["Claims match read evidence."],
                    "rollback_conditions": ["Holdout accuracy decreases."],
                },
                "risk": {
                    "level": "medium",
                    "regression_scenarios": ["Correct answers may be altered."],
                },
            }
        ],
    }


class RepairMemoryTest(unittest.TestCase):
    def _memory_and_context(self, root: Path) -> tuple[dict, dict]:
        memory, _ = build_repair_memory(
            outcome_path=_outcome(root), memory_root=root / "memory"
        )
        cohort_path = _write(root / "cohort.json", _cohort())
        hypotheses_path = _write(root / "hypotheses.json", _hypotheses())
        context = build_planning_memory_context(
            memory_root=root / "memory",
            cohort_path=cohort_path,
            hypotheses_path=hypotheses_path,
            output_path=root / "context.json",
        )
        return memory, context

    def test_builds_compact_grounded_rejected_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory, path = build_repair_memory(
                outcome_path=_outcome(root), memory_root=root / "memory"
            )

        self.assertEqual(memory["schema_version"], "deepread-repair-memory-v2")
        self.assertEqual(memory["constraints"]["protect_task_ids"], ["q1"])
        self.assertEqual(memory["constraints"]["sealed_regression_count"], 1)
        holdout = next(
            item for item in memory["observed_failures"] if item["role"] == "holdout"
        )
        self.assertEqual(holdout["regressed_task_ids"], [])
        self.assertEqual(holdout["task_id_visibility"], "sealed")
        self.assertEqual(len(memory["attempt"]["fingerprint"]), 64)
        self.assertTrue(path.name.startswith("memory-"))
        self.assertNotIn("trace", memory)

    def test_rejects_accepted_outcome_and_tampered_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "only be built from a rejected"):
                build_repair_memory(
                    outcome_path=_outcome(root, outcome="accepted"),
                    memory_root=root / "memory",
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome_path = _outcome(root)
            plan_path = root / "old-plan.json"
            value = json.loads(plan_path.read_text())
            value["tampered"] = True
            plan_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                build_repair_memory(
                    outcome_path=outcome_path, memory_root=root / "memory"
                )

    def test_memory_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome_path = _outcome(root)
            build_repair_memory(outcome_path=outcome_path, memory_root=root / "memory")
            with self.assertRaises(FileExistsError):
                build_repair_memory(
                    outcome_path=outcome_path, memory_root=root / "memory"
                )

    def test_context_selects_only_exact_source_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory, context = self._memory_and_context(Path(directory))

        failures = context["entries"][0]["relevant_failures"]
        self.assertEqual([item["memory_id"] for item in failures], [memory["memory_id"]])
        self.assertEqual(
            failures[0]["overlapping_paths"],
            ["systems/deepread/DeepRead/agent/runner.py"],
        )
        self.assertFalse(context["selection"]["silent_truncation"])

    def test_context_refuses_silent_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome_path = _outcome(root)
            build_repair_memory(
                outcome_path=outcome_path, memory_root=root / "memory"
            )
            second_outcome = json.loads(outcome_path.read_text())
            second_outcome["decision_id"] = "outcome-second"
            second_outcome["candidate_id"] = "candidate-second"
            second_path = _write(root / "outcome-second.json", second_outcome)
            build_repair_memory(
                outcome_path=second_path, memory_root=root / "memory"
            )
            cohort_path = _write(root / "cohort.json", _cohort())
            hypotheses_path = _write(root / "hypotheses.json", _hypotheses())
            with self.assertRaisesRegex(ValueError, "refusing to truncate"):
                build_planning_memory_context(
                    memory_root=root / "memory",
                    cohort_path=cohort_path,
                    hypotheses_path=hypotheses_path,
                    output_path=root / "context.json",
                    max_entries_per_hypothesis=1,
                )

    def test_validator_blocks_exact_rejected_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, context = self._memory_and_context(Path(directory))

        with self.assertRaisesRegex(
            ModificationPlanValidationError, "exactly repeats a rejected attempt"
        ):
            validate_modification_plan(
                _candidate_plan(),
                cohort=_cohort(),
                hypotheses=_hypotheses(),
                memory_context=context,
            )
        result = validate_modification_plan(
            _candidate_plan("Cross-check numeric claims only after conflicting reads."),
            cohort=_cohort(),
            hypotheses=_hypotheses(),
            memory_context=context,
        )
        self.assertEqual(result["plans"][0]["decision"], "proceed")

    def test_context_validator_rejects_modified_memory_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, context = self._memory_and_context(root)
            tampered = copy.deepcopy(context)
            tampered["entries"][0]["relevant_failures"][0]["constraints"][
                "protect_task_ids"
            ] = []
            cohort_path = root / "cohort.json"
            hypotheses_path = root / "hypotheses.json"
            with self.assertRaisesRegex(ValueError, "projection is inconsistent"):
                validate_planning_memory_context(
                    tampered,
                    cohort=_cohort(),
                    hypotheses=_hypotheses(),
                    cohort_sha256=_sha(cohort_path),
                    hypotheses_sha256=_sha(hypotheses_path),
                )

    def test_builds_accepted_preservation_memory_without_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, materialization, baseline = _accepted_lineage(root)
            memory, path = build_preservation_memory(
                outcome_path=outcome,
                materialization_path=materialization,
                baseline_entry_path=baseline,
                memory_root=root / "memory",
            )

        self.assertEqual(
            memory["schema_version"], "deepread-preservation-memory-v1"
        )
        self.assertEqual(memory["baseline_commit"], "b" * 40)
        self.assertTrue(all(not item["task_ids_exposed"] for item in memory["validated_gains"]))
        self.assertNotIn("secret-cross-task", json.dumps(memory))
        self.assertEqual(path.parent.name, "accepted")

    def test_planning_context_includes_matching_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, materialization, baseline = _accepted_lineage(root)
            preservation, _ = build_preservation_memory(
                outcome_path=outcome,
                materialization_path=materialization,
                baseline_entry_path=baseline,
                memory_root=root / "memory",
            )
            cohort_path = _write(root / "cohort.json", _cohort())
            hypotheses_path = _write(root / "hypotheses.json", _hypotheses())
            context = build_planning_memory_context(
                memory_root=root / "memory",
                cohort_path=cohort_path,
                hypotheses_path=hypotheses_path,
                output_path=root / "context.json",
            )

        successes = context["entries"][0]["relevant_successes"]
        self.assertEqual(
            [item["preservation_id"] for item in successes],
            [preservation["preservation_id"]],
        )
        self.assertNotIn("secret-cross-task", json.dumps(context))

    def test_preservation_memory_rejects_unregistered_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, materialization, baseline_path = _accepted_lineage(root)
            baseline = json.loads(baseline_path.read_text())
            baseline["commit"] = "c" * 40
            baseline_path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "identity is invalid"):
                build_preservation_memory(
                    outcome_path=outcome,
                    materialization_path=materialization,
                    baseline_entry_path=baseline_path,
                    memory_root=root / "memory",
                )


if __name__ == "__main__":
    unittest.main()
