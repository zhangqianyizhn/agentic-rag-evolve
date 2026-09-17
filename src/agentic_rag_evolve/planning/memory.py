"""Grounded, compact memory derived from rejected candidate outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MEMORY_SCHEMA = "deepread-repair-memory-v2"
PRESERVATION_MEMORY_SCHEMA = "deepread-preservation-memory-v1"
CONTEXT_SCHEMA = "deepread-planning-memory-context-v2"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def attempt_fingerprint(*, allowed_paths: list[str], required_behavior_delta: str) -> str:
    """Identify a semantic plan attempt without depending on candidate IDs."""

    basis = {
        "allowed_paths": sorted(set(allowed_paths)),
        "required_behavior_delta": " ".join(required_behavior_delta.split()),
    }
    return _canonical_sha256(basis)


def _failed_cohort_projection(outcome: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for raw in outcome.get("cohort_summary") or []:
        if not isinstance(raw, Mapping) or raw.get("passed"):
            continue
        role = str(raw.get("role") or "")
        task_ids = sorted(
            {str(item) for item in raw.get("regressed_task_ids") or []}
        )
        task_feedback_allowed = role == "development"
        failures.append(
            {
                "name": str(raw.get("name") or ""),
                "dataset": str(raw.get("dataset") or ""),
                "role": role,
                "mean_delta": raw.get("mean_delta"),
                "regressed_task_count": len(task_ids),
                "regressed_task_ids": task_ids if task_feedback_allowed else [],
                "task_id_visibility": (
                    "feedback_allowed" if task_feedback_allowed else "sealed"
                ),
                "token_cost_ratio": raw.get("token_cost_ratio"),
                "failure_reasons": sorted(
                    {str(item) for item in raw.get("failure_reasons") or []}
                ),
            }
        )
    return failures


def build_repair_memory(
    *, outcome_path: Path, memory_root: Path
) -> tuple[dict[str, Any], Path]:
    """Persist one immutable failure-memory entry after verifying its lineage."""

    outcome_path = Path(outcome_path).resolve()
    outcome, outcome_sha = _read_object(outcome_path, "candidate outcome")
    if outcome.get("schema_version") != "deepread-candidate-outcome-v1":
        raise ValueError("unsupported candidate outcome schema")
    if outcome.get("outcome") != "rejected":
        raise ValueError("repair memory can only be built from a rejected outcome")
    if outcome.get("promotion_status") != "not_eligible":
        raise ValueError("rejected outcome has inconsistent promotion status")

    artifacts = outcome.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("candidate outcome has no artifact lineage")
    plan_ref = artifacts.get("modification_plan")
    if not isinstance(plan_ref, Mapping):
        raise ValueError("candidate outcome has no modification plan reference")
    plan_path = Path(str(plan_ref.get("path") or "")).resolve()
    plan, plan_sha = _read_object(plan_path, "modification plan")
    if plan_sha != plan_ref.get("sha256"):
        raise ValueError("modification plan hash does not match outcome lineage")
    if plan.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")

    plan_id = str(outcome.get("plan_id") or "")
    plans = [item for item in plan.get("plans") or [] if item.get("plan_id") == plan_id]
    if len(plans) != 1 or plans[0].get("decision") != "proceed":
        raise ValueError("rejected outcome must reference one proceeding plan")
    selected = plans[0]
    scope = selected.get("edit_scope")
    contract = selected.get("change_contract")
    if not isinstance(scope, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("referenced plan is missing edit scope or change contract")
    paths = sorted({str(item) for item in scope.get("allowed_paths") or [] if item})
    delta = str(contract.get("required_behavior_delta") or "").strip()
    if not paths or not delta:
        raise ValueError("referenced plan has no attempted scope or behavior delta")

    failures = _failed_cohort_projection(outcome)
    if not failures:
        raise ValueError("rejected outcome has no failed cohort summary")

    fingerprint = attempt_fingerprint(
        allowed_paths=paths, required_behavior_delta=delta
    )
    source_decision_id = str(outcome.get("decision_id") or "")
    memory_id = "memory-" + _canonical_sha256(
        {"source_decision_id": source_decision_id, "attempt_fingerprint": fingerprint}
    )[:20]
    record = {
        "schema_version": MEMORY_SCHEMA,
        "memory_id": memory_id,
        "source_decision_id": source_decision_id,
        "candidate_id": str(outcome.get("candidate_id") or ""),
        "candidate_snapshot_sha256": str(
            outcome.get("candidate_snapshot_sha256") or ""
        ),
        "attempt": {
            "plan_id": plan_id,
            "hypothesis_id": str(selected.get("hypothesis_id") or ""),
            "allowed_paths": paths,
            "required_behavior_delta": delta,
            "must_preserve": list(contract.get("must_preserve") or []),
            "non_goals": list(contract.get("non_goals") or []),
            "fingerprint": fingerprint,
        },
        "observed_failures": failures,
        "constraints": {
            "avoid_exact_repeat": True,
            "requires_changed_strategy": True,
            "protect_task_ids": sorted(
                {
                    task_id
                    for failure in failures
                    for task_id in failure["regressed_task_ids"]
                }
            ),
            "sealed_regression_count": sum(
                failure["regressed_task_count"]
                for failure in failures
                if failure["task_id_visibility"] == "sealed"
            ),
        },
        "artifacts": {
            "candidate_outcome": {"path": str(outcome_path), "sha256": outcome_sha},
            "modification_plan": {"path": str(plan_path), "sha256": plan_sha},
        },
    }
    if not all(
        [
            record["source_decision_id"],
            record["candidate_id"],
            record["candidate_snapshot_sha256"],
            record["attempt"]["hypothesis_id"],
        ]
    ):
        raise ValueError("rejected outcome lineage is incomplete")

    destination = Path(memory_root).resolve() / "rejected" / f"{memory_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record, destination


def _validate_repair_memory_record(memory: Mapping[str, Any]) -> None:
    """Re-derive a memory record from its pinned outcome and plan artifacts."""

    if memory.get("schema_version") != MEMORY_SCHEMA:
        raise ValueError("unsupported repair memory schema")
    artifacts = memory.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("repair memory has no artifact lineage")
    outcome_ref = artifacts.get("candidate_outcome")
    plan_ref = artifacts.get("modification_plan")
    if not isinstance(outcome_ref, Mapping) or not isinstance(plan_ref, Mapping):
        raise ValueError("repair memory artifact lineage is incomplete")
    outcome, outcome_sha = _read_object(
        Path(str(outcome_ref.get("path") or "")), "candidate outcome"
    )
    plan, plan_sha = _read_object(
        Path(str(plan_ref.get("path") or "")), "modification plan"
    )
    if outcome_sha != outcome_ref.get("sha256") or plan_sha != plan_ref.get("sha256"):
        raise ValueError("repair memory artifact hash does not match lineage")
    if outcome.get("schema_version") != "deepread-candidate-outcome-v1" or outcome.get(
        "outcome"
    ) != "rejected":
        raise ValueError("repair memory source is not a rejected candidate outcome")
    outcome_plan_ref = (outcome.get("artifacts") or {}).get("modification_plan") or {}
    if outcome_plan_ref.get("sha256") != plan_sha:
        raise ValueError("repair memory plan is not pinned by its outcome")
    plan_id = str(outcome.get("plan_id") or "")
    plans = [item for item in plan.get("plans") or [] if item.get("plan_id") == plan_id]
    if len(plans) != 1 or plans[0].get("decision") != "proceed":
        raise ValueError("repair memory source plan is not a proceeding plan")
    selected = plans[0]
    contract = selected.get("change_contract") or {}
    paths = sorted(set((selected.get("edit_scope") or {}).get("allowed_paths") or []))
    delta = str(contract.get("required_behavior_delta") or "").strip()
    fingerprint = attempt_fingerprint(
        allowed_paths=paths, required_behavior_delta=delta
    )
    expected_attempt = {
        "plan_id": plan_id,
        "hypothesis_id": str(selected.get("hypothesis_id") or ""),
        "allowed_paths": paths,
        "required_behavior_delta": delta,
        "must_preserve": list(contract.get("must_preserve") or []),
        "non_goals": list(contract.get("non_goals") or []),
        "fingerprint": fingerprint,
    }
    source_decision_id = str(outcome.get("decision_id") or "")
    expected_id = "memory-" + _canonical_sha256(
        {"source_decision_id": source_decision_id, "attempt_fingerprint": fingerprint}
    )[:20]
    expected_failures = _failed_cohort_projection(outcome)
    if memory.get("memory_id") != expected_id or memory.get("attempt") != expected_attempt:
        raise ValueError("repair memory attempt projection is inconsistent")
    if memory.get("observed_failures") != expected_failures:
        raise ValueError("repair memory failure projection is inconsistent")
    expected_constraints = {
        "avoid_exact_repeat": True,
        "requires_changed_strategy": True,
        "protect_task_ids": sorted(
            {
                task_id
                for failure in expected_failures
                for task_id in failure["regressed_task_ids"]
            }
        ),
        "sealed_regression_count": sum(
            failure["regressed_task_count"]
            for failure in expected_failures
            if failure["task_id_visibility"] == "sealed"
        ),
    }
    if memory.get("constraints") != expected_constraints:
        raise ValueError("repair memory constraints are inconsistent")
    if (
        memory.get("source_decision_id") != source_decision_id
        or memory.get("candidate_id") != outcome.get("candidate_id")
        or memory.get("candidate_snapshot_sha256")
        != outcome.get("candidate_snapshot_sha256")
    ):
        raise ValueError("repair memory candidate lineage is inconsistent")


def _pinned_artifact(
    owner: Mapping[str, Any], key: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    reference = (owner.get("artifacts") or {}).get(key)
    if not isinstance(reference, Mapping):
        raise ValueError(f"{label} has no {key} artifact reference")
    path = Path(str(reference.get("path") or "")).resolve()
    value, digest = _read_object(path, key.replace("_", " "))
    if digest != reference.get("sha256"):
        raise ValueError(f"{key} artifact hash does not match {label} lineage")
    return value, path, digest


def _validate_baseline_entry_identity(baseline: Mapping[str, Any]) -> None:
    try:
        generation = int(baseline.get("generation"))
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline entry has no valid generation") from exc
    basis = {
        "generation": generation,
        "commit": baseline.get("commit"),
        "tree": baseline.get("tree"),
        "parent_baseline_id": baseline.get("parent_baseline_id"),
        "parent_commit": baseline.get("parent_commit"),
        "materialization_id": baseline.get("materialization_id"),
        "materialization_path": baseline.get("materialization_path"),
        "materialization_sha256": baseline.get("materialization_sha256"),
        "source_repo": baseline.get("source_repo"),
        "status": baseline.get("status"),
    }
    digest = _canonical_sha256(basis)
    baseline_id = f"baseline-{generation:04d}-{digest[:16]}"
    if (
        generation < 1
        or baseline.get("entry_basis_sha256") != digest
        or baseline.get("baseline_id") != baseline_id
        or baseline.get("durable_ref")
        != f"refs/agentic-rag-evolve/baselines/{baseline_id}"
    ):
        raise ValueError("baseline entry identity is invalid")


def _derive_preservation_memory(
    *, outcome_path: Path, materialization_path: Path, baseline_entry_path: Path
) -> dict[str, Any]:
    outcome_path = Path(outcome_path).resolve()
    materialization_path = Path(materialization_path).resolve()
    baseline_entry_path = Path(baseline_entry_path).resolve()
    outcome, outcome_sha = _read_object(outcome_path, "candidate outcome")
    materialization, materialization_sha = _read_object(
        materialization_path, "candidate materialization"
    )
    baseline, baseline_sha = _read_object(baseline_entry_path, "baseline entry")
    if (
        outcome.get("schema_version") != "deepread-candidate-outcome-v1"
        or outcome.get("outcome") != "accepted"
        or outcome.get("promotion_status") != "eligible_for_materialization"
    ):
        raise ValueError("preservation memory requires an accepted candidate outcome")
    if (
        materialization.get("schema_version")
        != "deepread-candidate-materialization-v1"
        or materialization.get("status") != "materialized_detached"
    ):
        raise ValueError("preservation memory requires a completed materialization")
    expected_materialization_id = "materialization-" + _canonical_sha256(
        {
            "decision_id": materialization.get("decision_id"),
            "outcome_record_sha256": materialization.get("outcome_record_sha256"),
            "materialized_commit": materialization.get("materialized_commit"),
        }
    )[:20]
    if materialization.get("materialization_id") != expected_materialization_id:
        raise ValueError("candidate materialization identity is invalid")
    if (
        materialization.get("outcome_record_sha256") != outcome_sha
        or Path(str(materialization.get("outcome_record") or "")).resolve()
        != outcome_path
        or materialization.get("decision_id") != outcome.get("decision_id")
    ):
        raise ValueError("materialization does not match accepted outcome")
    if baseline.get("schema_version") != "deepread-baseline-entry-v1":
        raise ValueError("unsupported baseline entry schema")
    _validate_baseline_entry_identity(baseline)
    if (
        baseline.get("materialization_id") != materialization.get("materialization_id")
        or baseline.get("materialization_sha256") != materialization_sha
        or Path(str(baseline.get("materialization_path") or "")).resolve()
        != materialization_path
        or baseline.get("commit") != materialization.get("materialized_commit")
    ):
        raise ValueError("baseline entry does not match candidate materialization")

    plan, plan_path, plan_sha = _pinned_artifact(
        outcome, "modification_plan", "candidate outcome"
    )
    gate, gate_path, gate_sha = _pinned_artifact(
        outcome, "validation_gate", "candidate outcome"
    )
    if gate.get("schema_version") != "deepread-validation-gate-v1" or not gate.get(
        "passed"
    ):
        raise ValueError("accepted outcome does not reference a passing validation gate")
    plan_id = str(outcome.get("plan_id") or "")
    plans = [item for item in plan.get("plans") or [] if item.get("plan_id") == plan_id]
    if len(plans) != 1 or plans[0].get("decision") != "proceed":
        raise ValueError("accepted outcome must reference one proceeding plan")
    selected = plans[0]
    contract = selected.get("change_contract") or {}
    paths = sorted(set((selected.get("edit_scope") or {}).get("allowed_paths") or []))
    delta = str(contract.get("required_behavior_delta") or "").strip()
    if not paths or not delta:
        raise ValueError("accepted plan has no scope or behavior delta")
    fingerprint = attempt_fingerprint(
        allowed_paths=paths, required_behavior_delta=delta
    )
    gains = []
    for cohort in gate.get("cohorts") or []:
        if not isinstance(cohort, Mapping) or not cohort.get("passed"):
            raise ValueError("accepted validation gate contains a failing cohort")
        gains.append(
            {
                "name": str(cohort.get("name") or ""),
                "dataset": str(cohort.get("dataset") or ""),
                "role": str(cohort.get("role") or ""),
                "mean_delta": cohort.get("mean_delta"),
                "improved_task_count": len(cohort.get("improved_task_ids") or []),
                "regressed_task_count": len(cohort.get("regressed_task_ids") or []),
                "token_cost_ratio": cohort.get("token_cost_ratio"),
                "task_ids_exposed": False,
            }
        )
    if not gains:
        raise ValueError("accepted validation gate has no cohort evidence")
    source_decision_id = str(outcome.get("decision_id") or "")
    baseline_id = str(baseline.get("baseline_id") or "")
    preservation_id = "preservation-" + _canonical_sha256(
        {
            "source_decision_id": source_decision_id,
            "baseline_id": baseline_id,
            "attempt_fingerprint": fingerprint,
        }
    )[:20]
    return {
        "schema_version": PRESERVATION_MEMORY_SCHEMA,
        "preservation_id": preservation_id,
        "source_decision_id": source_decision_id,
        "baseline_id": baseline_id,
        "baseline_commit": str(baseline.get("commit") or ""),
        "candidate_id": str(outcome.get("candidate_id") or ""),
        "attempt": {
            "plan_id": plan_id,
            "hypothesis_id": str(selected.get("hypothesis_id") or ""),
            "allowed_paths": paths,
            "required_behavior_delta": delta,
            "fingerprint": fingerprint,
        },
        "preservation_constraints": {
            "must_preserve": list(contract.get("must_preserve") or []),
            "non_goals": list(contract.get("non_goals") or []),
            "regression_scenarios": list(
                (selected.get("risk") or {}).get("regression_scenarios") or []
            ),
        },
        "validated_gains": gains,
        "artifacts": {
            "candidate_outcome": {"path": str(outcome_path), "sha256": outcome_sha},
            "modification_plan": {"path": str(plan_path), "sha256": plan_sha},
            "validation_gate": {"path": str(gate_path), "sha256": gate_sha},
            "materialization": {
                "path": str(materialization_path),
                "sha256": materialization_sha,
            },
            "baseline_entry": {
                "path": str(baseline_entry_path),
                "sha256": baseline_sha,
            },
        },
    }


def build_preservation_memory(
    *,
    outcome_path: Path,
    materialization_path: Path,
    baseline_entry_path: Path,
    memory_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Persist gains only after the accepted commit becomes a registered baseline."""

    record = _derive_preservation_memory(
        outcome_path=outcome_path,
        materialization_path=materialization_path,
        baseline_entry_path=baseline_entry_path,
    )
    if not all(
        [
            record["source_decision_id"],
            record["baseline_id"],
            record["baseline_commit"],
            record["candidate_id"],
            record["attempt"]["hypothesis_id"],
        ]
    ):
        raise ValueError("accepted candidate lineage is incomplete")
    destination = (
        Path(memory_root).resolve()
        / "accepted"
        / f"{record['preservation_id']}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record, destination


def _validate_preservation_memory_record(memory: Mapping[str, Any]) -> None:
    if memory.get("schema_version") != PRESERVATION_MEMORY_SCHEMA:
        raise ValueError("unsupported preservation memory schema")
    artifacts = memory.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("preservation memory has no artifact lineage")
    required = {"candidate_outcome", "materialization", "baseline_entry"}
    if not required.issubset(artifacts):
        raise ValueError("preservation memory artifact lineage is incomplete")
    expected = _derive_preservation_memory(
        outcome_path=Path(str(artifacts["candidate_outcome"].get("path") or "")),
        materialization_path=Path(str(artifacts["materialization"].get("path") or "")),
        baseline_entry_path=Path(str(artifacts["baseline_entry"].get("path") or "")),
    )
    if memory != expected:
        raise ValueError("preservation memory projection is inconsistent")


def _hypothesis_paths(
    hypothesis: Mapping[str, Any], diagnoses: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    paths: set[str] = set()
    for ref in hypothesis.get("affected_source_refs") or []:
        if not isinstance(ref, Mapping):
            raise ValueError("hypothesis source ref must be an object")
        task_id = str(ref.get("task_id") or "")
        try:
            index = int(ref.get("index"))
            source = (diagnoses[task_id].get("affected_sources") or [])[index]
            path = str(source.get("path") or "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("hypothesis contains an unresolved source ref") from exc
        if not path:
            raise ValueError("hypothesis source ref resolves to an empty path")
        paths.add(path)
    return sorted(paths)


def validate_planning_memory_context(
    value: Mapping[str, Any],
    *,
    cohort: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    cohort_sha256: str,
    hypotheses_sha256: str,
) -> None:
    """Verify that a planning context is the exact projection it claims to be."""

    if value.get("schema_version") != CONTEXT_SCHEMA:
        raise ValueError("unsupported planning memory context schema")
    if value.get("cohort_id") != cohort.get("cohort_id"):
        raise ValueError("planning memory context cohort_id does not match input")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("planning memory context has no input lineage")
    for label, expected in (
        ("cohort", cohort_sha256),
        ("hypotheses", hypotheses_sha256),
    ):
        ref = inputs.get(label)
        if not isinstance(ref, Mapping) or ref.get("sha256") != expected:
            raise ValueError(f"planning memory {label} hash does not match input")
    selection = value.get("selection")
    if not isinstance(selection, Mapping) or selection.get("rule") != (
        "exact_affected_source_path_overlap"
    ):
        raise ValueError("unsupported planning memory selection rule")
    if selection.get("silent_truncation") is not False:
        raise ValueError("planning memory must not silently truncate failures")
    maximum = selection.get("max_entries_per_hypothesis")
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError("planning memory has an invalid entry limit")

    diagnoses = {
        str(item.get("task_id") or ""): item
        for item in cohort.get("eligible_diagnoses") or []
    }
    hypothesis_by_id = {
        str(item.get("hypothesis_id") or ""): item
        for item in hypotheses.get("hypotheses") or []
    }
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("planning memory entries must be a list")
    seen_hypotheses: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("planning memory entry must be an object")
        hypothesis_id = str(entry.get("hypothesis_id") or "")
        if hypothesis_id not in hypothesis_by_id or hypothesis_id in seen_hypotheses:
            raise ValueError("planning memory has unknown or duplicate hypothesis")
        seen_hypotheses.add(hypothesis_id)
        paths = _hypothesis_paths(hypothesis_by_id[hypothesis_id], diagnoses)
        if entry.get("affected_paths") != paths:
            raise ValueError("planning memory affected paths do not match hypothesis")
        failures = entry.get("relevant_failures")
        if not isinstance(failures, list) or len(failures) > maximum:
            raise ValueError("planning memory relevant failures exceed their bound")
        seen_memories: set[str] = set()
        for failure in failures:
            if not isinstance(failure, Mapping):
                raise ValueError("planning memory failure must be an object")
            artifact = failure.get("memory_artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("planning memory failure has no artifact reference")
            memory, sha = _read_object(
                Path(str(artifact.get("path") or "")), "repair memory"
            )
            if sha != artifact.get("sha256"):
                raise ValueError("repair memory artifact hash does not match context")
            memory_id = str(memory.get("memory_id") or "")
            if memory.get("schema_version") != MEMORY_SCHEMA:
                raise ValueError("unsupported repair memory schema")
            _validate_repair_memory_record(memory)
            if memory_id != failure.get("memory_id") or memory_id in seen_memories:
                raise ValueError("planning memory has mismatched or duplicate memory ID")
            seen_memories.add(memory_id)
            overlap = sorted(set(paths) & set(memory.get("attempt", {}).get("allowed_paths") or []))
            if not overlap or failure.get("overlapping_paths") != overlap:
                raise ValueError("planning memory source overlap is inconsistent")
            for field in ("attempt", "observed_failures", "constraints"):
                if failure.get(field) != memory.get(field):
                    raise ValueError(f"planning memory {field} projection is inconsistent")
        successes = entry.get("relevant_successes")
        if not isinstance(successes, list) or len(successes) > maximum:
            raise ValueError("planning memory relevant successes exceed their bound")
        seen_preservations: set[str] = set()
        for success in successes:
            if not isinstance(success, Mapping):
                raise ValueError("planning preservation entry must be an object")
            artifact = success.get("memory_artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("planning preservation entry has no artifact reference")
            memory, sha = _read_object(
                Path(str(artifact.get("path") or "")), "preservation memory"
            )
            if sha != artifact.get("sha256"):
                raise ValueError("preservation memory artifact hash does not match context")
            _validate_preservation_memory_record(memory)
            preservation_id = str(memory.get("preservation_id") or "")
            if (
                preservation_id != success.get("preservation_id")
                or preservation_id in seen_preservations
            ):
                raise ValueError(
                    "planning memory has mismatched or duplicate preservation ID"
                )
            seen_preservations.add(preservation_id)
            overlap = sorted(
                set(paths) & set(memory.get("attempt", {}).get("allowed_paths") or [])
            )
            if not overlap or success.get("overlapping_paths") != overlap:
                raise ValueError("planning preservation source overlap is inconsistent")
            for field in (
                "attempt",
                "preservation_constraints",
                "validated_gains",
                "baseline_id",
                "baseline_commit",
            ):
                if success.get(field) != memory.get(field):
                    raise ValueError(
                        f"planning preservation {field} projection is inconsistent"
                    )
    if seen_hypotheses != set(hypothesis_by_id):
        raise ValueError("planning memory is missing hypotheses")


def build_planning_memory_context(
    *,
    memory_root: Path,
    cohort_path: Path,
    hypotheses_path: Path,
    output_path: Path,
    max_entries_per_hypothesis: int = 12,
) -> dict[str, Any]:
    """Select prior failures by exact affected-source overlap; never truncate silently."""

    if max_entries_per_hypothesis < 1:
        raise ValueError("max_entries_per_hypothesis must be positive")
    cohort, cohort_sha = _read_object(cohort_path, "hypothesis cohort")
    hypotheses, hypotheses_sha = _read_object(hypotheses_path, "hypotheses")
    if cohort.get("schema_version") != "deepread-hypothesis-cohort-v1":
        raise ValueError("unsupported hypothesis cohort schema")
    if hypotheses.get("schema_version") != "deepread-improvement-hypotheses-v1":
        raise ValueError("unsupported improvement hypothesis schema")
    if cohort.get("cohort_id") != hypotheses.get("cohort_id"):
        raise ValueError("cohort and hypothesis IDs do not match")

    memories = []
    for path in sorted((Path(memory_root).resolve() / "rejected").glob("*.json")):
        value, sha = _read_object(path, "repair memory")
        if value.get("schema_version") != MEMORY_SCHEMA:
            raise ValueError(f"unsupported repair memory schema: {path}")
        _validate_repair_memory_record(value)
        memories.append((value, path, sha))
    preservations = []
    for path in sorted((Path(memory_root).resolve() / "accepted").glob("*.json")):
        value, sha = _read_object(path, "preservation memory")
        _validate_preservation_memory_record(value)
        preservations.append((value, path, sha))
    diagnoses = {
        str(item.get("task_id") or ""): item
        for item in cohort.get("eligible_diagnoses") or []
    }
    entries = []
    for hypothesis in hypotheses.get("hypotheses") or []:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        paths = _hypothesis_paths(hypothesis, diagnoses)
        matches = []
        for memory, path, sha in memories:
            attempted_paths = set(memory.get("attempt", {}).get("allowed_paths") or [])
            overlap = sorted(set(paths) & attempted_paths)
            if overlap:
                matches.append(
                    {
                        "memory_id": memory.get("memory_id"),
                        "overlapping_paths": overlap,
                        "attempt": memory.get("attempt"),
                        "observed_failures": memory.get("observed_failures"),
                        "constraints": memory.get("constraints"),
                        "memory_artifact": {"path": str(path), "sha256": sha},
                    }
                )
        successes = []
        for memory, path, sha in preservations:
            attempted_paths = set(memory.get("attempt", {}).get("allowed_paths") or [])
            overlap = sorted(set(paths) & attempted_paths)
            if overlap:
                successes.append(
                    {
                        "preservation_id": memory.get("preservation_id"),
                        "baseline_id": memory.get("baseline_id"),
                        "baseline_commit": memory.get("baseline_commit"),
                        "overlapping_paths": overlap,
                        "attempt": memory.get("attempt"),
                        "preservation_constraints": memory.get(
                            "preservation_constraints"
                        ),
                        "validated_gains": memory.get("validated_gains"),
                        "memory_artifact": {"path": str(path), "sha256": sha},
                    }
                )
        if len(matches) > max_entries_per_hypothesis:
            raise ValueError(
                f"{hypothesis_id} has {len(matches)} relevant memories; "
                f"refusing to truncate to {max_entries_per_hypothesis}"
            )
        if len(successes) > max_entries_per_hypothesis:
            raise ValueError(
                f"{hypothesis_id} has {len(successes)} relevant preservation memories; "
                f"refusing to truncate to {max_entries_per_hypothesis}"
            )
        entries.append(
            {
                "hypothesis_id": hypothesis_id,
                "affected_paths": paths,
                "relevant_failures": matches,
                "relevant_successes": successes,
            }
        )

    result = {
        "schema_version": CONTEXT_SCHEMA,
        "cohort_id": cohort.get("cohort_id"),
        "entries": entries,
        "selection": {
            "rule": "exact_affected_source_path_overlap",
            "max_entries_per_hypothesis": max_entries_per_hypothesis,
            "silent_truncation": False,
        },
        "inputs": {
            "cohort": {"path": str(Path(cohort_path).resolve()), "sha256": cohort_sha},
            "hypotheses": {
                "path": str(Path(hypotheses_path).resolve()),
                "sha256": hypotheses_sha,
            },
        },
    }
    validate_planning_memory_context(
        result,
        cohort=cohort,
        hypotheses=hypotheses,
        cohort_sha256=cohort_sha,
        hypotheses_sha256=hypotheses_sha,
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result
