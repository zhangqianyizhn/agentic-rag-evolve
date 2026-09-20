"""Build a compact terminal report without copying large or sealed artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from agentic_rag_evolve.planning import (
    validate_preservation_memory,
    validate_repair_memory,
)


ITERATION_REPORT_SCHEMA = "deepread-iteration-report-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _pinned_artifact(
    outcome: Mapping[str, Any], key: str, expected_schema: str
) -> tuple[dict[str, Any], dict[str, str]]:
    reference = (outcome.get("artifacts") or {}).get(key)
    if not isinstance(reference, Mapping):
        raise ValueError(f"candidate outcome has no {key} artifact reference")
    path = str(reference.get("path") or "")
    if not path:
        raise ValueError(f"candidate outcome has an empty {key} artifact path")
    value, actual = _read_object(Path(path), key.replace("_", " "))
    if actual["sha256"] != reference.get("sha256"):
        raise ValueError(f"{key} artifact hash does not match outcome lineage")
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported {key} artifact schema")
    return value, actual


def _validate_outcome_identity(
    outcome: Mapping[str, Any],
    outcome_ref: Mapping[str, str],
    gate_ref: Mapping[str, str],
) -> None:
    status = str(outcome.get("outcome") or "")
    if status not in {"accepted", "rejected"}:
        raise ValueError("iteration outcome must be accepted or rejected")
    basis = {
        "candidate_id": outcome.get("candidate_id"),
        "candidate_snapshot_sha256": outcome.get("candidate_snapshot_sha256"),
        "validation_gate_sha256": gate_ref.get("sha256"),
        "outcome": status,
    }
    digest = _canonical_sha256(basis)
    if outcome.get("decision_basis_sha256") != digest:
        raise ValueError("candidate outcome decision basis is invalid")
    if outcome.get("decision_id") != "outcome-" + digest[:20]:
        raise ValueError("candidate outcome decision ID is invalid")
    if not outcome_ref.get("sha256"):
        raise ValueError("candidate outcome hash is missing")


def _cohort_projection(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    projected = []
    for raw in gate.get("cohorts") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("validation gate cohort must be an object")
        role = str(raw.get("role") or "")
        if role not in {"development", "holdout", "cross_dataset"}:
            raise ValueError(f"unsupported validation cohort role: {role!r}")
        improved = raw.get("improved_task_ids") or []
        regressed = raw.get("regressed_task_ids") or []
        if not isinstance(improved, list) or not isinstance(regressed, list):
            raise ValueError("validation task ID fields must be lists")
        projected.append(
            {
                "name": str(raw.get("name") or ""),
                "dataset": str(raw.get("dataset") or ""),
                "role": role,
                "passed": bool(raw.get("passed")),
                "task_count": int(
                    raw.get("task_count") or len(raw.get("pairs") or [])
                ),
                "mean_delta": raw.get("mean_delta"),
                "improved_task_count": len(improved),
                "regressed_task_count": len(regressed),
                "token_cost_ratio": raw.get("token_cost_ratio"),
                "failure_reasons": sorted(
                    {str(item) for item in raw.get("failure_reasons") or []}
                ),
                "task_ids_exposed": False,
            }
        )
    if not projected:
        raise ValueError("validation gate has no cohorts")
    return projected


def _terminal_projection(
    outcome: Mapping[str, Any],
    terminal_memory: Mapping[str, Any],
    terminal_ref: Mapping[str, str],
) -> dict[str, Any]:
    decision_id = str(outcome.get("decision_id") or "")
    if outcome.get("outcome") == "rejected":
        validate_repair_memory(terminal_memory)
        if terminal_memory.get("source_decision_id") != decision_id:
            raise ValueError("repair memory belongs to another candidate outcome")
        constraints = terminal_memory.get("constraints") or {}
        return {
            "status": "rejected_retained",
            "memory_id": terminal_memory.get("memory_id"),
            "attempt_fingerprint": (terminal_memory.get("attempt") or {}).get(
                "fingerprint"
            ),
            "feedback_task_count": len(constraints.get("protect_task_ids") or []),
            "sealed_regression_count": int(
                constraints.get("sealed_regression_count") or 0
            ),
            "baseline_changed": False,
            "terminal_artifact": dict(terminal_ref),
        }

    validate_preservation_memory(terminal_memory)
    if terminal_memory.get("source_decision_id") != decision_id:
        raise ValueError("preservation memory belongs to another candidate outcome")
    return {
        "status": "accepted_registered",
        "preservation_id": terminal_memory.get("preservation_id"),
        "attempt_fingerprint": (terminal_memory.get("attempt") or {}).get(
            "fingerprint"
        ),
        "validated_gain_count": len(terminal_memory.get("validated_gains") or []),
        "baseline_changed": True,
        "baseline": {
            "baseline_id": terminal_memory.get("baseline_id"),
            "commit": terminal_memory.get("baseline_commit"),
        },
        "terminal_artifact": dict(terminal_ref),
    }


def _diagnosis_and_hypotheses(
    *,
    cohort: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if cohort.get("schema_version") != "deepread-hypothesis-cohort-v1":
        raise ValueError("unsupported hypothesis cohort schema")
    if hypotheses.get("schema_version") != "deepread-improvement-hypotheses-v1":
        raise ValueError("unsupported improvement hypotheses schema")
    cohort_id = str(cohort.get("cohort_id") or "")
    if not cohort_id or hypotheses.get("cohort_id") != cohort_id:
        raise ValueError("cohort and hypotheses IDs do not match")
    if plan.get("cohort_id") != cohort_id:
        raise ValueError("modification plan cohort ID does not match diagnoses")
    eligible = cohort.get("eligible_diagnoses") or []
    excluded = cohort.get("excluded_diagnoses") or []
    if not isinstance(eligible, list) or not isinstance(excluded, list):
        raise ValueError("hypothesis cohort diagnosis fields must be lists")
    routes = Counter(str(item.get("route") or "unknown") for item in excluded)
    hypothesis_items = hypotheses.get("hypotheses") or []
    if not isinstance(hypothesis_items, list):
        raise ValueError("improvement hypotheses must be a list")
    hypothesis_ids = {
        str(item.get("hypothesis_id") or "") for item in hypothesis_items
    }
    selected_ids = {
        str(item.get("hypothesis_id") or "") for item in plan.get("plans") or []
    }
    if not selected_ids.issubset(hypothesis_ids):
        raise ValueError("modification plan references an unknown hypothesis")
    maturity = Counter(
        str(item.get("maturity") or "unknown") for item in hypothesis_items
    )
    return (
        {
            "cohort_id": cohort_id,
            "input_count": len(eligible) + len(excluded),
            "eligible_count": len(eligible),
            "excluded_count": len(excluded),
            "excluded_routes": dict(sorted(routes.items())),
            "task_ids_exposed": False,
        },
        {
            "status": hypotheses.get("status"),
            "hypothesis_count": len(hypothesis_items),
            "maturity_counts": dict(sorted(maturity.items())),
            "selected_hypothesis_count": len(selected_ids),
            "task_ids_exposed": False,
        },
    )


def _feedback_projection(
    feedback: Mapping[str, Any], outcome: Mapping[str, Any], gate_sha256: str
) -> dict[str, Any]:
    if feedback.get("schema_version") != "deepread-regression-feedback-v1":
        raise ValueError("unsupported regression feedback schema")
    for field in ("candidate_id", "plan_id", "candidate_snapshot_sha256"):
        if feedback.get(field) != outcome.get(field):
            raise ValueError(f"regression feedback {field} does not match outcome")
    if feedback.get("final_test_accessed") is not False:
        raise ValueError("regression feedback accessed final test")
    scope = feedback.get("diagnosis_scope") or {}
    if scope.get("allowed_roles") != ["development"]:
        raise ValueError("regression feedback diagnosis scope is not development-only")
    task_ids = scope.get("task_ids") or []
    if not isinstance(task_ids, list):
        raise ValueError("regression feedback task IDs must be a list")
    sealed = feedback.get("sealed_cohort_summaries") or []
    development = feedback.get("development_cohorts") or []
    if not isinstance(sealed, list) or not isinstance(development, list):
        raise ValueError("regression feedback cohort summaries must be lists")
    if any(item.get("role") != "development" for item in development):
        raise ValueError("regression feedback exposes a non-development cohort")
    forbidden = {
        "regressed_task_ids",
        "improved_task_ids",
        "baseline_evaluation",
        "candidate_evaluation",
    }
    if any(forbidden & set(item) for item in sealed if isinstance(item, Mapping)):
        raise ValueError("sealed regression feedback exposes task-level fields")
    feedback_gate = (feedback.get("artifacts") or {}).get("validation_gate") or {}
    if feedback_gate.get("sha256") != gate_sha256:
        raise ValueError("regression feedback gate does not match outcome")
    basis = {
        "validation_gate_sha256": gate_sha256,
        "candidate_snapshot_sha256": outcome.get("candidate_snapshot_sha256"),
        "development_regressed_task_ids": sorted(str(item) for item in task_ids),
    }
    if feedback.get("feedback_id") != "feedback-" + _canonical_sha256(basis)[:20]:
        raise ValueError("regression feedback identity is invalid")
    expected_status = "ready" if task_ids else "no_development_regressions"
    if feedback.get("status") != expected_status:
        raise ValueError("regression feedback status is inconsistent")
    return {
        "status": feedback.get("status"),
        "development_regression_count": len(task_ids),
        "sealed_regression_count": sum(
            int(item.get("regressed_task_count") or 0) for item in sealed
        ),
        "task_ids_exposed": False,
    }


def build_iteration_report(
    *,
    outcome_path: Path,
    terminal_memory_path: Path,
    cohort_path: Path,
    hypotheses_path: Path,
    output_path: Path,
    regression_feedback_path: Path | None = None,
) -> dict[str, Any]:
    """Persist one compact report after an iteration reaches a durable terminal state."""

    outcome, outcome_ref = _read_object(outcome_path, "candidate outcome")
    if outcome.get("schema_version") != "deepread-candidate-outcome-v1":
        raise ValueError("unsupported candidate outcome schema")
    manifest, manifest_ref = _pinned_artifact(
        outcome, "candidate_manifest", "deepread-candidate-manifest-v1"
    )
    plan, plan_ref = _pinned_artifact(
        outcome, "modification_plan", "deepread-modification-plan-v1"
    )
    audit, audit_ref = _pinned_artifact(
        outcome, "candidate_audit", "deepread-candidate-audit-v1"
    )
    test_audit, test_audit_ref = _pinned_artifact(
        outcome, "candidate_test_audit", "deepread-candidate-test-audit-v1"
    )
    suite, suite_ref = _pinned_artifact(
        outcome, "validation_suite", "deepread-validation-suite-v1"
    )
    gate, gate_ref = _pinned_artifact(
        outcome, "validation_gate", "deepread-validation-gate-v1"
    )
    _validate_outcome_identity(outcome, outcome_ref, gate_ref)
    if audit.get("candidate_manifest_sha256") != manifest_ref["sha256"]:
        raise ValueError("candidate audit is not bound to manifest")
    if test_audit.get("candidate_audit_sha256") != audit_ref["sha256"]:
        raise ValueError("candidate test audit is not bound to static audit")
    if gate.get("candidate_audit_sha256") != audit_ref["sha256"]:
        raise ValueError("validation gate is not bound to static audit")
    if gate.get("candidate_test_audit_sha256") != test_audit_ref["sha256"]:
        raise ValueError("validation gate is not bound to fixed-test audit")
    if gate.get("validation_suite_sha256") != suite_ref["sha256"]:
        raise ValueError("validation gate is not bound to validation suite")
    if not plan_ref["sha256"] or any(
        value != plan_ref["sha256"]
        for value in (
            manifest.get("plan_sha256"),
            audit.get("plan_sha256"),
            test_audit.get("plan_sha256"),
            gate.get("plan_sha256"),
        )
    ):
        raise ValueError("candidate artifacts do not share the modification plan")
    for field in ("candidate_id", "plan_id"):
        expected = outcome.get(field)
        for label, artifact in (
            ("manifest", manifest),
            ("audit", audit),
            ("test audit", test_audit),
            ("gate", gate),
        ):
            if artifact.get(field) != expected:
                raise ValueError(f"{label} {field} does not match outcome")
    snapshot = outcome.get("candidate_snapshot_sha256")
    for label, value in (
        ("audit", audit.get("candidate_snapshot_sha256")),
        ("test audit", test_audit.get("candidate_snapshot_sha256")),
        ("gate", gate.get("candidate_snapshot_sha256")),
    ):
        if value != snapshot:
            raise ValueError(f"{label} snapshot does not match outcome")
    if not audit.get("passed") or not test_audit.get("passed"):
        raise ValueError("candidate audits did not pass")
    if gate.get("gate_level") != "promotion" or suite.get("gate_level") != "promotion":
        raise ValueError("iteration report requires a promotion gate")
    if (
        suite.get("candidate_id") != outcome.get("candidate_id")
        or suite.get("plan_id") != outcome.get("plan_id")
        or suite.get("candidate_snapshot_sha256") != snapshot
    ):
        raise ValueError("validation suite does not match outcome")
    expected_passed = outcome.get("outcome") == "accepted"
    if bool(gate.get("passed")) != expected_passed:
        raise ValueError("validation gate decision does not match outcome")
    gate_cohorts = gate.get("cohorts") or []
    failed = sorted(
        str(item.get("name") or "") for item in gate_cohorts if not item.get("passed")
    )
    if failed != sorted(str(item) for item in gate.get("failed_cohorts") or []):
        raise ValueError("validation gate failed cohort list is inconsistent")
    if failed != sorted(str(item) for item in outcome.get("failed_cohorts") or []):
        raise ValueError("candidate outcome failed cohorts do not match gate")
    expected_summary = [
        {
            "name": item.get("name"),
            "dataset": item.get("dataset"),
            "role": item.get("role"),
            "passed": bool(item.get("passed")),
            "mean_delta": item.get("mean_delta"),
            "regressed_task_ids": item.get("regressed_task_ids") or [],
            "token_cost_ratio": item.get("token_cost_ratio"),
            "failure_reasons": item.get("failure_reasons") or [],
        }
        for item in gate_cohorts
    ]
    if outcome.get("cohort_summary") != expected_summary:
        raise ValueError("candidate outcome cohort summary does not match gate")
    expected_reasons = ["promotion_gate_passed"] if expected_passed else [
        f"cohort:{item.get('name')}:{reason}"
        for item in gate_cohorts
        if not item.get("passed")
        for reason in item.get("failure_reasons") or ["failed"]
    ]
    if outcome.get("decision_reasons") != expected_reasons:
        raise ValueError("candidate outcome reasons do not match gate")

    plans = [
        item
        for item in plan.get("plans") or []
        if item.get("plan_id") == outcome.get("plan_id")
    ]
    if len(plans) != 1 or plans[0].get("decision") != "proceed":
        raise ValueError("iteration outcome must reference one proceeding plan")
    selected = plans[0]
    cohort, cohort_ref = _read_object(cohort_path, "hypothesis cohort")
    hypotheses, hypotheses_ref = _read_object(
        hypotheses_path, "improvement hypotheses"
    )
    diagnosis_summary, hypothesis_summary = _diagnosis_and_hypotheses(
        cohort=cohort, hypotheses=hypotheses, plan=plan
    )
    terminal_memory, terminal_ref = _read_object(
        terminal_memory_path, "terminal memory"
    )
    terminal = _terminal_projection(outcome, terminal_memory, terminal_ref)

    feedback_summary: dict[str, Any] | None = None
    feedback_ref: dict[str, str] | None = None
    if regression_feedback_path is not None:
        feedback, feedback_ref = _read_object(
            regression_feedback_path, "regression feedback"
        )
        feedback_summary = _feedback_projection(
            feedback, outcome, gate_ref["sha256"]
        )

    checks = test_audit.get("checks") or []
    if not isinstance(checks, list):
        raise ValueError("candidate test audit checks must be a list")
    changed_paths = audit.get("changed_paths") or []
    if not isinstance(changed_paths, list):
        raise ValueError("candidate changed paths must be a list")
    report_basis = {
        "decision_id": outcome.get("decision_id"),
        "outcome_sha256": outcome_ref["sha256"],
        "terminal_memory_sha256": terminal_ref["sha256"],
    }
    report = {
        "schema_version": ITERATION_REPORT_SCHEMA,
        "iteration_id": "iteration-" + _canonical_sha256(report_basis)[:20],
        "diagnosis": diagnosis_summary,
        "hypotheses": hypothesis_summary,
        "plan": {
            "plan_id": outcome.get("plan_id"),
            "hypothesis_id": selected.get("hypothesis_id"),
            "decision": selected.get("decision"),
            "allowed_paths": sorted(
                {
                    str(item)
                    for item in (selected.get("edit_scope") or {}).get(
                        "allowed_paths"
                    )
                    or []
                }
            ),
            "required_behavior_delta": (selected.get("change_contract") or {}).get(
                "required_behavior_delta"
            ),
            "risk_level": (selected.get("risk") or {}).get("level"),
        },
        "candidate": {
            "candidate_id": outcome.get("candidate_id"),
            "base_commit": outcome.get("base_commit"),
            "snapshot_sha256": snapshot,
            "changed_file_count": len(changed_paths),
            "changed_paths": sorted(str(item) for item in changed_paths),
            "static_audit_passed": True,
            "fixed_tests_passed": True,
            "fixed_test_count": sum(
                int(item.get("tests_run") or 0) for item in checks
            ),
        },
        "validation": {
            "gate_level": "promotion",
            "passed": bool(gate.get("passed")),
            "failed_cohort_count": len(gate.get("failed_cohorts") or []),
            "cohorts": _cohort_projection(gate),
            "task_ids_exposed": False,
        },
        "decision": {
            "decision_id": outcome.get("decision_id"),
            "outcome": outcome.get("outcome"),
            "reasons": list(outcome.get("decision_reasons") or []),
            "next_action": outcome.get("next_action"),
        },
        "terminal": terminal,
        "regression_feedback": feedback_summary,
        "artifacts": {
            "hypothesis_cohort": cohort_ref,
            "improvement_hypotheses": hypotheses_ref,
            "modification_plan": plan_ref,
            "candidate_audit": audit_ref,
            "candidate_test_audit": test_audit_ref,
            "candidate_outcome": outcome_ref,
            "terminal_memory": terminal_ref,
            **({"regression_feedback": feedback_ref} if feedback_ref else {}),
        },
        "large_payloads_embedded": False,
        "final_test_accessed": False,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report
