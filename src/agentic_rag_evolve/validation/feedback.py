"""Build a sealed-boundary regression feedback artifact for rediagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FEEDBACK_SCHEMA = "deepread-regression-feedback-v1"


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return value, hashlib.sha256(data).hexdigest()


def _artifact(path: Path, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = Path(path).resolve()
    value, digest = _read_object(resolved, label)
    return value, {"path": str(resolved), "sha256": digest}


def _file_reference(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _required_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    return Path(value).resolve()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_same(label: str, expected: Any, actual: Any) -> None:
    if not expected or expected != actual:
        raise ValueError(f"{label} does not match validation lineage")


def _task_ids(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = [str(item) for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty task IDs")
    return result


def _development_projection(
    cohort: Mapping[str, Any], suite_cohort: Mapping[str, Any]
) -> dict[str, Any]:
    task_ids = _task_ids(suite_cohort.get("task_ids"), label="development task_ids")
    regressed = _task_ids(
        cohort.get("regressed_task_ids"), label="development regressed_task_ids"
    )
    pair_ids = _task_ids(
        [item.get("task_id") for item in cohort.get("pairs") or []],
        label="development pair task_ids",
    )
    if set(pair_ids) != set(task_ids):
        raise ValueError("development gate pairs do not match validation suite tasks")
    if not set(regressed).issubset(task_ids):
        raise ValueError("development regression IDs are outside the validation suite")

    baseline_path = _required_path(
        cohort.get("baseline_evaluation"), label="development baseline evaluation"
    )
    candidate_path = _required_path(
        cohort.get("candidate_evaluation"), label="development candidate evaluation"
    )
    baseline_ref = _file_reference(baseline_path)
    candidate_ref = _file_reference(candidate_path)
    _require_same(
        "development baseline evaluation hash",
        cohort.get("baseline_evaluation_sha256"),
        baseline_ref["sha256"],
    )
    _require_same(
        "development candidate evaluation hash",
        cohort.get("candidate_evaluation_sha256"),
        candidate_ref["sha256"],
    )
    return {
        "name": str(cohort.get("name") or ""),
        "dataset": str(cohort.get("dataset") or ""),
        "role": "development",
        "primary_metric": str(cohort.get("primary_metric") or ""),
        "task_count": int(cohort.get("task_count") or 0),
        "mean_delta": cohort.get("mean_delta"),
        "regressed_task_count": len(regressed),
        "regressed_task_ids": sorted(regressed),
        "token_cost_ratio": cohort.get("token_cost_ratio"),
        "failure_reasons": sorted(
            {str(item) for item in cohort.get("failure_reasons") or []}
        ),
        "baseline_evaluation": baseline_ref,
        "candidate_evaluation": candidate_ref,
    }


def _sealed_projection(cohort: Mapping[str, Any]) -> dict[str, Any]:
    regressed = _task_ids(
        cohort.get("regressed_task_ids"), label="sealed regressed_task_ids"
    )
    return {
        "name": str(cohort.get("name") or ""),
        "dataset": str(cohort.get("dataset") or ""),
        "role": str(cohort.get("role") or ""),
        "task_count": int(cohort.get("task_count") or 0),
        "mean_delta": cohort.get("mean_delta"),
        "regressed_task_count": len(regressed),
        "token_cost_ratio": cohort.get("token_cost_ratio"),
        "failure_reasons": sorted(
            {str(item) for item in cohort.get("failure_reasons") or []}
        ),
        "task_id_visibility": "sealed",
    }


def build_regression_feedback(
    *,
    gate_path: Path,
    suite_path: Path,
    candidate_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write feedback that authorizes rediagnosis of development regressions only."""

    gate, gate_ref = _artifact(gate_path, "validation gate")
    suite, suite_ref = _artifact(suite_path, "validation suite")
    audit, audit_ref = _artifact(candidate_audit_path, "candidate audit")
    if gate.get("schema_version") != "deepread-validation-gate-v1":
        raise ValueError("unsupported validation gate schema")
    if suite.get("schema_version") != "deepread-validation-suite-v1":
        raise ValueError("unsupported validation suite schema")
    if audit.get("schema_version") != "deepread-candidate-audit-v1":
        raise ValueError("unsupported candidate audit schema")
    if not audit.get("passed"):
        raise ValueError("candidate static audit must pass before regression feedback")
    _require_same(
        "validation suite hash", suite_ref["sha256"], gate.get("validation_suite_sha256")
    )
    _require_same(
        "candidate audit hash", audit_ref["sha256"], gate.get("candidate_audit_sha256")
    )
    for field in ("candidate_id", "plan_id", "candidate_snapshot_sha256"):
        _require_same(f"gate/suite {field}", suite.get(field), gate.get(field))
        _require_same(f"gate/audit {field}", audit.get(field), gate.get(field))

    gate_level = str(gate.get("gate_level") or "")
    if gate_level not in {"development", "promotion"}:
        raise ValueError("regression feedback requires a development or promotion gate")
    if suite.get("gate_level") != gate_level:
        raise ValueError("validation suite gate level does not match gate")
    candidate_root = _required_path(
        audit.get("candidate_path"), label="candidate source root"
    )
    if not candidate_root.is_dir():
        raise ValueError(f"candidate source root is not a directory: {candidate_root}")

    suite_cohorts = suite.get("cohorts")
    gate_cohorts = gate.get("cohorts")
    if not isinstance(suite_cohorts, list) or not isinstance(gate_cohorts, list):
        raise ValueError("validation artifacts must contain cohort lists")
    suite_by_name = {
        str(item.get("name") or ""): item
        for item in suite_cohorts
        if isinstance(item, Mapping)
    }
    if len(suite_by_name) != len(suite_cohorts) or "" in suite_by_name:
        raise ValueError("validation suite cohort names must be unique and non-empty")

    development = []
    sealed = []
    seen_names: set[str] = set()
    for raw in gate_cohorts:
        if not isinstance(raw, Mapping):
            raise ValueError("validation gate cohort must be an object")
        name = str(raw.get("name") or "")
        if not name or name in seen_names or name not in suite_by_name:
            raise ValueError("validation gate cohort names do not match suite")
        seen_names.add(name)
        spec = suite_by_name[name]
        role = str(raw.get("role") or "")
        if role != spec.get("role") or raw.get("dataset") != spec.get("dataset"):
            raise ValueError(f"validation cohort {name!r} does not match suite")
        if role == "development":
            development.append(_development_projection(raw, spec))
        elif role in {"holdout", "cross_dataset"}:
            sealed.append(_sealed_projection(raw))
        else:
            raise ValueError(f"unsupported validation cohort role: {role!r}")
    if seen_names != set(suite_by_name):
        raise ValueError("validation gate is missing suite cohorts")
    if not development:
        raise ValueError("regression feedback requires a development cohort")

    eligible_ids = sorted(
        {
            task_id
            for cohort in development
            for task_id in cohort["regressed_task_ids"]
        }
    )
    basis = {
        "validation_gate_sha256": gate_ref["sha256"],
        "candidate_snapshot_sha256": gate.get("candidate_snapshot_sha256"),
        "development_regressed_task_ids": eligible_ids,
    }
    feedback = {
        "schema_version": FEEDBACK_SCHEMA,
        "feedback_id": "feedback-" + _canonical_sha256(basis)[:20],
        "candidate_id": str(gate.get("candidate_id") or ""),
        "plan_id": str(gate.get("plan_id") or ""),
        "candidate_snapshot_sha256": str(
            gate.get("candidate_snapshot_sha256") or ""
        ),
        "gate_level": gate_level,
        "status": "ready" if eligible_ids else "no_development_regressions",
        "diagnosis_scope": {
            "allowed_roles": ["development"],
            "task_ids": eligible_ids,
            "source_revision": "candidate_snapshot",
            "candidate_source_root": str(candidate_root),
        },
        "development_cohorts": development,
        "sealed_cohort_summaries": sealed,
        "artifacts": {
            "validation_gate": gate_ref,
            "validation_suite": suite_ref,
            "candidate_audit": audit_ref,
        },
        "final_test_accessed": False,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(feedback, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return feedback
