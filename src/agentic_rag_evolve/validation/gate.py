"""Deterministic paired validation gate for DeepRead candidates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROLES = {"development", "holdout", "cross_dataset"}
METRICS = {"f1", "recall", "accuracy_normalized"}


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_evaluation(path: Path) -> dict[str, Mapping[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"evaluation must be a JSON list: {path}")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"evaluation item must be an object: {path}")
        task_id = str(item.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise ValueError(f"evaluation has missing or duplicate task_id: {path}")
        indexed[task_id] = item
    return indexed


def _resolve(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evaluation path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _metric(item: Mapping[str, Any], name: str, field: str) -> float:
    value = (item.get("metrics") or {}).get(name)
    if value is None:
        raise ValueError(f"{field} has no comparable {name} metric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}.{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}.{name} must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{field}.{name} must be between 0 and 1")
    return result


def _tokens(item: Mapping[str, Any]) -> int:
    usage = item.get("token_usage") or {}
    return max(0, int(usage.get("input_tokens") or 0)) + max(
        0, int(usage.get("output_tokens") or 0)
    )


def _ratio(current: int, baseline: int) -> float | None:
    if baseline == 0:
        return 1.0 if current == 0 else None
    return current / baseline


def _pair_cohort(
    spec: Mapping[str, Any], *, suite_root: Path, epsilon: float
) -> dict[str, Any]:
    _exact(
        spec,
        {
            "name", "dataset", "role", "baseline_evaluation",
            "candidate_evaluation", "task_ids", "primary_metric",
            "min_mean_delta", "min_improved", "max_regressions",
            "max_token_cost_ratio",
        },
        "validation cohort",
    )
    name = str(spec.get("name") or "").strip()
    dataset = str(spec.get("dataset") or "").strip()
    role = str(spec.get("role") or "")
    metric_name = str(spec.get("primary_metric") or "")
    if not name or not dataset:
        raise ValueError("validation cohort requires name and dataset")
    if role not in ROLES:
        raise ValueError(f"validation cohort {name!r} has unsupported role {role!r}")
    if metric_name not in METRICS:
        raise ValueError(
            f"validation cohort {name!r} has unsupported primary_metric {metric_name!r}"
        )
    task_ids = [str(item) for item in spec.get("task_ids") or []]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"validation cohort {name!r} task_ids must be non-empty and unique")
    baseline_path = _resolve(suite_root, spec.get("baseline_evaluation"))
    candidate_path = _resolve(suite_root, spec.get("candidate_evaluation"))
    baseline = _load_evaluation(baseline_path)
    candidate = _load_evaluation(candidate_path)
    missing_baseline = sorted(set(task_ids) - set(baseline))
    missing_candidate = sorted(set(task_ids) - set(candidate))
    if missing_baseline or missing_candidate:
        raise ValueError(
            f"validation cohort {name!r} is missing tasks; "
            f"baseline={missing_baseline}, candidate={missing_candidate}"
        )

    pairs = []
    improved = []
    regressed = []
    baseline_tokens = 0
    candidate_tokens = 0
    baseline_errors = 0
    candidate_errors = 0
    deltas = []
    for task_id in task_ids:
        old = baseline[task_id]
        new = candidate[task_id]
        if old.get("question") != new.get("question") or old.get("sample_id") != new.get("sample_id"):
            raise ValueError(f"validation task identity mismatch: {name}/{task_id}")
        old_value = _metric(old, metric_name, f"baseline[{task_id}]")
        new_value = _metric(new, metric_name, f"candidate[{task_id}]")
        delta = new_value - old_value
        deltas.append(delta)
        if delta > epsilon:
            improved.append(task_id)
        elif delta < -epsilon:
            regressed.append(task_id)
        old_error = str((old.get("prediction") or {}).get("status") or "ok") != "ok"
        new_error = str((new.get("prediction") or {}).get("status") or "ok") != "ok"
        baseline_errors += int(old_error)
        candidate_errors += int(new_error)
        baseline_tokens += _tokens(old)
        candidate_tokens += _tokens(new)
        pairs.append(
            {
                "task_id": task_id,
                "baseline": old_value,
                "candidate": new_value,
                "delta": delta,
                "baseline_prediction_error": old_error,
                "candidate_prediction_error": new_error,
            }
        )

    mean_delta = sum(deltas) / len(deltas)
    token_ratio = _ratio(candidate_tokens, baseline_tokens)
    min_mean_delta = float(spec.get("min_mean_delta", 0.0))
    min_improved = int(spec.get("min_improved", 0))
    max_regressions = int(spec.get("max_regressions", 0))
    max_token_cost_ratio = float(spec.get("max_token_cost_ratio", 1.25))
    if not math.isfinite(min_mean_delta):
        raise ValueError(f"validation cohort {name!r}: min_mean_delta must be finite")
    if min_improved < 0 or max_regressions < 0:
        raise ValueError(f"validation cohort {name!r}: count thresholds must be non-negative")
    if not math.isfinite(max_token_cost_ratio) or max_token_cost_ratio <= 0:
        raise ValueError(f"validation cohort {name!r}: invalid token cost ratio")
    if role != "development" and min_improved != 0:
        raise ValueError(f"validation cohort {name!r}: min_improved is development-only")
    checks = {
        "mean_delta": mean_delta >= min_mean_delta,
        "improved_count": len(improved) >= min_improved,
        "regression_count": len(regressed) <= max_regressions,
        "prediction_errors": candidate_errors <= baseline_errors,
        "token_cost": token_ratio is not None and token_ratio <= max_token_cost_ratio,
    }
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "name": name,
        "dataset": dataset,
        "role": role,
        "primary_metric": metric_name,
        "task_count": len(task_ids),
        "mean_delta": mean_delta,
        "improved_task_ids": improved,
        "regressed_task_ids": regressed,
        "baseline_prediction_errors": baseline_errors,
        "candidate_prediction_errors": candidate_errors,
        "baseline_answer_tokens": baseline_tokens,
        "candidate_answer_tokens": candidate_tokens,
        "token_cost_ratio": token_ratio,
        "thresholds": {
            "min_mean_delta": min_mean_delta,
            "min_improved": min_improved,
            "max_regressions": max_regressions,
            "max_token_cost_ratio": max_token_cost_ratio,
            "comparison_epsilon": epsilon,
        },
        "checks": checks,
        "passed": not failures,
        "failure_reasons": failures,
        "pairs": pairs,
        "baseline_evaluation": str(baseline_path),
        "baseline_evaluation_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
        "candidate_evaluation": str(candidate_path),
        "candidate_evaluation_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
    }


def evaluate_validation_gate(
    *,
    suite_path: Path,
    candidate_audit_path: Path,
    candidate_test_audit_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    suite_path = Path(suite_path).resolve()
    suite = _load_object(suite_path, "validation suite")
    audit = _load_object(candidate_audit_path, "candidate audit")
    test_audit = _load_object(candidate_test_audit_path, "candidate test audit")
    plan_bytes = Path(plan_path).read_bytes()
    plan_document = json.loads(plan_bytes)
    if not isinstance(plan_document, dict):
        raise ValueError("modification plan must be a JSON object")
    if suite.get("schema_version") != "deepread-validation-suite-v1":
        raise ValueError("unsupported validation suite schema")
    _exact(
        suite,
        {
            "schema_version", "candidate_id", "plan_id",
            "candidate_snapshot_sha256", "gate_level", "comparison_epsilon",
            "cohorts",
        },
        "validation suite",
    )
    if audit.get("schema_version") != "deepread-candidate-audit-v1":
        raise ValueError("unsupported candidate audit schema")
    if test_audit.get("schema_version") != "deepread-candidate-test-audit-v1":
        raise ValueError("unsupported candidate test audit schema")
    if plan_document.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")
    if not audit.get("passed"):
        raise ValueError("candidate static audit must pass before behavior validation")
    if not test_audit.get("passed"):
        raise ValueError("candidate fixed tests must pass before behavior validation")
    candidate_audit_sha256 = hashlib.sha256(
        Path(candidate_audit_path).read_bytes()
    ).hexdigest()
    if test_audit.get("candidate_audit_sha256") != candidate_audit_sha256:
        raise ValueError("candidate test audit is not bound to the static audit")
    if suite.get("candidate_id") != audit.get("candidate_id"):
        raise ValueError("validation suite candidate_id does not match audit")
    if suite.get("plan_id") != audit.get("plan_id"):
        raise ValueError("validation suite plan_id does not match audit")
    if suite.get("candidate_snapshot_sha256") != audit.get("candidate_snapshot_sha256"):
        raise ValueError("validation suite candidate snapshot does not match audit")
    for field in (
        "candidate_id", "plan_id", "plan_sha256", "candidate_snapshot_sha256",
        "test_policy_id", "test_policy_sha256",
    ):
        if test_audit.get(field) != audit.get(field):
            raise ValueError(f"candidate test audit {field} does not match static audit")
    if hashlib.sha256(plan_bytes).hexdigest() != audit.get("plan_sha256"):
        raise ValueError("modification plan hash does not match candidate audit")
    plans = [
        item
        for item in plan_document.get("plans") or []
        if item.get("plan_id") == suite.get("plan_id")
    ]
    if len(plans) != 1 or plans[0].get("decision") != "proceed":
        raise ValueError("validation suite requires one proceeding plan")

    epsilon = float(suite.get("comparison_epsilon", 1e-9))
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("comparison_epsilon must be non-negative")
    gate_level = str(suite.get("gate_level") or "")
    if gate_level not in {"development", "promotion"}:
        raise ValueError("gate_level must be development or promotion")
    cohort_specs = suite.get("cohorts") or []
    if not isinstance(cohort_specs, list) or not cohort_specs:
        raise ValueError("validation suite requires at least one cohort")
    names = [str(item.get("name") or "") for item in cohort_specs]
    if len(names) != len(set(names)):
        raise ValueError("validation suite cohort names must be unique")
    cohorts = [
        _pair_cohort(item, suite_root=suite_path.parent, epsilon=epsilon)
        for item in cohort_specs
    ]
    roles = {item["role"] for item in cohorts}
    required_roles = {"development", "holdout"}
    if gate_level == "promotion":
        required_roles.add("cross_dataset")
    missing_roles = required_roles - roles
    if missing_roles:
        raise ValueError(f"validation suite is missing required roles: {sorted(missing_roles)}")

    development_keys = {
        (item["dataset"], pair["task_id"])
        for item in cohorts
        if item["role"] == "development"
        for pair in item["pairs"]
    }
    holdout_keys = {
        (item["dataset"], pair["task_id"])
        for item in cohorts
        if item["role"] != "development"
        for pair in item["pairs"]
    }
    overlap = sorted(development_keys & holdout_keys)
    if overlap:
        raise ValueError(f"development tasks leak into validation cohorts: {overlap}")
    failures = [item["name"] for item in cohorts if not item["passed"]]
    return {
        "schema_version": "deepread-validation-gate-v1",
        "candidate_id": suite.get("candidate_id"),
        "plan_id": suite.get("plan_id"),
        "gate_level": gate_level,
        "candidate_snapshot_sha256": audit.get("candidate_snapshot_sha256"),
        "test_policy_id": test_audit.get("test_policy_id"),
        "test_policy_sha256": test_audit.get("test_policy_sha256"),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "validation_suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "candidate_audit_sha256": candidate_audit_sha256,
        "candidate_test_audit_sha256": hashlib.sha256(
            Path(candidate_test_audit_path).read_bytes()
        ).hexdigest(),
        "passed": not failures,
        "failed_cohorts": failures,
        "cohorts": cohorts,
    }
