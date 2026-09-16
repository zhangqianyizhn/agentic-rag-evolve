"""Append-only outcome records for validated DeepRead candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _require_schema(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"unsupported {label} schema")


def _same(label: str, expected: Any, actual: Any) -> None:
    if not expected or expected != actual:
        raise ValueError(f"{label} does not match candidate lineage")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_candidate_record(
    registry_root: Path, candidate_id: str, snapshot: str
) -> Path | None:
    for path in sorted(Path(registry_root).glob("*/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing candidate registry record: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid existing candidate registry record: {path}")
        if (
            value.get("candidate_id") == candidate_id
            and value.get("candidate_snapshot_sha256") == snapshot
        ):
            return path
    return None


def record_candidate_outcome(
    *,
    registry_root: Path,
    candidate_manifest_path: Path,
    plan_path: Path,
    candidate_audit_path: Path,
    candidate_test_audit_path: Path,
    validation_suite_path: Path,
    validation_gate_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Persist one terminal promotion decision without mutating Git or the candidate."""

    manifest, manifest_sha = _read(candidate_manifest_path, "candidate manifest")
    plan, plan_sha = _read(plan_path, "modification plan")
    audit, audit_sha = _read(candidate_audit_path, "candidate audit")
    test_audit, test_audit_sha = _read(
        candidate_test_audit_path, "candidate test audit"
    )
    suite, suite_sha = _read(validation_suite_path, "validation suite")
    gate, gate_sha = _read(validation_gate_path, "validation gate")
    _require_schema(manifest, "deepread-candidate-manifest-v1", "candidate manifest")
    _require_schema(plan, "deepread-modification-plan-v1", "modification plan")
    _require_schema(audit, "deepread-candidate-audit-v1", "candidate audit")
    _require_schema(
        test_audit, "deepread-candidate-test-audit-v1", "candidate test audit"
    )
    _require_schema(suite, "deepread-validation-suite-v1", "validation suite")
    _require_schema(gate, "deepread-validation-gate-v1", "validation gate")

    candidate_id = str(manifest.get("candidate_id") or "")
    plan_id = str(manifest.get("plan_id") or "")
    snapshot = str(audit.get("candidate_snapshot_sha256") or "")
    if not candidate_id or not plan_id or len(snapshot) != 64:
        raise ValueError("candidate lineage identifiers are incomplete")
    for label, value in (
        ("static audit candidate_id", audit.get("candidate_id")),
        ("test audit candidate_id", test_audit.get("candidate_id")),
        ("validation suite candidate_id", suite.get("candidate_id")),
        ("validation gate candidate_id", gate.get("candidate_id")),
    ):
        _same(label, candidate_id, value)
    for label, value in (
        ("static audit plan_id", audit.get("plan_id")),
        ("test audit plan_id", test_audit.get("plan_id")),
        ("validation suite plan_id", suite.get("plan_id")),
        ("validation gate plan_id", gate.get("plan_id")),
    ):
        _same(label, plan_id, value)
    matching_plans = [
        item for item in plan.get("plans") or [] if item.get("plan_id") == plan_id
    ]
    if len(matching_plans) != 1 or matching_plans[0].get("decision") != "proceed":
        raise ValueError("candidate outcome requires one proceeding plan")

    _same("manifest plan hash", manifest.get("plan_sha256"), plan_sha)
    _same("static audit plan hash", plan_sha, audit.get("plan_sha256"))
    _same("test audit plan hash", plan_sha, test_audit.get("plan_sha256"))
    _same("validation gate plan hash", plan_sha, gate.get("plan_sha256"))
    _same(
        "static audit manifest hash",
        manifest_sha,
        audit.get("candidate_manifest_sha256"),
    )
    _same(
        "static audit base commit", manifest.get("base_commit"), audit.get("base_commit")
    )
    _same("static audit HEAD", manifest.get("base_commit"), audit.get("head_commit"))
    _same(
        "static audit candidate path",
        manifest.get("candidate_path"),
        audit.get("candidate_path"),
    )
    _same(
        "test audit static audit hash",
        audit_sha,
        test_audit.get("candidate_audit_sha256"),
    )
    _same(
        "validation gate static audit hash",
        audit_sha,
        gate.get("candidate_audit_sha256"),
    )
    _same(
        "validation gate test audit hash",
        test_audit_sha,
        gate.get("candidate_test_audit_sha256"),
    )
    _same(
        "validation gate suite hash", suite_sha, gate.get("validation_suite_sha256")
    )
    for label, value in (
        ("test audit snapshot", test_audit.get("candidate_snapshot_sha256")),
        ("validation suite snapshot", suite.get("candidate_snapshot_sha256")),
        ("validation gate snapshot", gate.get("candidate_snapshot_sha256")),
    ):
        _same(label, snapshot, value)
    for label, expected, actual in (
        (
            "manifest/test audit policy id",
            manifest.get("test_policy_id"),
            test_audit.get("test_policy_id"),
        ),
        (
            "manifest/gate policy id",
            manifest.get("test_policy_id"),
            gate.get("test_policy_id"),
        ),
        (
            "manifest/test audit policy hash",
            manifest.get("test_policy_sha256"),
            test_audit.get("test_policy_sha256"),
        ),
        (
            "manifest/gate policy hash",
            manifest.get("test_policy_sha256"),
            gate.get("test_policy_sha256"),
        ),
    ):
        _same(label, expected, actual)
    if not audit.get("passed") or not test_audit.get("passed"):
        raise ValueError("candidate audits must pass before outcome recording")
    if gate.get("gate_level") != "promotion" or suite.get("gate_level") != "promotion":
        raise ValueError("only a promotion-level gate can produce a terminal outcome")

    cohorts = gate.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("validation gate has no cohort results")
    names = [str(item.get("name") or "") for item in cohorts]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("validation gate cohort names must be non-empty and unique")
    roles = {str(item.get("role") or "") for item in cohorts}
    required_roles = {"development", "holdout", "cross_dataset"}
    if not required_roles.issubset(roles):
        raise ValueError("promotion outcome is missing required cohort roles")
    failed = sorted(
        str(item.get("name") or "") for item in cohorts if not item.get("passed")
    )
    declared_failed = sorted(str(item) for item in gate.get("failed_cohorts") or [])
    passed = bool(gate.get("passed"))
    if failed != declared_failed or passed == bool(failed):
        raise ValueError("validation gate decision is internally inconsistent")
    outcome = "accepted" if passed else "rejected"
    reasons = ["promotion_gate_passed"] if passed else [
        f"cohort:{item.get('name')}:{reason}"
        for item in cohorts
        if not item.get("passed")
        for reason in item.get("failure_reasons") or ["failed"]
    ]
    basis = {
        "candidate_id": candidate_id,
        "candidate_snapshot_sha256": snapshot,
        "validation_gate_sha256": gate_sha,
        "outcome": outcome,
    }
    decision_id = "outcome-" + _canonical_sha256(basis)[:20]
    artifacts = {
        "candidate_manifest": {
            "path": str(Path(candidate_manifest_path).resolve()),
            "sha256": manifest_sha,
        },
        "modification_plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": plan_sha,
        },
        "candidate_audit": {
            "path": str(Path(candidate_audit_path).resolve()),
            "sha256": audit_sha,
        },
        "candidate_test_audit": {
            "path": str(Path(candidate_test_audit_path).resolve()),
            "sha256": test_audit_sha,
        },
        "validation_suite": {
            "path": str(Path(validation_suite_path).resolve()),
            "sha256": suite_sha,
        },
        "validation_gate": {
            "path": str(Path(validation_gate_path).resolve()),
            "sha256": gate_sha,
        },
    }
    record = {
        "schema_version": "deepread-candidate-outcome-v1",
        "decision_id": decision_id,
        "candidate_id": candidate_id,
        "plan_id": plan_id,
        "outcome": outcome,
        "candidate_status": outcome,
        "promotion_status": (
            "eligible_for_materialization" if passed else "not_eligible"
        ),
        "base_commit": manifest.get("base_commit"),
        "candidate_path": manifest.get("candidate_path"),
        "candidate_snapshot_sha256": snapshot,
        "decision_reasons": reasons,
        "failed_cohorts": failed,
        "cohort_summary": [
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
            for item in cohorts
        ],
        "artifacts": artifacts,
        "next_action": (
            "materialize_accepted_candidate" if passed else "retain_for_failure_memory"
        ),
        "git_mutation_performed": False,
        "decision_basis_sha256": _canonical_sha256(basis),
    }

    registry_root = Path(registry_root).resolve()
    existing = _existing_candidate_record(registry_root, candidate_id, snapshot)
    if existing is not None:
        raise FileExistsError(
            f"candidate snapshot already has an outcome record: {existing}"
        )
    destination = registry_root / outcome / f"{decision_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record, destination
