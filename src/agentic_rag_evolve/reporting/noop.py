"""Compact terminal report for an iteration that creates no candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .iteration import ITERATION_REPORT_SCHEMA


def _read(path: Path, schema: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"unsupported {label} schema")
    return value, {"path": str(resolved), "sha256": hashlib.sha256(data).hexdigest()}


def build_no_candidate_iteration_report(
    *, cohort_path: Path, hypotheses_path: Path, plan_path: Path, output_path: Path
) -> dict[str, Any]:
    cohort, cohort_ref = _read(cohort_path, "deepread-hypothesis-cohort-v1", "cohort")
    hypotheses, hypotheses_ref = _read(
        hypotheses_path, "deepread-improvement-hypotheses-v1", "hypotheses"
    )
    plan, plan_ref = _read(plan_path, "deepread-modification-plan-v1", "plan")
    cohort_id = str(cohort.get("cohort_id") or "")
    if not cohort_id or hypotheses.get("cohort_id") != cohort_id or plan.get("cohort_id") != cohort_id:
        raise ValueError("no-candidate artifacts have inconsistent cohort IDs")
    plans = plan.get("plans") or []
    if any(isinstance(item, dict) and item.get("decision") == "proceed" for item in plans):
        raise ValueError("no-candidate report cannot contain a proceeding plan")
    basis = {
        "cohort_sha256": cohort_ref["sha256"],
        "hypotheses_sha256": hypotheses_ref["sha256"],
        "plan_sha256": plan_ref["sha256"],
        "outcome": "no_candidate",
    }
    digest = hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": ITERATION_REPORT_SCHEMA,
        "iteration_id": "iteration-" + digest[:20],
        "diagnosis": {
            "cohort_id": cohort_id,
            "input_count": int((cohort.get("counts") or {}).get("input") or 0),
            "eligible_count": len(cohort.get("eligible_diagnoses") or []),
            "excluded_count": len(cohort.get("excluded_diagnoses") or []),
            "task_ids_exposed": False,
        },
        "hypotheses": {
            "status": hypotheses.get("status"),
            "hypothesis_count": len(hypotheses.get("hypotheses") or []),
            "task_ids_exposed": False,
        },
        "plan": {
            "plan_count": len(plans),
            "proceeding_plan_count": 0,
        },
        "candidate": None,
        "validation": None,
        "decision": {
            "outcome": "no_candidate",
            "reason": (
                "no_repair_eligible_diagnoses"
                if not cohort.get("eligible_diagnoses")
                else "all_modification_plans_deferred"
            ),
            "baseline_changed": False,
        },
        "terminal": {"status": "completed_without_candidate"},
        "artifacts": {
            "hypothesis_cohort": cohort_ref,
            "improvement_hypotheses": hypotheses_ref,
            "modification_plan": plan_ref,
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
