"""Deterministic repair-cohort selection before semantic hypothesis aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_diagnosis(path: Path) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    try:
        diagnosis = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid diagnosis JSON: {path}") from exc
    if not isinstance(diagnosis, Mapping):
        raise ValueError(f"diagnosis must be an object: {path}")
    if diagnosis.get("schema_version") != "deepread-diagnosis-v1":
        raise ValueError(f"unsupported diagnosis schema: {path}")
    task_id = str(diagnosis.get("task_id") or "").strip()
    if not task_id:
        raise ValueError(f"diagnosis has no task_id: {path}")
    status = diagnosis.get("status")
    if status not in {"diagnosed", "not_agent_failure", "insufficient_evidence"}:
        raise ValueError(f"unsupported diagnosis status {status!r}: {path}")
    return dict(diagnosis), hashlib.sha256(data).hexdigest()


def _eligible_record(
    diagnosis: Mapping[str, Any], *, source: str, sha256: str
) -> dict[str, Any]:
    return {
        "task_id": diagnosis["task_id"],
        "source": source,
        "sha256": sha256,
        "failure_manifestation": diagnosis.get("failure_manifestation"),
        "earliest_intervention": diagnosis.get("earliest_intervention"),
        "root_cause_hypothesis": diagnosis.get("root_cause_hypothesis"),
        "supporting_evidence": diagnosis.get("supporting_evidence") or [],
        "contradicting_evidence": diagnosis.get("contradicting_evidence") or [],
        "counterfactual": diagnosis.get("counterfactual"),
        "affected_sources": diagnosis.get("affected_sources") or [],
        "uncertainties": diagnosis.get("uncertainties") or [],
    }


def _excluded_record(
    diagnosis: Mapping[str, Any], *, source: str, sha256: str
) -> dict[str, Any]:
    status = str(diagnosis["status"])
    route = "evaluation_review" if status == "not_agent_failure" else "evidence_review"
    reason = (
        "Diagnosis found no evolvable DeepRead defect."
        if status == "not_agent_failure"
        else "Diagnosis lacks enough grounded evidence for repair planning."
    )
    return {
        "task_id": diagnosis["task_id"],
        "source": source,
        "sha256": sha256,
        "status": status,
        "route": route,
        "reason": reason,
        "root_cause_hypothesis": diagnosis.get("root_cause_hypothesis"),
        "uncertainties": diagnosis.get("uncertainties") or [],
    }


def build_hypothesis_cohort(diagnosis_paths: Sequence[Path]) -> dict[str, Any]:
    """Select repair-eligible diagnoses without inventing semantic clusters."""

    if not diagnosis_paths:
        raise ValueError("at least one diagnosis is required")
    loaded = []
    seen_task_ids: set[str] = set()
    for input_path in diagnosis_paths:
        path = Path(input_path)
        diagnosis, sha256 = _load_diagnosis(path)
        task_id = str(diagnosis["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate diagnosis task_id: {task_id}")
        seen_task_ids.add(task_id)
        loaded.append((task_id, str(path), diagnosis, sha256))
    loaded.sort(key=lambda item: item[0])

    eligible = []
    excluded = []
    identity_parts = []
    for task_id, source, diagnosis, sha256 in loaded:
        identity_parts.append(f"{task_id}:{sha256}")
        if diagnosis["status"] == "diagnosed":
            eligible.append(
                _eligible_record(diagnosis, source=source, sha256=sha256)
            )
        else:
            excluded.append(
                _excluded_record(diagnosis, source=source, sha256=sha256)
            )

    cohort_id = hashlib.sha256("\n".join(identity_parts).encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": "deepread-hypothesis-cohort-v1",
        "cohort_id": cohort_id,
        "counts": {
            "input": len(loaded),
            "eligible": len(eligible),
            "excluded": len(excluded),
        },
        "eligible_diagnoses": eligible,
        "excluded_diagnoses": excluded,
    }


def build_hypothesis_cohort_with_audits(
    diagnosis_paths: Sequence[Path], diagnosis_audit_paths: Sequence[Path]
) -> dict[str, Any]:
    """Include model-skipped diagnosis audits as explicitly excluded cases."""

    records: list[tuple[str, str, dict[str, Any], str, bool]] = []
    for input_path in diagnosis_paths:
        path = Path(input_path)
        diagnosis, sha256 = _load_diagnosis(path)
        records.append((str(diagnosis["task_id"]), str(path), diagnosis, sha256, False))
    for input_path in diagnosis_audit_paths:
        path = Path(input_path)
        data = path.read_bytes()
        audit = json.loads(data)
        if not isinstance(audit, Mapping) or audit.get("schema_version") != "deepread-diagnosis-audit-v1":
            raise ValueError(f"unsupported diagnosis audit schema: {path}")
        if audit.get("status") != "skipped":
            raise ValueError(f"only skipped diagnosis audits may enter cohort exclusion: {path}")
        task_id = str(audit.get("task_id") or "").strip()
        route = audit.get("route") or {}
        if not task_id or not isinstance(route, Mapping) or not route.get("target"):
            raise ValueError(f"skipped diagnosis audit is incomplete: {path}")
        records.append((task_id, str(path), dict(audit), hashlib.sha256(data).hexdigest(), True))
    if not records:
        raise ValueError("at least one diagnosis or skipped diagnosis audit is required")
    records.sort(key=lambda item: item[0])
    seen: set[str] = set()
    eligible = []
    excluded = []
    identity_parts = []
    for task_id, source, value, sha256, skipped in records:
        if task_id in seen:
            raise ValueError(f"duplicate diagnosis task_id: {task_id}")
        seen.add(task_id)
        identity_parts.append(f"{task_id}:{sha256}")
        if skipped:
            route = value.get("route") or {}
            excluded.append(
                {
                    "task_id": task_id,
                    "source": source,
                    "sha256": sha256,
                    "status": "diagnosis_skipped",
                    "route": str(route.get("target")),
                    "reason": str(value.get("reason") or route.get("reason") or "Diagnosis was not authorized."),
                    "root_cause_hypothesis": None,
                    "uncertainties": [],
                }
            )
        elif value["status"] == "diagnosed":
            eligible.append(_eligible_record(value, source=source, sha256=sha256))
        else:
            excluded.append(_excluded_record(value, source=source, sha256=sha256))
    cohort_id = hashlib.sha256("\n".join(identity_parts).encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": "deepread-hypothesis-cohort-v1",
        "cohort_id": cohort_id,
        "counts": {"input": len(records), "eligible": len(eligible), "excluded": len(excluded)},
        "eligible_diagnoses": eligible,
        "excluded_diagnoses": excluded,
    }


def write_hypothesis_cohort(
    diagnosis_paths: Sequence[Path], output_path: Path,
    diagnosis_audit_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite hypothesis cohort: {output_path}")
    cohort = build_hypothesis_cohort_with_audits(
        diagnosis_paths, diagnosis_audit_paths
    ) if diagnosis_audit_paths else build_hypothesis_cohort(diagnosis_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return cohort
