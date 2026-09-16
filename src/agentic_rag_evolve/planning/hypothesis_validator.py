"""Grounding and partition validation for improvement hypotheses."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


class HypothesisValidationError(ValueError):
    """A hypothesis result is malformed or not grounded in its cohort."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HypothesisValidationError(f"{field} must be an object")
    return value


def _list(value: Any, field: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list):
        raise HypothesisValidationError(f"{field} must be a list")
    if len(value) > maximum:
        raise HypothesisValidationError(f"{field} exceeds {maximum} items")
    return value


def _text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HypothesisValidationError(f"{field} must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        raise HypothesisValidationError(f"{field} exceeds {maximum} characters")
    return text


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        raise HypothesisValidationError(
            f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string_list(value: Any, field: str, maximum: int) -> list[str]:
    return [
        _text(item, f"{field}[{index}]", 500)
        for index, item in enumerate(_list(value, field, maximum))
    ]


def _evidence_ref(
    value: Any,
    *,
    field: str,
    diagnoses: Mapping[str, Mapping[str, Any]],
    member_ids: set[str],
    expected_side: str,
) -> dict[str, Any]:
    ref = dict(_object(value, field))
    _exact(ref, {"task_id", "side", "index", "rationale"}, field)
    task_id = str(ref.get("task_id") or "")
    if task_id not in member_ids:
        raise HypothesisValidationError(f"{field} references non-member task {task_id!r}")
    side = str(ref.get("side") or "")
    key = {"supporting": "supporting_evidence", "contradicting": "contradicting_evidence"}.get(side)
    if key is None:
        raise HypothesisValidationError(f"{field}.side is unsupported: {side!r}")
    if side != expected_side:
        raise HypothesisValidationError(
            f"{field}.side must be {expected_side!r}, got {side!r}"
        )
    try:
        index = int(ref.get("index"))
    except (TypeError, ValueError) as exc:
        raise HypothesisValidationError(f"{field}.index must be an integer") from exc
    evidence = diagnoses[task_id].get(key) or []
    if index < 0 or index >= len(evidence):
        raise HypothesisValidationError(f"{field} references unknown evidence index")
    ref["task_id"] = task_id
    ref["side"] = side
    ref["index"] = index
    ref["rationale"] = _text(ref.get("rationale"), f"{field}.rationale", 500)
    return ref


def _source_ref(
    value: Any,
    *,
    field: str,
    diagnoses: Mapping[str, Mapping[str, Any]],
    member_ids: set[str],
) -> dict[str, Any]:
    ref = dict(_object(value, field))
    _exact(ref, {"task_id", "index", "rationale"}, field)
    task_id = str(ref.get("task_id") or "")
    if task_id not in member_ids:
        raise HypothesisValidationError(f"{field} references non-member task {task_id!r}")
    try:
        index = int(ref.get("index"))
    except (TypeError, ValueError) as exc:
        raise HypothesisValidationError(f"{field}.index must be an integer") from exc
    sources = diagnoses[task_id].get("affected_sources") or []
    if index < 0 or index >= len(sources):
        raise HypothesisValidationError(f"{field} references unknown source index")
    return {
        "task_id": task_id,
        "index": index,
        "rationale": _text(ref.get("rationale"), f"{field}.rationale", 500),
    }


def validate_hypotheses(value: Any, *, cohort: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_object(value, "hypotheses"))
    _exact(
        result,
        {"schema_version", "cohort_id", "status", "hypotheses", "unclustered_task_ids"},
        "hypotheses",
    )
    if result.get("schema_version") != "deepread-improvement-hypotheses-v1":
        raise HypothesisValidationError("unsupported hypothesis schema_version")
    if str(result.get("cohort_id") or "") != str(cohort.get("cohort_id") or ""):
        raise HypothesisValidationError("cohort_id does not match input")
    if result.get("status") not in {"ready", "no_hypotheses"}:
        raise HypothesisValidationError("status must be ready or no_hypotheses")

    diagnoses = {
        str(item["task_id"]): item for item in cohort.get("eligible_diagnoses") or []
    }
    assigned: set[str] = set()
    normalized_hypotheses = []
    raw_hypotheses = _list(result.get("hypotheses"), "hypotheses", 12)
    for h_index, raw in enumerate(raw_hypotheses):
        field = f"hypotheses[{h_index}]"
        hypothesis = dict(_object(raw, field))
        expected_fields = {
            "title", "task_ids", "common_mechanism", "earliest_intervention_pattern",
            "target_cohort", "behavior_delta", "supporting_evidence_refs",
            "contradicting_evidence_refs", "affected_source_refs", "validation",
            "uncertainties",
        }
        _exact(hypothesis, expected_fields, field)
        task_ids = [str(item) for item in _list(hypothesis.get("task_ids"), f"{field}.task_ids", 100)]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise HypothesisValidationError(f"{field}.task_ids must be non-empty and unique")
        unknown = set(task_ids) - set(diagnoses)
        overlap = set(task_ids) & assigned
        if unknown:
            raise HypothesisValidationError(f"{field} references unknown tasks: {sorted(unknown)}")
        if overlap:
            raise HypothesisValidationError(f"tasks assigned to multiple hypotheses: {sorted(overlap)}")
        member_ids = set(task_ids)
        assigned.update(member_ids)

        supporting = [
            _evidence_ref(item, field=f"{field}.supporting_evidence_refs[{index}]", diagnoses=diagnoses, member_ids=member_ids, expected_side="supporting")
            for index, item in enumerate(_list(hypothesis.get("supporting_evidence_refs"), f"{field}.supporting_evidence_refs", 24))
        ]
        if {item["task_id"] for item in supporting} != member_ids:
            raise HypothesisValidationError(f"{field} needs supporting evidence from every member task")
        contradicting = [
            _evidence_ref(item, field=f"{field}.contradicting_evidence_refs[{index}]", diagnoses=diagnoses, member_ids=member_ids, expected_side="contradicting")
            for index, item in enumerate(_list(hypothesis.get("contradicting_evidence_refs"), f"{field}.contradicting_evidence_refs", 24))
        ]
        if not contradicting:
            raise HypothesisValidationError(f"{field} requires contradicting evidence")
        sources = [
            _source_ref(item, field=f"{field}.affected_source_refs[{index}]", diagnoses=diagnoses, member_ids=member_ids)
            for index, item in enumerate(_list(hypothesis.get("affected_source_refs"), f"{field}.affected_source_refs", 12))
        ]
        if not sources:
            raise HypothesisValidationError(f"{field} requires an affected source reference")

        target = dict(_object(hypothesis.get("target_cohort"), f"{field}.target_cohort"))
        _exact(target, {"inclusion_signals", "exclusion_signals"}, f"{field}.target_cohort")
        target = {
            "inclusion_signals": _string_list(target.get("inclusion_signals"), f"{field}.target_cohort.inclusion_signals", 8),
            "exclusion_signals": _string_list(target.get("exclusion_signals"), f"{field}.target_cohort.exclusion_signals", 8),
        }
        if not target["inclusion_signals"] or not target["exclusion_signals"]:
            raise HypothesisValidationError(
                f"{field}.target_cohort requires inclusion and exclusion signals"
            )
        validation = dict(_object(hypothesis.get("validation"), f"{field}.validation"))
        _exact(validation, {"expected_observation", "falsifier", "regression_guards"}, f"{field}.validation")
        validation = {
            "expected_observation": _text(validation.get("expected_observation"), f"{field}.validation.expected_observation", 800),
            "falsifier": _text(validation.get("falsifier"), f"{field}.validation.falsifier", 800),
            "regression_guards": _string_list(validation.get("regression_guards"), f"{field}.validation.regression_guards", 8),
        }
        if not validation["regression_guards"]:
            raise HypothesisValidationError(
                f"{field}.validation requires at least one regression guard"
            )
        stable_key = "\n".join(sorted(task_ids))
        normalized_hypotheses.append(
            {
                "hypothesis_id": "hyp_" + hashlib.sha256(stable_key.encode()).hexdigest()[:12],
                "maturity": "recurring" if len(task_ids) >= 2 else "singleton",
                "title": _text(hypothesis.get("title"), f"{field}.title", 160),
                "task_ids": sorted(task_ids),
                "common_mechanism": _text(hypothesis.get("common_mechanism"), f"{field}.common_mechanism", 1500),
                "earliest_intervention_pattern": _text(hypothesis.get("earliest_intervention_pattern"), f"{field}.earliest_intervention_pattern", 800),
                "target_cohort": target,
                "behavior_delta": _text(hypothesis.get("behavior_delta"), f"{field}.behavior_delta", 1200),
                "supporting_evidence_refs": supporting,
                "contradicting_evidence_refs": contradicting,
                "affected_source_refs": sources,
                "validation": validation,
                "uncertainties": _string_list(hypothesis.get("uncertainties"), f"{field}.uncertainties", 8),
            }
        )

    unclustered = [str(item) for item in _list(result.get("unclustered_task_ids"), "unclustered_task_ids", 1000)]
    if len(unclustered) != len(set(unclustered)):
        raise HypothesisValidationError("unclustered_task_ids contains duplicates")
    unknown_unclustered = set(unclustered) - set(diagnoses)
    if unknown_unclustered:
        raise HypothesisValidationError(f"unclustered_task_ids contains unknown tasks: {sorted(unknown_unclustered)}")
    overlap = assigned & set(unclustered)
    if overlap:
        raise HypothesisValidationError(f"tasks are both clustered and unclustered: {sorted(overlap)}")
    uncovered = set(diagnoses) - assigned - set(unclustered)
    if uncovered:
        raise HypothesisValidationError(f"eligible tasks are not accounted for: {sorted(uncovered)}")
    if result["status"] == "ready" and not normalized_hypotheses:
        raise HypothesisValidationError("ready status requires at least one hypothesis")
    if result["status"] == "no_hypotheses" and normalized_hypotheses:
        raise HypothesisValidationError("no_hypotheses status requires an empty hypothesis list")
    result["hypotheses"] = normalized_hypotheses
    result["unclustered_task_ids"] = sorted(unclustered)
    return result
