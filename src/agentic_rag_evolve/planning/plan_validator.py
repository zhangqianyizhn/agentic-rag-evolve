"""Validation and source-scope resolution for modification plans."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .memory import CONTEXT_SCHEMA, attempt_fingerprint


FORBIDDEN_ROOTS = [
    ".env",
    "benchmarks/",
    "docs/",
    "runner/",
    "src/agentic_rag_evolve/",
    "tests/",
]


class ModificationPlanValidationError(ValueError):
    """A modification plan exceeds its hypothesis or validation boundary."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModificationPlanValidationError(f"{field} must be an object")
    return value


def _list(value: Any, field: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ModificationPlanValidationError(f"{field} must be a list")
    if len(value) > maximum:
        raise ModificationPlanValidationError(f"{field} exceeds {maximum} items")
    return value


def _text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModificationPlanValidationError(f"{field} must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        raise ModificationPlanValidationError(f"{field} exceeds {maximum} characters")
    return text


def _strings(value: Any, field: str, maximum: int, *, required: bool = True) -> list[str]:
    result = [
        _text(item, f"{field}[{index}]", 600)
        for index, item in enumerate(_list(value, field, maximum))
    ]
    if required and not result:
        raise ModificationPlanValidationError(f"{field} must not be empty")
    return result


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ModificationPlanValidationError(
            f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _source_key(value: Mapping[str, Any]) -> tuple[str, int]:
    task_id = str(value.get("task_id") or "")
    try:
        index = int(value.get("index"))
    except (TypeError, ValueError) as exc:
        raise ModificationPlanValidationError("source ref index must be an integer") from exc
    return task_id, index


def _resolve_source_ref(
    value: Any,
    *,
    field: str,
    permitted: set[tuple[str, int]],
    diagnoses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ref = dict(_object(value, field))
    _exact(ref, {"task_id", "index", "rationale"}, field)
    key = _source_key(ref)
    if key not in permitted:
        raise ModificationPlanValidationError(
            f"{field} is not an affected source ref from the hypothesis"
        )
    task_id, index = key
    sources = diagnoses[task_id].get("affected_sources") or []
    if index < 0 or index >= len(sources):
        raise ModificationPlanValidationError(f"{field} references unknown source")
    source = sources[index]
    path = _text(source.get("path"), f"{field}.resolved.path", 300)
    if not path.startswith("systems/deepread/DeepRead/"):
        raise ModificationPlanValidationError(
            f"{field} resolves outside the evolvable DeepRead root"
        )
    return {
        "task_id": task_id,
        "index": index,
        "path": path,
        "symbol": _text(source.get("symbol"), f"{field}.resolved.symbol", 200),
        "rationale": _text(ref.get("rationale"), f"{field}.rationale", 500),
    }


def validate_modification_plan(
    value: Any,
    *,
    cohort: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    memory_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(_object(value, "modification_plan"))
    _exact(
        result,
        {"schema_version", "cohort_id", "hypothesis_set_status", "plans"},
        "modification_plan",
    )
    if result.get("schema_version") != "deepread-modification-plan-v1":
        raise ModificationPlanValidationError("unsupported modification plan schema")
    cohort_id = str(cohort.get("cohort_id") or "")
    if str(result.get("cohort_id") or "") != cohort_id:
        raise ModificationPlanValidationError("cohort_id does not match input")
    if result.get("hypothesis_set_status") != hypotheses.get("status"):
        raise ModificationPlanValidationError("hypothesis_set_status does not match input")

    prior_fingerprints: dict[str, set[str]] = {}
    if memory_context is not None:
        if memory_context.get("schema_version") != CONTEXT_SCHEMA:
            raise ModificationPlanValidationError("unsupported planning memory context schema")
        if memory_context.get("cohort_id") != cohort_id:
            raise ModificationPlanValidationError("memory context cohort_id does not match input")
        for entry in memory_context.get("entries") or []:
            hypothesis_id = str(entry.get("hypothesis_id") or "")
            fingerprints = prior_fingerprints.setdefault(hypothesis_id, set())
            for failure in entry.get("relevant_failures") or []:
                fingerprint = str((failure.get("attempt") or {}).get("fingerprint") or "")
                if fingerprint:
                    fingerprints.add(fingerprint)

    hypothesis_by_id = {
        str(item["hypothesis_id"]): item for item in hypotheses.get("hypotheses") or []
    }
    diagnoses = {
        str(item["task_id"]): item for item in cohort.get("eligible_diagnoses") or []
    }
    seen: set[str] = set()
    normalized = []
    for index, raw in enumerate(_list(result.get("plans"), "plans", 12)):
        field = f"plans[{index}]"
        plan = dict(_object(raw, field))
        _exact(
            plan,
            {
                "hypothesis_id", "decision", "rationale", "allowed_source_refs",
                "change_contract", "validation_plan", "risk",
            },
            field,
        )
        hypothesis_id = str(plan.get("hypothesis_id") or "")
        if hypothesis_id not in hypothesis_by_id:
            raise ModificationPlanValidationError(f"{field} references unknown hypothesis")
        if hypothesis_id in seen:
            raise ModificationPlanValidationError(f"duplicate plan for {hypothesis_id}")
        seen.add(hypothesis_id)
        hypothesis = hypothesis_by_id[hypothesis_id]
        decision = str(plan.get("decision") or "")
        if decision not in {"proceed", "defer"}:
            raise ModificationPlanValidationError(f"{field}.decision is unsupported")
        if hypothesis.get("maturity") == "singleton" and decision != "defer":
            raise ModificationPlanValidationError(
                f"{field}: singleton hypothesis cannot proceed automatically"
            )

        permitted = {
            _source_key(item) for item in hypothesis.get("affected_source_refs") or []
        }
        source_refs = [
            _resolve_source_ref(
                item,
                field=f"{field}.allowed_source_refs[{ref_index}]",
                permitted=permitted,
                diagnoses=diagnoses,
            )
            for ref_index, item in enumerate(
                _list(plan.get("allowed_source_refs"), f"{field}.allowed_source_refs", 12)
            )
        ]
        source_keys = [(item["task_id"], item["index"]) for item in source_refs]
        if len(source_keys) != len(set(source_keys)):
            raise ModificationPlanValidationError(
                f"{field}.allowed_source_refs contains duplicates"
            )
        if decision == "proceed" and not source_refs:
            raise ModificationPlanValidationError(
                f"{field}.allowed_source_refs is required when proceeding"
            )

        contract = dict(_object(plan.get("change_contract"), f"{field}.change_contract"))
        _exact(
            contract,
            {"current_behavior", "required_behavior_delta", "must_preserve", "non_goals"},
            f"{field}.change_contract",
        )
        contract = {
            "current_behavior": _text(contract.get("current_behavior"), f"{field}.change_contract.current_behavior", 1000),
            "required_behavior_delta": _text(contract.get("required_behavior_delta"), f"{field}.change_contract.required_behavior_delta", 1200),
            "must_preserve": _strings(contract.get("must_preserve"), f"{field}.change_contract.must_preserve", 10),
            "non_goals": _strings(contract.get("non_goals"), f"{field}.change_contract.non_goals", 10),
        }

        validation = dict(_object(plan.get("validation_plan"), f"{field}.validation_plan"))
        _exact(
            validation,
            {
                "development_task_ids", "holdout_selection_rules",
                "expected_observations", "rollback_conditions",
            },
            f"{field}.validation_plan",
        )
        development_ids = [
            str(item)
            for item in _list(
                validation.get("development_task_ids"),
                f"{field}.validation_plan.development_task_ids",
                100,
            )
        ]
        if len(development_ids) != len(set(development_ids)):
            raise ModificationPlanValidationError(f"{field} has duplicate development tasks")
        hypothesis_tasks = set(hypothesis.get("task_ids") or [])
        if set(development_ids) - hypothesis_tasks:
            raise ModificationPlanValidationError(
                f"{field} development tasks are outside the hypothesis cohort"
            )
        if decision == "proceed" and set(development_ids) != hypothesis_tasks:
            raise ModificationPlanValidationError(
                f"{field} proceeding plan must validate every hypothesis task"
            )
        validation = {
            "development_task_ids": sorted(development_ids),
            "holdout_selection_rules": _strings(validation.get("holdout_selection_rules"), f"{field}.validation_plan.holdout_selection_rules", 8),
            "expected_observations": _strings(validation.get("expected_observations"), f"{field}.validation_plan.expected_observations", 10),
            "rollback_conditions": _strings(validation.get("rollback_conditions"), f"{field}.validation_plan.rollback_conditions", 10),
        }

        risk = dict(_object(plan.get("risk"), f"{field}.risk"))
        _exact(risk, {"level", "regression_scenarios"}, f"{field}.risk")
        level = str(risk.get("level") or "")
        if level not in {"low", "medium", "high"}:
            raise ModificationPlanValidationError(f"{field}.risk.level is unsupported")
        unique_paths = sorted({item["path"] for item in source_refs})
        if decision == "proceed":
            fingerprint = attempt_fingerprint(
                allowed_paths=unique_paths,
                required_behavior_delta=contract["required_behavior_delta"],
            )
            if fingerprint in prior_fingerprints.get(hypothesis_id, set()):
                raise ModificationPlanValidationError(
                    f"{field} exactly repeats a rejected attempt"
                )
        stable_key = f"{cohort_id}:{hypothesis_id}"
        normalized.append(
            {
                "plan_id": "plan_" + hashlib.sha256(stable_key.encode()).hexdigest()[:12],
                "hypothesis_id": hypothesis_id,
                "decision": decision,
                "rationale": _text(plan.get("rationale"), f"{field}.rationale", 1000),
                "edit_scope": {
                    "allowed_sources": source_refs,
                    "allowed_paths": unique_paths,
                    "max_files_to_modify": len(unique_paths),
                    "forbidden_roots": list(FORBIDDEN_ROOTS),
                    "must_inspect_before_edit": True,
                },
                "change_contract": contract,
                "validation_plan": validation,
                "risk": {
                    "level": level,
                    "regression_scenarios": _strings(risk.get("regression_scenarios"), f"{field}.risk.regression_scenarios", 10),
                },
            }
        )

    missing = set(hypothesis_by_id) - seen
    if missing:
        raise ModificationPlanValidationError(
            f"hypotheses missing modification plans: {sorted(missing)}"
        )
    result["plans"] = sorted(normalized, key=lambda item: item["hypothesis_id"])
    return result
