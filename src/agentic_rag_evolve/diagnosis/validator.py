"""Strict validation for open-ended, evidence-anchored DeepRead diagnoses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DIAGNOSIS_STATUSES = {"diagnosed", "not_agent_failure", "insufficient_evidence"}
EVIDENCE_LAYERS = {"corpus", "candidate", "read", "answer"}
TOP_LEVEL_FIELDS = {
    "schema_version",
    "task_id",
    "status",
    "failure_manifestation",
    "earliest_intervention",
    "root_cause_hypothesis",
    "supporting_evidence",
    "contradicting_evidence",
    "counterfactual",
    "affected_sources",
    "uncertainties",
}


class DiagnosisValidationError(ValueError):
    """A diagnosis is structurally valid JSON but not adequately grounded."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosisValidationError(f"{field} must be an object")
    return value


def _list(value: Any, field: str, *, maximum: int = 8) -> Sequence[Any]:
    if not isinstance(value, list):
        raise DiagnosisValidationError(f"{field} must be a list")
    if len(value) > maximum:
        raise DiagnosisValidationError(f"{field} exceeds {maximum} items")
    return value


def _string_list(value: Any, field: str, *, maximum: int = 8) -> Sequence[Any]:
    """Normalize one model-produced string without weakening item validation."""

    if isinstance(value, str):
        value = [value]
    return _list(value, field, maximum=maximum)


def _text(value: Any, field: str, *, maximum: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise DiagnosisValidationError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise DiagnosisValidationError(f"{field} must not be empty")
    if len(text) > maximum:
        raise DiagnosisValidationError(f"{field} exceeds {maximum} characters")
    return text


def _exact_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise DiagnosisValidationError(f"{field} has unknown fields: {sorted(unknown)}")


def _turns(bundle: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    values: dict[int, Mapping[str, Any]] = {}
    for turn in (bundle.get("trajectory") or {}).get("turns") or []:
        round_value = turn.get("round")
        if round_value is not None:
            values[int(round_value)] = turn
    return values


def _validate_turn_reference(
    *, turn: Any, tool_call_id: Any, bundle: Mapping[str, Any], field: str
) -> None:
    try:
        round_number = int(turn)
    except (TypeError, ValueError) as exc:
        raise DiagnosisValidationError(f"{field}.turn must be an integer") from exc
    known_turns = _turns(bundle)
    if round_number not in known_turns:
        raise DiagnosisValidationError(f"{field} references unknown turn {round_number}")
    if tool_call_id is not None:
        known_calls = {
            str(tool.get("call_id"))
            for tool in known_turns[round_number].get("tools") or []
        }
        if str(tool_call_id) not in known_calls:
            raise DiagnosisValidationError(
                f"{field} references unknown tool call {tool_call_id!r} in turn {round_number}"
            )


def _manifest(bundle: Mapping[str, Any], kind: str) -> dict[str, Mapping[str, Any]]:
    plural = "sources" if kind == "source" else "payloads"
    return {
        str(item.get("path")): item
        for item in (bundle.get("access") or {}).get(plural) or []
    }


def _was_read(
    *,
    path: str,
    start: int,
    end: int,
    observed_reads: Sequence[Mapping[str, Any]],
    start_field: str,
    end_field: str,
) -> bool:
    return any(
        str(item.get("path")) == path
        and int(item.get(start_field, -1)) <= start
        and int(item.get(end_field, -1)) >= end
        for item in observed_reads
    )


def _validate_source_range(
    value: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    observed_source_reads: Sequence[Mapping[str, Any]] | None,
    field: str,
) -> None:
    path = _text(value.get("path"), f"{field}.path", maximum=300)
    manifest = _manifest(bundle, "source")
    if path not in manifest:
        raise DiagnosisValidationError(f"{field} references non-allowlisted source {path!r}")
    try:
        start = int(value.get("start_line"))
        end = int(value.get("end_line"))
    except (TypeError, ValueError) as exc:
        raise DiagnosisValidationError(f"{field} source lines must be integers") from exc
    if start < 1 or end < start or end > int(manifest[path].get("line_count", 0)):
        raise DiagnosisValidationError(f"{field} has invalid source line range")
    if observed_source_reads is not None and not _was_read(
        path=path,
        start=start,
        end=end,
        observed_reads=observed_source_reads,
        start_field="start_line",
        end_field="end_line",
    ):
        raise DiagnosisValidationError(f"{field} cites source lines not read by the agent")


def _resolve_field(root: Mapping[str, Any], dotted: str) -> Any:
    value: Any = root
    for part in dotted.split("."):
        if not part or part.startswith("_") or not isinstance(value, Mapping) or part not in value:
            raise DiagnosisValidationError(f"evaluation reference does not exist: {dotted}")
        value = value[part]
    return value


def _validate_anchor(
    value: Any,
    *,
    bundle: Mapping[str, Any],
    observed_source_reads: Sequence[Mapping[str, Any]] | None,
    observed_payload_reads: Sequence[Mapping[str, Any]] | None,
    field: str,
) -> dict[str, Any]:
    anchor = dict(_object(value, field))
    kind = str(anchor.get("kind") or "")
    nested_kinds = ("trajectory", "coverage", "evaluation", "source", "payload")
    nested_kind = kind if kind in nested_kinds else next(
        (
            candidate
            for candidate in nested_kinds
            if isinstance(anchor.get(candidate), Mapping)
        ),
        None,
    )
    if nested_kind is not None and isinstance(anchor.get(nested_kind), Mapping):
        nested = dict(anchor.pop(nested_kind))
        anchor["kind"] = nested_kind
        for key, nested_value in nested.items():
            anchor.setdefault(key, nested_value)
        kind = nested_kind
    if kind == "judge":
        anchor["kind"] = "evaluation"
        judge = anchor.pop("judge", None)
        if isinstance(judge, Mapping):
            for key, nested_value in judge.items():
                anchor.setdefault(key, nested_value)
        referenced_field = str(anchor.get("field") or "reasoning")
        anchor["field"] = (
            referenced_field
            if referenced_field.startswith("judge.")
            else f"judge.{referenced_field}"
        )
        kind = "evaluation"
    allowed = {
        "trajectory": {"kind", "claim", "turn", "tool_call_id"},
        "coverage": {"kind", "claim", "evidence_index", "layer"},
        "evaluation": {"kind", "claim", "field"},
        "source": {"kind", "claim", "path", "start_line", "end_line"},
        "payload": {"kind", "claim", "path", "offset_chars", "end_chars"},
    }
    if kind not in allowed:
        raise DiagnosisValidationError(f"{field}.kind is unsupported: {kind!r}")
    _exact_fields(anchor, allowed[kind], field)
    _text(anchor.get("claim"), f"{field}.claim", maximum=500)

    if kind == "trajectory":
        _validate_turn_reference(
            turn=anchor.get("turn"),
            tool_call_id=anchor.get("tool_call_id"),
            bundle=bundle,
            field=field,
        )
    elif kind == "coverage":
        try:
            index = int(anchor.get("evidence_index"))
        except (TypeError, ValueError) as exc:
            raise DiagnosisValidationError(
                f"{field}.evidence_index must be an integer"
            ) from exc
        evidence = (bundle.get("evidence_coverage") or {}).get("evidence") or []
        if index < 0 or index >= len(evidence):
            raise DiagnosisValidationError(f"{field} references unknown evidence index")
        if anchor.get("layer") not in EVIDENCE_LAYERS:
            raise DiagnosisValidationError(f"{field} references unknown coverage layer")
        if anchor.get("layer") not in (evidence[index].get("layers") or {}):
            raise DiagnosisValidationError(f"{field} references absent coverage data")
    elif kind == "evaluation":
        dotted = _text(anchor.get("field"), f"{field}.field", maximum=120)
        _resolve_field(bundle.get("evaluation") or {}, dotted)
    elif kind == "source":
        _validate_source_range(
            anchor,
            bundle=bundle,
            observed_source_reads=observed_source_reads,
            field=field,
        )
    else:
        path = _text(anchor.get("path"), f"{field}.path", maximum=300)
        manifest = _manifest(bundle, "payload")
        if path not in manifest:
            raise DiagnosisValidationError(f"{field} references unknown payload {path!r}")
        try:
            start = int(anchor.get("offset_chars"))
            end = int(anchor.get("end_chars"))
        except (TypeError, ValueError) as exc:
            raise DiagnosisValidationError(f"{field} payload offsets must be integers") from exc
        if start < 0 or end < start or end > int(manifest[path].get("bytes", 0)):
            raise DiagnosisValidationError(f"{field} has invalid payload range")
        if observed_payload_reads is not None and not _was_read(
            path=path,
            start=start,
            end=end,
            observed_reads=observed_payload_reads,
            start_field="offset_chars",
            end_field="end_chars",
        ):
            raise DiagnosisValidationError(f"{field} cites payload content not read by the agent")
    return anchor


def validate_diagnosis(
    value: Any,
    *,
    bundle: Mapping[str, Any],
    observed_source_reads: Sequence[Mapping[str, Any]] | None = None,
    observed_payload_reads: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a normalized diagnosis or raise on any ungrounded reference."""

    diagnosis = dict(_object(value, "diagnosis"))
    _exact_fields(diagnosis, TOP_LEVEL_FIELDS, "diagnosis")
    if diagnosis.get("schema_version") != "deepread-diagnosis-v1":
        raise DiagnosisValidationError("unsupported diagnosis schema_version")
    task_id = str((bundle.get("task") or {}).get("task_id") or "")
    if str(diagnosis.get("task_id") or "") != task_id:
        raise DiagnosisValidationError("diagnosis task_id does not match bundle")
    status = str(diagnosis.get("status") or "")
    if status not in DIAGNOSIS_STATUSES:
        raise DiagnosisValidationError(f"unsupported diagnosis status: {status!r}")
    _text(
        diagnosis.get("failure_manifestation"),
        "failure_manifestation",
        maximum=1_500,
    )
    _text(
        diagnosis.get("root_cause_hypothesis"),
        "root_cause_hypothesis",
        maximum=1_500,
    )

    earliest = diagnosis.get("earliest_intervention")
    if earliest is not None:
        earliest = dict(_object(earliest, "earliest_intervention"))
        _exact_fields(
            earliest,
            {"turn", "tool_call_id", "rationale"},
            "earliest_intervention",
        )
        _text(earliest.get("rationale"), "earliest_intervention.rationale", maximum=600)
        _validate_turn_reference(
            turn=earliest.get("turn"),
            tool_call_id=earliest.get("tool_call_id"),
            bundle=bundle,
            field="earliest_intervention",
        )
    elif status == "diagnosed":
        raise DiagnosisValidationError("diagnosed result requires earliest_intervention")

    for list_name in ("supporting_evidence", "contradicting_evidence"):
        items = _list(diagnosis.get(list_name), list_name)
        if status == "diagnosed" and not items:
            raise DiagnosisValidationError(f"diagnosed result requires {list_name}")
        diagnosis[list_name] = [
            _validate_anchor(
                item,
                bundle=bundle,
                observed_source_reads=observed_source_reads,
                observed_payload_reads=observed_payload_reads,
                field=f"{list_name}[{index}]",
            )
            for index, item in enumerate(items)
        ]

    counterfactual = dict(_object(diagnosis.get("counterfactual"), "counterfactual"))
    _exact_fields(
        counterfactual,
        {"change", "expected_observation", "falsifier"},
        "counterfactual",
    )
    for field in ("change", "expected_observation", "falsifier"):
        _text(counterfactual.get(field), f"counterfactual.{field}", maximum=700)

    affected = _list(diagnosis.get("affected_sources"), "affected_sources", maximum=6)
    if status == "diagnosed" and not affected:
        raise DiagnosisValidationError("diagnosed result requires affected_sources")
    for index, item in enumerate(affected):
        source = dict(_object(item, f"affected_sources[{index}]"))
        _exact_fields(
            source,
            {"path", "start_line", "end_line", "symbol", "rationale"},
            f"affected_sources[{index}]",
        )
        _text(source.get("symbol"), f"affected_sources[{index}].symbol", maximum=160)
        _text(source.get("rationale"), f"affected_sources[{index}].rationale", maximum=600)
        _validate_source_range(
            source,
            bundle=bundle,
            observed_source_reads=observed_source_reads,
            field=f"affected_sources[{index}]",
        )

    uncertainties = _string_list(
        diagnosis.get("uncertainties"), "uncertainties", maximum=8
    )
    diagnosis["uncertainties"] = [
        _text(item, f"uncertainties[{index}]", maximum=500)
        for index, item in enumerate(uncertainties)
    ]
    if status == "insufficient_evidence" and not uncertainties:
        raise DiagnosisValidationError(
            "insufficient_evidence result requires at least one uncertainty"
        )
    return diagnosis
