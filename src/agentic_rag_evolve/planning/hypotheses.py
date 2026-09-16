"""Open-ended aggregation of grounded diagnoses into falsifiable hypotheses."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .hypothesis_validator import HypothesisValidationError, validate_hypotheses


class HypothesisModel(Protocol):
    model_name: str

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class HypothesisRunReport:
    cohort_id: str
    status: str
    model_calls: int
    validation_failures: int
    hypothesis_count: int
    output_file: str | None
    audit_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You aggregate grounded DeepRead diagnoses into falsifiable improvement hypotheses.

Do not use a predefined defect taxonomy or repair operator. Group diagnoses only when their evidence supports the same underlying agent mechanism and one behavior change could plausibly affect them. Similar answer topics, datasets, or affected filenames alone are not a shared mechanism. Preserve counterevidence and uncertainty. Do not write code, a patch, or a detailed implementation plan.

Return only one JSON object with exactly these top-level fields:
- schema_version="deepread-improvement-hypotheses-v1"
- cohort_id (copy from input)
- status="ready" or "no_hypotheses"
- hypotheses
- unclustered_task_ids

Each hypothesis has exactly:
- title
- task_ids
- common_mechanism
- earliest_intervention_pattern
- target_cohort: {inclusion_signals, exclusion_signals}
- behavior_delta
- supporting_evidence_refs
- contradicting_evidence_refs
- affected_source_refs
- validation: {expected_observation, falsifier, regression_guards}
- uncertainties

Evidence refs are {task_id, side, index, rationale}, where side is supporting or contradicting and index addresses that diagnosis's corresponding evidence list. Source refs are {task_id, index, rationale}, where index addresses affected_sources. Never copy or invent an evidence anchor or source path. Every clustered task needs at least one supporting ref. Each hypothesis needs counterevidence. Every eligible task must appear exactly once, either in one hypothesis or unclustered_task_ids. A singleton may be retained as an exploratory hypothesis when its mechanism is specific and falsifiable; do not claim recurrence for it."""


def _parse_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise HypothesisValidationError("model output contains no JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HypothesisValidationError("model output contains invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise HypothesisValidationError("model output must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _empty_result(cohort: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "deepread-improvement-hypotheses-v1",
        "cohort_id": cohort["cohort_id"],
        "status": "no_eligible_diagnoses",
        "hypotheses": [],
        "unclustered_task_ids": [],
    }


def run_hypothesis_aggregation(
    *,
    cohort_path: Path,
    output_path: Path,
    model: HypothesisModel,
    max_validation_failures: int = 1,
    max_output_tokens: int | None = None,
) -> HypothesisRunReport:
    if max_validation_failures < 0:
        raise ValueError("max_validation_failures must be non-negative")
    cohort = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
    if cohort.get("schema_version") != "deepread-hypothesis-cohort-v1":
        raise ValueError("unsupported hypothesis cohort schema")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite hypotheses: {output_path}")
    audit_path = output_path.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite hypothesis audit: {audit_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_id = str(cohort.get("cohort_id") or "")
    audit: dict[str, Any] = {
        "schema_version": "deepread-hypothesis-audit-v1",
        "cohort_id": cohort_id,
        "model": model.model_name,
        "events": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        "status": "running",
    }
    _write_json(audit_path, audit)

    if not cohort.get("eligible_diagnoses"):
        result = _empty_result(cohort)
        _write_json(output_path, result)
        audit["status"] = "no_eligible_diagnoses"
        _write_json(audit_path, audit)
        return HypothesisRunReport(
            cohort_id, "no_eligible_diagnoses", 0, 0, 0, str(output_path), str(audit_path)
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "cohort_id": cohort_id,
                    "eligible_diagnoses": cohort["eligible_diagnoses"],
                },
                ensure_ascii=False,
            ),
        },
    ]
    validation_failures = 0
    model_calls = 0
    while True:
        model_calls += 1
        event: dict[str, Any] = {
            "call": model_calls,
            "status": "pending",
            "message_count": len(messages),
            "request_bytes": len(
                json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
            ),
        }
        audit["events"].append(event)
        _write_json(audit_path, audit)
        payload: dict[str, Any] = {
            "model": model.model_name,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        started = time.monotonic()
        try:
            response = model.complete(payload)
        except Exception as exc:
            event.update(
                {
                    "status": "error",
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "provider_attempts": getattr(model, "last_attempts", 1),
                    "retry_delays_seconds": list(
                        getattr(model, "last_retry_delays", [])
                    ),
                }
            )
            audit["status"] = "error"
            audit["error"] = event["error"]
            _write_json(audit_path, audit)
            return HypothesisRunReport(
                cohort_id, "error", model_calls, validation_failures, 0, None, str(audit_path)
            )
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        audit["token_usage"]["input_tokens"] += input_tokens
        audit["token_usage"]["output_tokens"] += output_tokens
        audit["token_usage"]["reasoning_tokens"] += reasoning_tokens
        choices = response.get("choices") or []
        content = str(((choices[0].get("message") or {}) if choices else {}).get("content") or "")
        event.update(
            {
                "status": "ok",
                "latency_seconds": round(time.monotonic() - started, 3),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                },
                "provider_attempts": getattr(model, "last_attempts", 1),
                "retry_delays_seconds": list(
                    getattr(model, "last_retry_delays", [])
                ),
            }
        )
        parsed: Mapping[str, Any] | None = None
        try:
            parsed = _parse_json_object(content)
            result = validate_hypotheses(parsed, cohort=cohort)
        except HypothesisValidationError as exc:
            validation_failures += 1
            event["validation_error"] = str(exc)
            if parsed is not None:
                _write_json(output_path.with_suffix(".candidate.json"), parsed)
            if validation_failures > max_validation_failures:
                audit["status"] = "validation_error"
                audit["error"] = str(exc)
                _write_json(audit_path, audit)
                return HypothesisRunReport(
                    cohort_id,
                    "validation_error",
                    model_calls,
                    validation_failures,
                    0,
                    None,
                    str(audit_path),
                )
            _write_json(audit_path, audit)
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"The hypothesis JSON failed validation: {exc}. Return the complete "
                            "corrected JSON. Keep all evidence/source references as cohort indices; "
                            "do not invent paths or anchors."
                        ),
                    },
                ]
            )
            continue
        _write_json(output_path, result)
        audit["status"] = "ok"
        _write_json(audit_path, audit)
        return HypothesisRunReport(
            cohort_id,
            "ok",
            model_calls,
            validation_failures,
            len(result["hypotheses"]),
            str(output_path),
            str(audit_path),
        )
