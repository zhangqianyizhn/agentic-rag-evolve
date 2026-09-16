"""Generate bounded modification plans without editing DeepRead."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .plan_validator import ModificationPlanValidationError, validate_modification_plan


class PlanModel(Protocol):
    model_name: str

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ModificationPlanReport:
    cohort_id: str
    status: str
    model_calls: int
    validation_failures: int
    plan_count: int
    output_file: str | None
    audit_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You convert grounded DeepRead improvement hypotheses into bounded modification plans.

Do not write code or patches. Do not name a predefined repair operator. For each hypothesis, decide proceed or defer. A singleton hypothesis must be deferred. A proceeding plan must use all hypothesis tasks as development cases and must define holdout selection rules, expected observations, rollback conditions, preserved behavior, non-goals, and regression scenarios.

You may select source scope only by copying {task_id,index,rationale} references from the hypothesis's affected_source_refs. Never output file paths; the validator resolves them. Do not target framework, runner, evaluator, provider, benchmark, tests, or documentation code.

Return one JSON object with exactly:
schema_version="deepread-modification-plan-v1"; cohort_id; hypothesis_set_status copied from the hypothesis input; plans.

Each plan has exactly:
- hypothesis_id
- decision: proceed or defer
- rationale
- allowed_source_refs: [{task_id,index,rationale}]
- change_contract: {current_behavior,required_behavior_delta,must_preserve,non_goals}
- validation_plan: {development_task_ids,holdout_selection_rules,expected_observations,rollback_conditions}
- risk: {level,regression_scenarios}

Return one plan for every hypothesis. Keep changes minimal and conditional on the target cohort. A deferred plan may have an empty allowed_source_refs/development_task_ids but must still state what evidence or recurrence is needed before proceeding."""


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ModificationPlanValidationError("model output contains no JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ModificationPlanValidationError("model output contains invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ModificationPlanValidationError("model output must be an object")
    return value


def run_modification_planning(
    *,
    cohort_path: Path,
    hypotheses_path: Path,
    output_path: Path,
    model: PlanModel,
    max_validation_failures: int = 1,
    max_output_tokens: int | None = None,
) -> ModificationPlanReport:
    if max_validation_failures < 0:
        raise ValueError("max_validation_failures must be non-negative")
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    cohort = json.loads(Path(cohort_path).read_text())
    hypotheses = json.loads(Path(hypotheses_path).read_text())
    if cohort.get("schema_version") != "deepread-hypothesis-cohort-v1":
        raise ValueError("unsupported hypothesis cohort schema")
    if hypotheses.get("schema_version") != "deepread-improvement-hypotheses-v1":
        raise ValueError("unsupported improvement hypothesis schema")
    if cohort.get("cohort_id") != hypotheses.get("cohort_id"):
        raise ValueError("cohort and hypothesis IDs do not match")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite modification plan: {output_path}")
    audit_path = output_path.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite modification plan audit: {audit_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_id = str(cohort["cohort_id"])
    audit: dict[str, Any] = {
        "schema_version": "deepread-modification-plan-audit-v1",
        "cohort_id": cohort_id,
        "model": model.model_name,
        "events": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        "status": "running",
    }
    _write_json(audit_path, audit)

    if not hypotheses.get("hypotheses"):
        result = {
            "schema_version": "deepread-modification-plan-v1",
            "cohort_id": cohort_id,
            "hypothesis_set_status": hypotheses.get("status"),
            "plans": [],
        }
        _write_json(output_path, result)
        audit["status"] = "no_plannable_hypotheses"
        _write_json(audit_path, audit)
        return ModificationPlanReport(
            cohort_id,
            "no_plannable_hypotheses",
            0,
            0,
            0,
            str(output_path),
            str(audit_path),
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "cohort_id": cohort_id,
                    "hypotheses": hypotheses["hypotheses"],
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
            return ModificationPlanReport(
                cohort_id,
                "error",
                model_calls,
                validation_failures,
                0,
                None,
                str(audit_path),
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
            parsed = _parse(content)
            result = validate_modification_plan(
                parsed, cohort=cohort, hypotheses=hypotheses
            )
        except ModificationPlanValidationError as exc:
            validation_failures += 1
            event["validation_error"] = str(exc)
            if parsed is not None:
                _write_json(output_path.with_suffix(".candidate.json"), parsed)
            if validation_failures > max_validation_failures:
                audit["status"] = "validation_error"
                audit["error"] = str(exc)
                _write_json(audit_path, audit)
                return ModificationPlanReport(
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
                            f"The modification plan failed validation: {exc}. Return the complete "
                            "corrected JSON. Use only affected_source_refs from each hypothesis."
                        ),
                    },
                ]
            )
            continue
        _write_json(output_path, result)
        audit["status"] = "ok"
        _write_json(audit_path, audit)
        return ModificationPlanReport(
            cohort_id,
            "ok",
            model_calls,
            validation_failures,
            len(result["plans"]),
            str(output_path),
            str(audit_path),
        )
