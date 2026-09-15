"""Restricted tool-using agent for evidence-anchored DeepRead diagnosis."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from agentic_rag_evolve.diagnostics import DiagnosticArtifactReader

from .validator import DiagnosisValidationError, validate_diagnosis


class DiagnosisModel(Protocol):
    model_name: str

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DiagnosisRunReport:
    task_id: str
    status: str
    rounds: int
    tool_calls: int
    validation_failures: int
    diagnosis_file: str | None
    audit_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIAGNOSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "List source files visible for this diagnosis.",
            "parameters": {
                "type": "object",
                "properties": {"component": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "Read a bounded range from an allowlisted DeepRead source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_payload",
            "description": "Read a bounded character range from a trajectory payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset_chars": {"type": "integer", "minimum": 0},
                    "limit_chars": {"type": "integer", "minimum": 1, "maximum": 12000},
                },
                "required": ["path", "offset_chars", "limit_chars"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = """You diagnose an evolvable DeepRead document-QA system from a validated bundle.

Use only facts in the bundle and content returned by the provided tools. Do not invent source code, tool results, line numbers, or a fixed defect category. Separate observable manifestation from root-cause hypothesis. Find the earliest behavior that could have changed the outcome, not merely the last wrong answer. Inspect relevant source before returning status=diagnosed. Include evidence that challenges your hypothesis and a falsifiable counterfactual. Do not write a patch or choose a repair operator.

Return only one JSON object with exactly these fields:
schema_version="deepread-diagnosis-v1"; task_id; status (diagnosed, not_agent_failure, or insufficient_evidence); failure_manifestation; earliest_intervention ({turn, tool_call_id or null, rationale} or null); root_cause_hypothesis; supporting_evidence; contradicting_evidence; counterfactual ({change, expected_observation, falsifier}); affected_sources; uncertainties.

Evidence anchors have kind and claim, plus: trajectory {turn, optional tool_call_id}; coverage {evidence_index, layer}; evaluation {field}; source {path,start_line,end_line}; payload {path,offset_chars,end_chars}. affected_sources entries require path,start_line,end_line,symbol,rationale. Cite only source/payload ranges you actually read. Keep each claim focused and the complete result concise."""


def _diagnosis_input(bundle: Mapping[str, Any]) -> dict[str, Any]:
    access = bundle.get("access") or {}
    return {
        "task": bundle.get("task"),
        "run_context": bundle.get("run_context"),
        "evaluation": bundle.get("evaluation"),
        "evidence_coverage": bundle.get("evidence_coverage"),
        "failure_signals": bundle.get("failure_signals"),
        "trajectory": bundle.get("trajectory"),
        "source_catalog": [
            {
                key: item.get(key)
                for key in ("path", "component", "purpose", "line_count")
            }
            for item in access.get("sources") or []
        ],
        "payload_catalog": [
            {key: item.get(key) for key in ("path", "bytes", "uses")}
            for item in access.get("payloads") or []
        ],
    }


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
            raise DiagnosisValidationError("model output contains no JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise DiagnosisValidationError("model output contains invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DiagnosisValidationError("model output must be a JSON object")
    return value


def _route(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = bundle.get("failure_signals") or {}
    route = signals.get("diagnosis_route")
    if isinstance(route, Mapping):
        return route
    triage = str(signals.get("triage") or "unknown")
    eligible = triage in {
        "execution_failure",
        "no_answer",
        "incorrect_answer",
        "partial_answer",
    }
    return {
        "eligible": eligible,
        "target": "deepread" if eligible else "unknown",
        "reason": f"Legacy bundle route derived from triage={triage}.",
    }


def _tool_arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments") or "{}"
    if isinstance(raw, Mapping):
        arguments = dict(raw)
    else:
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("tool arguments must be a JSON object")
        arguments = dict(parsed)
    return name, arguments


def _execute_tool(
    reader: DiagnosticArtifactReader,
    name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if name == "list_sources":
        return {"sources": reader.list_sources(arguments.get("component"))}
    if name == "read_source":
        return reader.read_source(
            str(arguments.get("path") or ""),
            start_line=int(arguments.get("start_line", 1)),
            end_line=int(arguments.get("end_line", 240)),
        )
    if name == "read_payload":
        return reader.read_payload(
            str(arguments.get("path") or ""),
            offset_chars=int(arguments.get("offset_chars", 0)),
            limit_chars=int(arguments.get("limit_chars", 12000)),
        )
    raise ValueError(f"unknown diagnosis tool: {name}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_diagnosis(
    *,
    bundle_path: Path,
    source_root: Path,
    output_path: Path,
    model: DiagnosisModel,
    max_rounds: int = 12,
    max_validation_failures: int = 2,
) -> DiagnosisRunReport:
    """Run one diagnosis with restricted artifact tools and grounded output validation."""

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if max_validation_failures < 0:
        raise ValueError("max_validation_failures must be non-negative")
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"diagnosis output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    reader = DiagnosticArtifactReader(
        bundle_path=Path(bundle_path), source_root=Path(source_root)
    )
    bundle = reader.bundle
    task_id = str((bundle.get("task") or {}).get("task_id") or "")
    route = _route(bundle)
    audit: dict[str, Any] = {
        "schema_version": "deepread-diagnosis-audit-v1",
        "task_id": task_id,
        "model": model.model_name,
        "route": dict(route),
        "events": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
    }
    audit_path = output_path / "audit.json"
    if not route.get("eligible") or route.get("target") != "deepread":
        audit["status"] = "skipped"
        audit["reason"] = route.get("reason")
        _write_json(audit_path, audit)
        return DiagnosisRunReport(task_id, "skipped", 0, 0, 0, None, audit_path.name)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(_diagnosis_input(bundle), ensure_ascii=False),
        },
    ]
    source_reads: list[Mapping[str, Any]] = []
    payload_reads: list[Mapping[str, Any]] = []
    tool_call_count = 0
    validation_failures = 0

    for round_number in range(1, max_rounds + 1):
        try:
            response = model.complete(
                {
                    "model": model.model_name,
                    "messages": messages,
                    "tools": DIAGNOSIS_TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.0,
                    "stream": False,
                }
            )
        except Exception as exc:
            audit["status"] = "error"
            audit["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(audit_path, audit)
            return DiagnosisRunReport(
                task_id,
                "error",
                round_number,
                tool_call_count,
                validation_failures,
                None,
                audit_path.name,
            )
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        audit["token_usage"]["input_tokens"] += input_tokens
        audit["token_usage"]["output_tokens"] += output_tokens
        choices = response.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = str(message.get("content") or "")
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_tool_calls, start=1):
            call = dict(raw_call)
            call.setdefault("id", f"diagnosis_{round_number}_{index}")
            call.setdefault("type", "function")
            tool_calls.append(call)
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        audit["events"].append(
            {
                "kind": "model",
                "round": round_number,
                "requested_tools": [
                    str((call.get("function") or {}).get("name") or "")
                    for call in tool_calls
                ],
                "final_candidate": bool(content and not tool_calls),
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }
        )

        if tool_calls:
            for call in tool_calls:
                tool_call_count += 1
                call_id = str(call["id"])
                name = ""
                arguments: dict[str, Any] = {}
                try:
                    name, arguments = _tool_arguments(call)
                    result = _execute_tool(reader, name, arguments)
                    ok = True
                    if name == "read_source":
                        source_reads.append(result)
                    elif name == "read_payload":
                        payload_reads.append(result)
                    tool_content = result
                    error = None
                except Exception as exc:
                    ok = False
                    error = f"{type(exc).__name__}: {exc}"
                    tool_content = {"ok": False, "error": error}
                audit["events"].append(
                    {
                        "kind": "tool",
                        "round": round_number,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "ok": ok,
                        **({"error": error} if error else {}),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_content, ensure_ascii=False),
                    }
                )
            continue

        try:
            parsed = _parse_json_object(content)
            diagnosis = validate_diagnosis(
                parsed,
                bundle=bundle,
                observed_source_reads=source_reads,
                observed_payload_reads=payload_reads,
            )
        except DiagnosisValidationError as exc:
            validation_failures += 1
            audit["events"][-1]["validation_error"] = str(exc)
            if validation_failures > max_validation_failures:
                audit["status"] = "validation_error"
                audit["error"] = str(exc)
                _write_json(audit_path, audit)
                return DiagnosisRunReport(
                    task_id,
                    "validation_error",
                    round_number,
                    tool_call_count,
                    validation_failures,
                    None,
                    audit_path.name,
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your diagnosis failed validation: {exc}. Correct the cited facts "
                        "or use the tools, then return the complete JSON object again."
                    ),
                }
            )
            continue

        diagnosis_path = output_path / "diagnosis.json"
        _write_json(diagnosis_path, diagnosis)
        audit["status"] = "ok"
        _write_json(audit_path, audit)
        return DiagnosisRunReport(
            task_id,
            "ok",
            round_number,
            tool_call_count,
            validation_failures,
            diagnosis_path.name,
            audit_path.name,
        )

    audit["status"] = "max_rounds"
    _write_json(audit_path, audit)
    return DiagnosisRunReport(
        task_id,
        "max_rounds",
        max_rounds,
        tool_call_count,
        validation_failures,
        None,
        audit_path.name,
    )
