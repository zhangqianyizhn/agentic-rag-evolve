"""Restricted tool-using agent for evidence-anchored DeepRead diagnosis."""

from __future__ import annotations

import json
import re
import time
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
            "description": "Read an allowlisted DeepRead source range; at most 240 lines are returned per call.",
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


ANCHOR_CONTRACT = """Evidence anchors are flat JSON objects with exactly one of these shapes:
{"kind":"trajectory","claim":"...","turn":1,"tool_call_id":"... or omit"}
{"kind":"coverage","claim":"...","evidence_index":0,"layer":"corpus|candidate|read|answer"}
{"kind":"evaluation","claim":"...","field":"judge.score"}
{"kind":"source","claim":"...","quote":"exact text inside the cited lines","path":"...","start_line":1,"end_line":20}
{"kind":"payload","claim":"...","quote":"exact text inside the cited characters","path":"...","offset_chars":0,"end_chars":100}
Never nest a trajectory, coverage, evaluation, source, payload, or judge object inside an anchor. Judge facts use kind="evaluation" and a field such as "judge.score"."""


SYSTEM_PROMPT = """You diagnose an evolvable DeepRead document-QA system from a validated bundle.

Use only facts in the bundle and content returned by the provided tools. Do not invent source code, tool results, line numbers, or a fixed defect category. Separate observable manifestation from root-cause hypothesis. Find the earliest behavior that could have changed the outcome, not merely the last wrong answer. Inspect the smallest relevant source range before returning status=diagnosed; normally one source file is enough, and indexing/retrieval code should not be inspected unless the evidence specifically implicates it. Include evidence that challenges your hypothesis and a falsifiable counterfactual. Stop exploring once the diagnosis is supported and return the JSON. Do not write a patch or choose a repair operator.

Return only one JSON object with exactly these fields:
schema_version="deepread-diagnosis-v1"; task_id; status (diagnosed, not_agent_failure, or insufficient_evidence); failure_manifestation; earliest_intervention ({turn, tool_call_id or null, rationale} or null); root_cause_hypothesis; supporting_evidence; contradicting_evidence; counterfactual ({change, expected_observation, falsifier}); affected_sources; uncertainties.

""" + ANCHOR_CONTRACT + """

affected_sources entries require path,start_line,end_line,symbol,rationale. Cite only source/payload ranges you actually read. If the evidence supports a dataset or evaluation mismatch rather than an evolvable DeepRead defect, use status="not_agent_failure" and an empty affected_sources list. Keep each claim focused and the complete result concise."""


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
        start_line = int(arguments.get("start_line", 1))
        requested_end = int(arguments.get("end_line", start_line + 239))
        result = reader.read_source(
            str(arguments.get("path") or ""),
            start_line=start_line,
            end_line=min(requested_end, start_line + 239),
        )
        result["has_more"] = result["end_line"] < result["total_lines"]
        if result["end_line"] < requested_end:
            result["requested_end_line"] = requested_end
        return result
    if name == "read_payload":
        return reader.read_payload(
            str(arguments.get("path") or ""),
            offset_chars=int(arguments.get("offset_chars", 0)),
            limit_chars=min(int(arguments.get("limit_chars", 12000)), 12000),
        )
    raise ValueError(f"unknown diagnosis tool: {name}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _candidate_shape(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep failed-output structure for debugging without retaining long text."""

    shape: dict[str, Any] = {
        "status": value.get("status"),
        "top_level_fields": sorted(str(key) for key in value),
    }
    for field in ("supporting_evidence", "contradicting_evidence"):
        items = value.get(field)
        if isinstance(items, list):
            shape[field] = [
                {
                    "kind": item.get("kind"),
                    "fields": sorted(str(key) for key in item),
                }
                for item in items
                if isinstance(item, Mapping)
            ]
    return shape


def run_diagnosis(
    *,
    bundle_path: Path,
    source_root: Path,
    output_path: Path,
    model: DiagnosisModel,
    max_rounds: int = 12,
    max_validation_failures: int = 1,
    max_tool_calls: int | None = None,
    max_output_tokens: int | None = None,
) -> DiagnosisRunReport:
    """Run one diagnosis with restricted artifact tools and grounded output validation."""

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if max_validation_failures < 0:
        raise ValueError("max_validation_failures must be non-negative")
    if max_tool_calls is not None and max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")
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
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        },
        "status": "running",
    }
    audit_path = output_path / "audit.json"
    _write_json(audit_path, audit)
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

    for round_number in range(1, max_rounds + 2):
        forced_finalization = round_number == max_rounds + 1
        if forced_finalization:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The investigation budget is exhausted. Do not call tools. Return the "
                        "complete diagnosis JSON now, using only evidence already read. If the "
                        "evidence is insufficient, return status=insufficient_evidence rather "
                        "than continuing investigation."
                    ),
                }
            )
        model_event: dict[str, Any] = {
            "kind": "model",
            "round": round_number,
            "status": "pending",
            "message_count": len(messages),
            "request_bytes": len(
                json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
        }
        audit["events"].append(model_event)
        _write_json(audit_path, audit)
        request_payload: dict[str, Any] = {
            "model": model.model_name,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }
        if not forced_finalization:
            request_payload["tools"] = DIAGNOSIS_TOOLS
            request_payload["tool_choice"] = "auto"
        if max_output_tokens is not None:
            request_payload["max_tokens"] = max_output_tokens
        request_started = time.monotonic()
        try:
            response = model.complete(request_payload)
        except KeyboardInterrupt:
            model_event["status"] = "interrupted"
            model_event["latency_seconds"] = round(
                time.monotonic() - request_started, 3
            )
            audit["status"] = "interrupted"
            _write_json(audit_path, audit)
            raise
        except Exception as exc:
            model_event["status"] = "error"
            model_event["latency_seconds"] = round(
                time.monotonic() - request_started, 3
            )
            model_event["error"] = f"{type(exc).__name__}: {exc}"
            model_event["provider_attempts"] = getattr(model, "last_attempts", 1)
            model_event["retry_delays_seconds"] = list(
                getattr(model, "last_retry_delays", [])
            )
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
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        audit["token_usage"]["input_tokens"] += input_tokens
        audit["token_usage"]["output_tokens"] += output_tokens
        audit["token_usage"]["reasoning_tokens"] += reasoning_tokens
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
        model_event.update(
            {
                "status": "ok",
                "latency_seconds": round(time.monotonic() - request_started, 3),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "requested_tools": [
                    str((call.get("function") or {}).get("name") or "")
                    for call in tool_calls
                ],
                "final_candidate": bool(content and not tool_calls),
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
        _write_json(audit_path, audit)

        if tool_calls:
            for call in tool_calls:
                tool_call_count += 1
                call_id = str(call["id"])
                name = ""
                arguments: dict[str, Any] = {}
                try:
                    name, arguments = _tool_arguments(call)
                    if max_tool_calls is not None and tool_call_count > max_tool_calls:
                        raise RuntimeError(
                            "diagnosis tool budget exhausted; return the final JSON now"
                        )
                    if name == "read_payload":
                        path = str(arguments.get("path") or "")
                        offset = int(arguments.get("offset_chars", 0))
                        limit = int(arguments.get("limit_chars", 12_000))
                        prior = [
                            item for item in payload_reads if item.get("path") == path
                        ]
                        total_chars = (
                            int(prior[0].get("total_chars") or offset + limit)
                            if prior else offset + limit
                        )
                        requested_end = min(total_chars, offset + limit)
                        if any(
                            int(item.get("offset_chars") or 0) <= offset
                            and int(item.get("end_chars") or 0) >= requested_end
                            for item in prior
                        ):
                            raise RuntimeError(
                                "payload range was already read; use the existing evidence and finalize"
                            )
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
                _write_json(audit_path, audit)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_content, ensure_ascii=False),
                    }
                )
            continue

        parsed: Mapping[str, Any] | None = None
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
            model_event["validation_error"] = str(exc)
            if parsed is not None:
                model_event["candidate_shape"] = _candidate_shape(parsed)
                _write_json(output_path / "candidate.json", parsed)
            _write_json(audit_path, audit)
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
                        "or use the tools, then return the complete JSON object again.\n\n"
                        f"{ANCHOR_CONTRACT}"
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
        max_rounds + 1,
        tool_call_count,
        validation_failures,
        None,
        audit_path.name,
    )
