"""Loss-aware compact view over append-only DeepRead JSONL traces."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


EVENT_KINDS = {
    "llm_request": "model.request",
    "llm_response": "model.response",
    "llm_http_attempt": "provider.attempt",
    "llm_http_success": "provider.success",
    "llm_http_error": "provider.error",
    "llm_token_debug": "provider.token_estimate",
    "tool_call": "tool.call",
    "tool_result": "tool.result",
    "tool_args_parse_error": "tool.arguments_error",
    "tool_calls_recovered_from_text": "tool.calls_recovered",
    "final_answer": "answer.final",
    "max_rounds_reached": "run.max_rounds",
    "llm_empty_message": "model.empty",
    "llm_thinking_only": "model.thinking_only",
}

_METADATA_FIELDS = {
    "schema_version",
    "event_id",
    "ts",
    "event",
    "run_id",
    "task_id",
    "query_id",
    "round",
}
_OMITTED_FIELDS = {"context_delta_preview", "base_url", "url"}


@dataclass(frozen=True, slots=True)
class CompilationReport:
    task_count: int
    event_count: int
    unassigned_event_count: int
    warning_count: int
    output_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_files"] = list(self.output_files)
        return data


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"trace record at {path}:{line_number} is not an object")
            records.append(value)
    return tuple(records)


def _question_query_id(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def _safe_name(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:160]
    return safe or hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:16]


def _compact_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in raw.items()
        if key not in _METADATA_FIELDS and key not in _OMITTED_FIELDS
    }
    omitted = sorted(key for key in _OMITTED_FIELDS if key in raw)
    if omitted:
        payload["omitted_fields"] = omitted
    return payload


def _compile_task(prediction: Mapping[str, Any], raw_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_id = str(prediction["task_id"])
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    request_by_round: dict[int, str] = {}
    tool_calls: dict[str, str] = {}
    latest_request: str | None = None
    latest_response: str | None = None
    latest_tool_call: str | None = None
    tool_counts: Counter[str] = Counter()
    retrieved_doc_ids: set[str] = set()
    read_doc_ids: set[str] = set()
    retrieval_hit_count = 0
    rounds: set[int] = set()
    raw_event_ids: set[str] = set()
    last_timestamp: datetime | None = None

    for sequence, raw in enumerate(raw_events, start=1):
        raw_name = str(raw.get("event") or "unknown")
        kind = EVENT_KINDS.get(raw_name, f"raw.{raw_name}")
        if raw_name not in EVENT_KINDS:
            warnings.append(f"unknown_event:{raw_name}")
        raw_event_id = str(raw.get("event_id") or f"legacy_{sequence:06d}")
        if raw_event_id in raw_event_ids:
            warnings.append(f"duplicate_raw_event_id:{raw_event_id}")
        raw_event_ids.add(raw_event_id)
        step_id = f"step_{sequence:04d}"
        timestamp = raw.get("ts")
        if timestamp:
            try:
                parsed_timestamp = datetime.fromisoformat(str(timestamp))
                if last_timestamp is not None and parsed_timestamp < last_timestamp:
                    warnings.append("non_monotonic_timestamp")
                last_timestamp = parsed_timestamp
            except ValueError:
                warnings.append(f"invalid_timestamp:{raw_event_id}")
        round_value = raw.get("round")
        round_number = int(round_value) if round_value is not None else None
        if round_number is not None:
            rounds.add(round_number)

        parent_id: str | None = None
        if raw_name == "llm_request":
            latest_request = step_id
            if round_number is not None:
                request_by_round[round_number] = step_id
        elif raw_name in {"llm_response", "llm_http_attempt", "llm_http_success", "llm_http_error", "llm_token_debug"}:
            parent_id = request_by_round.get(round_number) if round_number is not None else latest_request
            if raw_name == "llm_response":
                latest_response = step_id
        elif raw_name == "tool_call":
            parent_id = latest_response
            latest_tool_call = step_id
            call_id = str(raw.get("tool_call_id") or "")
            if call_id:
                tool_calls[call_id] = step_id
            tool_counts[str(raw.get("tool") or "unknown")] += 1
            args = raw.get("args") or {}
            if raw.get("tool") == "read_section" and isinstance(args, dict) and args.get("doc_id") is not None:
                read_doc_ids.add(str(args["doc_id"]))
        elif raw_name == "tool_result":
            explicit_call_id = str(raw.get("tool_call_id") or "")
            parent_id = tool_calls.get(explicit_call_id) if explicit_call_id else latest_tool_call
            if not explicit_call_id and parent_id is not None:
                warnings.append(f"legacy_inferred_tool_parent:{raw_event_id}")
            result = raw.get("result") or {}
            if isinstance(result, dict):
                hits = result.get("results") or []
                if isinstance(hits, list):
                    retrieval_hit_count += len(hits)
                    for hit in hits:
                        ref = hit.get("ref") if isinstance(hit, dict) else None
                        if isinstance(ref, dict) and ref.get("doc_id") is not None:
                            retrieved_doc_ids.add(str(ref["doc_id"]))
        elif raw_name == "final_answer":
            parent_id = latest_response

        if raw_name in {"llm_response", "tool_call", "tool_result", "final_answer"} and parent_id is None:
            warnings.append(f"orphan_event:{raw_event_id}")

        events.append({
            "id": step_id,
            "sequence": sequence,
            "kind": kind,
            "timestamp": timestamp,
            "round": round_number,
            "parent_id": parent_id,
            "raw_event_id": raw_event_id,
            "payload": _compact_payload(raw),
        })

    terminal = next(
        (event["kind"] for event in reversed(events) if event["kind"] in {"answer.final", "run.max_rounds"}),
        None,
    )
    if terminal is None and prediction.get("status") == "ok":
        warnings.append("missing_terminal_event")
    run_ids = {str(event.get("run_id")) for event in raw_events if event.get("run_id")}
    prediction_run_id = prediction.get("run_id")
    if prediction_run_id:
        run_ids.add(str(prediction_run_id))
    if len(run_ids) > 1:
        warnings.append("multiple_run_ids")

    return {
        "schema_version": "deepread-trajectory-v1",
        "run_id": next(iter(run_ids), None),
        "task_id": task_id,
        "query_id": _question_query_id(str(prediction.get("question") or "")),
        "question": prediction.get("question"),
        "status": prediction.get("status"),
        "answer": prediction.get("answer"),
        "token_usage": prediction.get("token_usage") or {},
        "events": events,
        "summary": {
            "round_count": len(rounds),
            "tool_call_count": sum(tool_counts.values()),
            "tool_counts": dict(sorted(tool_counts.items())),
            "retrieval_hit_count": retrieval_hit_count,
            "retrieved_doc_ids": sorted(retrieved_doc_ids),
            "read_doc_ids": sorted(read_doc_ids),
            "terminal_event": terminal,
            "warnings": sorted(set(warnings)),
        },
    }


def compile_trajectories(
    *,
    trace_path: Path,
    prediction_path: Path,
    output_path: Path,
) -> CompilationReport:
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"trajectory output directory must be empty: {output_path}")
    predictions = _load_jsonl(prediction_path)
    raw_events = _load_jsonl(trace_path)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    query_to_tasks: dict[str, list[str]] = defaultdict(list)
    known_task_ids: set[str] = set()
    for prediction in predictions:
        task_id = str(prediction.get("task_id") or "")
        if not task_id:
            raise ValueError("prediction has no task_id")
        if task_id in known_task_ids:
            raise ValueError(f"duplicate prediction task_id: {task_id}")
        known_task_ids.add(task_id)
        query_to_tasks[_question_query_id(str(prediction.get("question") or ""))].append(task_id)

    unassigned = 0
    for raw in raw_events:
        task_id = raw.get("task_id")
        if task_id is None:
            candidates = query_to_tasks.get(str(raw.get("query_id") or ""), [])
            if len(candidates) == 1:
                task_id = candidates[0]
        if task_id is None or str(task_id) not in known_task_ids:
            unassigned += 1
            continue
        by_task[str(task_id)].append(raw)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    warning_count = 0
    for prediction in predictions:
        task_id = str(prediction["task_id"])
        trajectory = _compile_task(prediction, by_task.get(task_id, ()))
        warning_count += len(trajectory["summary"]["warnings"])
        filename = f"{_safe_name(task_id)}.trajectory.json"
        (output_path / filename).write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_files.append(filename)

    report = CompilationReport(
        task_count=len(predictions),
        event_count=len(raw_events),
        unassigned_event_count=unassigned,
        warning_count=warning_count,
        output_files=tuple(output_files),
    )
    (output_path / "compilation_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
