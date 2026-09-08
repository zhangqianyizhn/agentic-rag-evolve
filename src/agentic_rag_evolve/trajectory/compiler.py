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
    "llm_call_error": "model.error",
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

DEFAULT_INLINE_RESULT_BYTES = 16_000

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


def _result_summary(result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(result).__name__}
    if isinstance(result, dict):
        summary["keys"] = sorted(str(key) for key in result)
        for key in ("results", "nodes", "sections"):
            value = result.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
    elif isinstance(result, list):
        summary["item_count"] = len(result)
    elif isinstance(result, str):
        summary["character_count"] = len(result)
    return summary


def _externalize_large_results(
    trajectory: dict[str, Any],
    *,
    output_path: Path,
    inline_result_bytes: int,
) -> None:
    task_name = _safe_name(str(trajectory["task_id"]))
    for turn_index, turn in enumerate(trajectory["turns"], start=1):
        for tool_index, tool in enumerate(turn.get("tools", ()), start=1):
            result = tool.get("result")
            if result is None:
                continue
            encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) <= inline_result_bytes:
                continue
            digest = hashlib.sha256(encoded).hexdigest()
            relative_path = Path("payloads") / task_name / (
                f"turn_{turn_index:03d}_tool_{tool_index:03d}_{digest[:12]}.json"
            )
            payload_path = output_path / relative_path
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(encoded)
            tool["result_summary"] = _result_summary(result)
            tool["result_ref"] = {
                "path": relative_path.as_posix(),
                "sha256": digest,
                "bytes": len(encoded),
            }
            del tool["result"]


def _finish_turn(turn: dict[str, Any]) -> dict[str, Any]:
    raw_ids = turn.pop("raw_event_ids")
    turn["raw_event_range"] = {
        "first": raw_ids[0],
        "last": raw_ids[-1],
        "count": len(raw_ids),
    }
    model = turn["model"]
    if model.get("provider_attempts", 0) <= 1:
        model.pop("provider_attempts", None)
    requested = model.get("requested_tools")
    actual = [tool["name"] for tool in turn["tools"]]
    if requested == actual:
        model.pop("requested_tools", None)
    if not turn["tools"]:
        turn.pop("tools")
    return turn


def _compile_task(prediction: Mapping[str, Any], raw_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_id = str(prediction["task_id"])
    warnings: list[str] = []
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None
    pending_tools: dict[str, dict[str, Any]] = {}
    latest_tool: dict[str, Any] | None = None
    tool_counts: Counter[str] = Counter()
    retrieved_doc_ids: set[str] = set()
    read_doc_ids: set[str] = set()
    retrieval_hit_count = 0
    rounds: set[int] = set()
    raw_event_ids: set[str] = set()
    last_timestamp: datetime | None = None
    terminal: str | None = None

    for sequence, raw in enumerate(raw_events, start=1):
        raw_name = str(raw.get("event") or "unknown")
        if raw_name not in EVENT_KINDS:
            warnings.append(f"unknown_event:{raw_name}")
        raw_event_id = str(raw.get("event_id") or f"legacy_{sequence:06d}")
        if raw_event_id in raw_event_ids:
            warnings.append(f"duplicate_raw_event_id:{raw_event_id}")
        raw_event_ids.add(raw_event_id)
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

        if raw_name == "llm_request":
            if current_turn is not None:
                turns.append(_finish_turn(current_turn))
            current_turn = {
                "round": round_number,
                "model": {},
                "tools": [],
                "raw_event_ids": [raw_event_id],
            }
            pending_tools = {}
            latest_tool = None
        elif raw_name in {"llm_http_attempt", "llm_http_success", "llm_http_error", "llm_call_error", "llm_token_debug", "llm_response"}:
            if current_turn is None:
                warnings.append(f"orphan_event:{raw_event_id}")
                continue
            current_turn["raw_event_ids"].append(raw_event_id)
            model = current_turn["model"]
            if raw_name == "llm_http_attempt":
                model["provider_attempts"] = int(model.get("provider_attempts", 0)) + 1
                if raw.get("model"):
                    model["name"] = raw["model"]
            elif raw_name in {"llm_http_error", "llm_call_error"}:
                model.setdefault("provider_errors", []).append(str(raw.get("error") or "unknown error"))
            elif raw_name == "llm_token_debug":
                model["token_estimate"] = {
                    "input": raw.get("input_tokens"),
                    "output": raw.get("output_tokens"),
                }
            elif raw_name == "llm_response":
                if raw.get("reasoning_content"):
                    model["reasoning"] = raw["reasoning_content"]
                if raw.get("content"):
                    model["content"] = raw["content"]
                tool_calls_preview = raw.get("tool_calls") or []
                if tool_calls_preview:
                    model["requested_tools"] = [
                        item.get("name") for item in tool_calls_preview if isinstance(item, dict)
                    ]
        elif raw_name == "tool_call":
            if current_turn is None:
                warnings.append(f"orphan_event:{raw_event_id}")
                continue
            current_turn["raw_event_ids"].append(raw_event_id)
            call_id = str(raw.get("tool_call_id") or "")
            tool_name = str(raw.get("tool") or "unknown")
            interaction = {
                "call_id": call_id or None,
                "name": tool_name,
                "arguments": raw.get("args") or {},
                "ok": None,
                "result": None,
                "call_event_id": raw_event_id,
            }
            current_turn["tools"].append(interaction)
            latest_tool = interaction
            if call_id:
                pending_tools[call_id] = interaction
            tool_counts[tool_name] += 1
            args = raw.get("args") or {}
            if raw.get("tool") == "read_section" and isinstance(args, dict) and args.get("doc_id") is not None:
                read_doc_ids.add(str(args["doc_id"]))
        elif raw_name == "tool_result":
            explicit_call_id = str(raw.get("tool_call_id") or "")
            interaction = pending_tools.get(explicit_call_id) if explicit_call_id else latest_tool
            if interaction is None:
                warnings.append(f"orphan_event:{raw_event_id}")
                continue
            if current_turn is not None:
                current_turn["raw_event_ids"].append(raw_event_id)
            interaction["result_event_id"] = raw_event_id
            interaction["ok"] = bool(raw.get("ok", True))
            interaction["result"] = raw.get("result")
            if raw.get("error"):
                interaction["error"] = raw["error"]
            if not explicit_call_id:
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
            if current_turn is not None:
                current_turn["raw_event_ids"].append(raw_event_id)
                if current_turn["model"].get("content") == prediction.get("answer"):
                    current_turn["model"].pop("content", None)
                    current_turn["model"]["content_ref"] = "$.answer"
            terminal = "answer.final"
        elif raw_name == "max_rounds_reached":
            if current_turn is not None:
                current_turn["raw_event_ids"].append(raw_event_id)
            terminal = "run.max_rounds"
        elif raw_name in {"tool_args_parse_error", "tool_calls_recovered_from_text"}:
            if current_turn is None:
                warnings.append(f"orphan_event:{raw_event_id}")
                continue
            current_turn["raw_event_ids"].append(raw_event_id)
            current_turn.setdefault("annotations", []).append({
                "kind": EVENT_KINDS[raw_name],
                "error": raw.get("error"),
                "recovered_kind": raw.get("recovered_kind"),
                "raw_event_id": raw_event_id,
            })
        elif raw_name in {"llm_empty_message", "llm_thinking_only"}:
            if current_turn is None:
                warnings.append(f"orphan_event:{raw_event_id}")
                continue
            current_turn["raw_event_ids"].append(raw_event_id)
            current_turn.setdefault("annotations", []).append({
                "kind": EVENT_KINDS[raw_name],
                "raw_event_id": raw_event_id,
            })
        elif raw_name not in EVENT_KINDS:
            if current_turn is not None:
                current_turn.setdefault("other_events", []).append({
                    "name": raw_name,
                    "raw_event_id": raw_event_id,
                })

    if current_turn is not None:
        turns.append(_finish_turn(current_turn))
    if terminal is None and prediction.get("status") == "ok":
        warnings.append("missing_terminal_event")
    run_ids = {str(event.get("run_id")) for event in raw_events if event.get("run_id")}
    prediction_run_id = prediction.get("run_id")
    if prediction_run_id:
        run_ids.add(str(prediction_run_id))
    if len(run_ids) > 1:
        warnings.append("multiple_run_ids")

    return {
        "schema_version": "deepread-trajectory-v2",
        "run_id": next(iter(run_ids), None),
        "task_id": task_id,
        "question": prediction.get("question"),
        "status": prediction.get("status"),
        "answer": prediction.get("answer"),
        "token_usage": prediction.get("token_usage") or {},
        "turns": turns,
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
    inline_result_bytes: int = DEFAULT_INLINE_RESULT_BYTES,
) -> CompilationReport:
    if inline_result_bytes < 0:
        raise ValueError("inline_result_bytes must not be negative")
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
        _externalize_large_results(
            trajectory,
            output_path=output_path,
            inline_result_bytes=inline_result_bytes,
        )
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
