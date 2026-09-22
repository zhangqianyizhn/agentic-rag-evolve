"""Plan-scoped code modification agent for isolated DeepRead candidates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .audit import candidate_snapshot_sha256, collect_changed_paths


class ModificationModel(Protocol):
    model_name: str

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CandidateModificationReport:
    candidate_id: str
    status: str
    model_calls: int
    tool_calls: int
    changed_file_count: int
    output_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "List the only candidate files that may be read or edited.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "Read at most 240 lines from an allowed candidate source file.",
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
            "name": "replace_text",
            "description": (
                "Replace one exact, uniquely occurring text block in an inspected allowed file. "
                "Use a sufficiently specific old_text block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "minLength": 1},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_diff",
            "description": "Show the current bounded candidate diff for self-review.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_modification",
            "description": "Submit after reviewing a non-empty, scope-valid, syntax-valid diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "implemented_behavior_delta": {"type": "string"},
                    "preservation_notes": {"type": "array", "items": {"type": "string"}},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "summary",
                    "implemented_behavior_delta",
                    "preservation_notes",
                    "uncertainties",
                ],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = """You implement one frozen DeepRead modification plan in an isolated candidate.

The framework, not this prompt, enforces the edit boundary. You have no shell and no arbitrary filesystem access. Inspect a file before editing it. Make the smallest change that implements required_behavior_delta while protecting must_preserve and non_goals. Ingestion and Markdown indexing are valid targets when exposed by the frozen plan; preserve deterministic document identities and artifact compatibility unless the change contract explicitly requires otherwise. A plan marked requires_store_rebuild will be evaluated on newly built candidate artifacts, so do not add compatibility shortcuts that silently reuse an old store. Do not alter evaluation behavior, providers, datasets, tests, or tracing merely to improve measured scores. Use exact replace_text operations, inspect the final diff, then call submit_modification. If the plan cannot be implemented safely within the exposed files, make no speculative edits and explain the blocker in your final response; never invent another edit scope.
"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments") or "{}"
    value = dict(raw) if isinstance(raw, Mapping) else json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be an object")
    return name, value


class CandidateEditor:
    MAX_LINES = 240
    MAX_READ_CHARS = 30_000
    MAX_REPLACEMENT_CHARS = 120_000
    MAX_DIFF_CHARS = 40_000

    def __init__(self, *, candidate_path: Path, allowed_paths: list[str], max_files: int):
        self.root = Path(candidate_path).resolve()
        self.allowed_paths = tuple(sorted(set(allowed_paths)))
        self.max_files = max_files
        self.inspected: set[str] = set()
        self.diff_reviewed = False
        for relative in self.allowed_paths:
            self._path(relative)

    def _path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative not in self.allowed_paths:
            raise PermissionError(f"candidate source is outside allowed scope: {relative}")
        raw = self.root / candidate
        resolved = raw.resolve()
        if not resolved.is_relative_to(self.root) or raw.is_symlink():
            raise PermissionError(f"candidate source is not a regular in-scope path: {relative}")
        if not resolved.is_file():
            raise ValueError(f"candidate source is not a file: {relative}")
        return resolved

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "path": relative,
                "sha256": _sha(self._path(relative).read_bytes()),
                "bytes": self._path(relative).stat().st_size,
            }
            for relative in self.allowed_paths
        ]

    def read(self, relative: str, start_line: int, end_line: int) -> dict[str, Any]:
        if start_line < 1 or end_line < start_line or end_line - start_line + 1 > self.MAX_LINES:
            raise ValueError(f"read range must contain 1-{self.MAX_LINES} lines")
        path = self._path(relative)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if start_line > len(lines) + 1:
            raise ValueError("start_line exceeds file length")
        content = "\n".join(lines[start_line - 1 : end_line])
        if len(content) > self.MAX_READ_CHARS:
            raise ValueError("source read is too large")
        self.inspected.add(relative)
        return {
            "path": relative,
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "total_lines": len(lines),
            "content": content,
        }

    def replace(self, relative: str, old_text: str, new_text: str) -> dict[str, Any]:
        if relative not in self.inspected:
            raise PermissionError(f"inspect source before editing: {relative}")
        if not old_text or len(old_text) + len(new_text) > self.MAX_REPLACEMENT_CHARS:
            raise ValueError("replacement is empty or too large")
        path = self._path(relative)
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old_text)
        if occurrences != 1:
            raise ValueError(f"old_text must occur exactly once; found {occurrences}")
        updated = text.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        changed = collect_changed_paths(self.root)
        if len(changed) > self.max_files or any(item not in self.allowed_paths for item in changed):
            path.write_text(text, encoding="utf-8")
            raise PermissionError("replacement would exceed the frozen edit scope")
        self.diff_reviewed = False
        syntax = None
        if path.suffix == ".py":
            try:
                compile(updated, str(path), "exec")
                syntax = "ok"
            except SyntaxError as exc:
                syntax = f"SyntaxError: {exc}"
        return {
            "path": relative,
            "old_sha256": _sha(text.encode()),
            "new_sha256": _sha(updated.encode()),
            "changed_paths": changed,
            "python_syntax": syntax,
        }

    def diff(self) -> dict[str, Any]:
        result = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--no-ext-diff", "--unified=3", "HEAD", "--", *self.allowed_paths],
            check=True,
            capture_output=True,
            text=True,
        )
        content = result.stdout
        truncated = len(content) > self.MAX_DIFF_CHARS
        self.diff_reviewed = True
        return {
            "changed_paths": collect_changed_paths(self.root),
            "diff": content[: self.MAX_DIFF_CHARS],
            "truncated": truncated,
        }

    def validate_submission(self) -> list[str]:
        violations: list[str] = []
        changed = collect_changed_paths(self.root)
        if not changed:
            violations.append("no_candidate_changes")
        if len(changed) > self.max_files:
            violations.append("changed_files_exceed_budget")
        if any(item not in self.allowed_paths for item in changed):
            violations.append("changed_files_outside_allowed_paths")
        if not self.diff_reviewed:
            violations.append("final_diff_not_reviewed")
        for relative in changed:
            path = self._path(relative)
            if path.suffix == ".py":
                try:
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")
                except Exception:
                    violations.append(f"python_syntax_error:{relative}")
        check = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--check", "HEAD"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0 or check.stdout or check.stderr:
            violations.append("git_diff_check_failed")
        return violations


def run_candidate_modification(
    *,
    manifest_path: Path,
    plan_path: Path,
    output_path: Path,
    model: ModificationModel,
    max_rounds: int = 20,
    max_tool_calls: int = 30,
    max_output_tokens: int | None = None,
) -> CandidateModificationReport:
    if max_rounds < 1 or max_tool_calls < 1:
        raise ValueError("agent budgets must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite modification artifact: {output_path}")
    manifest_bytes = Path(manifest_path).read_bytes()
    plan_bytes = Path(plan_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    document = json.loads(plan_bytes)
    if manifest.get("schema_version") != "deepread-candidate-manifest-v1":
        raise ValueError("unsupported candidate manifest schema")
    if document.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")
    if _sha(plan_bytes) != manifest.get("plan_sha256"):
        raise ValueError("modification plan hash does not match candidate manifest")
    matches = [item for item in document.get("plans") or [] if item.get("plan_id") == manifest.get("plan_id")]
    if len(matches) != 1 or matches[0].get("decision") != "proceed":
        raise ValueError("candidate must reference one proceeding plan")
    plan = matches[0]
    scope = plan.get("edit_scope") or {}
    candidate_path = Path(str(manifest["candidate_path"])).resolve()
    head = subprocess.run(
        ["git", "-C", str(candidate_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != manifest.get("base_commit"):
        raise ValueError("candidate HEAD does not match manifest base commit")
    if collect_changed_paths(candidate_path):
        raise ValueError("candidate must be clean before modification")
    editor = CandidateEditor(
        candidate_path=candidate_path,
        allowed_paths=[str(item) for item in scope.get("allowed_paths") or []],
        max_files=int(scope.get("max_files_to_modify") or 0),
    )
    artifact: dict[str, Any] = {
        "schema_version": "deepread-candidate-modification-v1",
        "candidate_id": manifest.get("candidate_id"),
        "candidate_manifest_sha256": _sha(manifest_bytes),
        "plan_id": manifest.get("plan_id"),
        "plan_sha256": manifest.get("plan_sha256"),
        "base_commit": head,
        "candidate_path": str(candidate_path),
        "model": model.model_name,
        "status": "running",
        "events": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
    }
    _write_json(output_path, artifact)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidate_id": manifest.get("candidate_id"),
                    "plan_id": manifest.get("plan_id"),
                    "change_contract": plan.get("change_contract"),
                    "risk": plan.get("risk"),
                    "validation_plan": plan.get("validation_plan"),
                    "edit_scope": scope,
                    "source_catalog": editor.catalog(),
                },
                ensure_ascii=False,
            ),
        },
    ]
    tool_count = 0
    for round_number in range(1, max_rounds + 1):
        payload: dict[str, Any] = {
            "model": model.model_name,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.0,
            "stream": False,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        started = time.monotonic()
        try:
            response = model.complete(payload)
        except Exception as exc:
            artifact.update(status="error", error=f"{type(exc).__name__}: {exc}")
            _write_json(output_path, artifact)
            return CandidateModificationReport(str(manifest["candidate_id"]), "error", round_number, tool_count, len(collect_changed_paths(candidate_path)), str(output_path))
        choices = response.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = str(message.get("content") or "")
        calls = [dict(item) for item in message.get("tool_calls") or []]
        for index, call in enumerate(calls, 1):
            call.setdefault("id", f"modify_{round_number}_{index}")
            call.setdefault("type", "function")
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        usage = response.get("usage") or {}
        tokens = {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        }
        for key, value in tokens.items():
            artifact["token_usage"][key] += value
        artifact["events"].append(
            {
                "kind": "model",
                "round": round_number,
                "latency_seconds": round(time.monotonic() - started, 3),
                "requested_tools": [str((item.get("function") or {}).get("name") or "") for item in calls],
                "token_usage": tokens,
            }
        )
        if not calls:
            messages.append({"role": "user", "content": "Use the available tools. Finish only with submit_modification after reviewing the diff."})
            _write_json(output_path, artifact)
            continue
        for call in calls:
            tool_count += 1
            call_id = str(call["id"])
            name = ""
            arguments: dict[str, Any] = {}
            submitted = False
            try:
                if tool_count > max_tool_calls:
                    raise RuntimeError("modification tool budget exhausted")
                name, arguments = _arguments(call)
                if name == "list_sources":
                    result: Any = {"sources": editor.catalog()}
                elif name == "read_source":
                    result = editor.read(
                        str(arguments.get("path") or ""),
                        int(arguments.get("start_line") or 1),
                        int(arguments.get("end_line") or 240),
                    )
                elif name == "replace_text":
                    result = editor.replace(
                        str(arguments.get("path") or ""),
                        str(arguments.get("old_text") or ""),
                        str(arguments.get("new_text") or ""),
                    )
                elif name == "show_diff":
                    result = editor.diff()
                elif name == "submit_modification":
                    violations = editor.validate_submission()
                    if violations:
                        raise ValueError("submission rejected: " + ", ".join(violations))
                    result = {"ok": True}
                    submitted = True
                else:
                    raise ValueError(f"unknown modification tool: {name}")
                ok, error = True, None
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                ok, error = False, result["error"]
            event_arguments: dict[str, Any] = dict(arguments)
            if name == "replace_text":
                event_arguments = {
                    "path": str(arguments.get("path") or ""),
                    "old_text_chars": len(str(arguments.get("old_text") or "")),
                    "new_text_chars": len(str(arguments.get("new_text") or "")),
                    "old_text_sha256": _sha(str(arguments.get("old_text") or "").encode()),
                    "new_text_sha256": _sha(str(arguments.get("new_text") or "").encode()),
                }
            artifact["events"].append(
                {"kind": "tool", "round": round_number, "name": name, "arguments": event_arguments, "ok": ok, **({"error": error} if error else {})}
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(result, ensure_ascii=False)})
            if submitted:
                changed = collect_changed_paths(candidate_path)
                artifact.update(
                    status="modified",
                    summary=str(arguments.get("summary") or ""),
                    implemented_behavior_delta=str(arguments.get("implemented_behavior_delta") or ""),
                    preservation_notes=list(arguments.get("preservation_notes") or []),
                    uncertainties=list(arguments.get("uncertainties") or []),
                    changed_paths=changed,
                    candidate_snapshot_sha256=candidate_snapshot_sha256(candidate_path, head_commit=head, changed_paths=changed),
                )
                _write_json(output_path, artifact)
                return CandidateModificationReport(str(manifest["candidate_id"]), "modified", round_number, tool_count, len(changed), str(output_path))
        _write_json(output_path, artifact)
    artifact.update(status="budget_exhausted", changed_paths=collect_changed_paths(candidate_path))
    _write_json(output_path, artifact)
    return CandidateModificationReport(str(manifest["candidate_id"]), "budget_exhausted", max_rounds, tool_count, len(artifact["changed_paths"]), str(output_path))
