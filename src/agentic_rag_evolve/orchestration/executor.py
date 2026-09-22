"""Run configured iteration steps and checkpoint each result in the ledger."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .ledger import append_iteration_event, read_iteration_status


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class IterationExecutionReport:
    iteration_id: str
    status: str
    steps_completed: int
    next_step: str | None
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, **kwargs)


def _load_runbook(path: Path) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict) or value.get("schema_version") != "deepread-iteration-runbook-v1":
        raise ValueError("unsupported iteration runbook schema")
    steps = value.get("steps")
    if not isinstance(steps, Mapping) or not steps:
        raise ValueError("iteration runbook has no steps")
    return value, hashlib.sha256(data).hexdigest()


def _resolve_artifact(base: Path, specification: Any) -> Path:
    if isinstance(specification, str):
        path = (base / specification).resolve() if not Path(specification).is_absolute() else Path(specification).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"iteration artifact is missing: {path}")
        return path
    if not isinstance(specification, Mapping) or set(specification) != {"glob"}:
        raise ValueError("artifact specification must be a path or one exact glob")
    pattern = str(specification["glob"])
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("artifact glob must be relative and cannot traverse parents")
    matches = sorted(path.resolve() for path in base.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"artifact glob must match exactly one file: {pattern} ({len(matches)} found)")
    return matches[0]


def execute_iteration(
    *,
    ledger_root: Path,
    runbook_path: Path,
    max_steps: int | None = None,
    command_runner: CommandRunner = _run,
) -> IterationExecutionReport:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    ledger_root = Path(ledger_root).resolve()
    runbook_path = Path(runbook_path).resolve()
    runbook, runbook_sha = _load_runbook(runbook_path)
    status = read_iteration_status(root=ledger_root)
    if runbook.get("iteration_id") != status["iteration_id"]:
        raise ValueError("runbook iteration_id does not match ledger")
    base = Path(str(runbook.get("workspace") or runbook_path.parent)).resolve()
    if not base.is_dir():
        raise ValueError("runbook workspace is not a directory")
    completed = 0
    while not status["terminal"] and (max_steps is None or completed < max_steps):
        step = str(status["next_step"])
        specification = (runbook.get("steps") or {}).get(step)
        if not isinstance(specification, Mapping):
            return IterationExecutionReport(status["iteration_id"], "awaiting_configuration", completed, step, False)
        command = specification.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise ValueError(f"runbook command for {step} must be a non-empty argv list")
        raw_cwd = specification.get("cwd")
        command_cwd = base if raw_cwd is None else Path(str(raw_cwd)).resolve()
        if not command_cwd.is_dir():
            raise ValueError(f"runbook cwd for {step} is not a directory: {command_cwd}")
        attempt = status["event_count"] + 1
        command_error = None
        try:
            result = command_runner(command, cwd=command_cwd)
        except Exception as exc:
            command_error = f"{type(exc).__name__}: {exc}"
            result = subprocess.CompletedProcess(command, 1, stdout="", stderr=command_error)
        log_dir = ledger_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{attempt:04d}-{step}.json"
        log_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-iteration-command-log-v1",
                    "step": step,
                    "command": command,
                    "cwd": str(command_cwd),
                    "returncode": result.returncode,
                    "stdout_tail": (result.stdout or "")[-20_000:],
                    "stderr_tail": (result.stderr or "")[-20_000:],
                    "runbook_sha256": runbook_sha,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if result.returncode != 0:
            append_iteration_event(
                root=ledger_root,
                step=step,
                status="failed",
                artifacts={"command_log": log_path, "runbook": runbook_path},
                error=command_error or f"command exited with status {result.returncode}",
            )
            return IterationExecutionReport(status["iteration_id"], "step_failed", completed, step, False)
        raw_artifacts = specification.get("artifacts")
        if not isinstance(raw_artifacts, Mapping) or "primary" not in raw_artifacts:
            raise ValueError(f"runbook step {step} must declare a primary artifact")
        try:
            artifacts = {
                str(name): _resolve_artifact(base, value)
                for name, value in raw_artifacts.items()
            }
            artifacts.update(command_log=log_path, runbook=runbook_path)
            append_iteration_event(
                root=ledger_root,
                step=step,
                status="completed",
                artifacts=artifacts,
            )
        except Exception as exc:
            append_iteration_event(
                root=ledger_root,
                step=step,
                status="failed",
                artifacts={"command_log": log_path, "runbook": runbook_path},
                error=f"artifact validation failed: {type(exc).__name__}: {exc}",
            )
            return IterationExecutionReport(status["iteration_id"], "step_failed", completed, step, False)
        completed += 1
        status = read_iteration_status(root=ledger_root)
    final_status = "terminal" if status["terminal"] else "step_limit_reached"
    return IterationExecutionReport(
        status["iteration_id"], final_status, completed, status["next_step"], status["terminal"]
    )
