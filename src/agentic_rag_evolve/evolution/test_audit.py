"""Run a fixed, non-shell test policy against an isolated candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .audit import candidate_snapshot_sha256, collect_changed_paths


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
_RAN_TESTS = re.compile(r"Ran\s+(\d+)\s+tests?\s+in\s+")


def _git(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=True, capture_output=True, text=True)


def _process(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), **kwargs)


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _relative_directory(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must stay inside the candidate")
    return value


def _bounded(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _safe_environment(candidate_path: Path, pycache_path: Path) -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_path),
        "PYTHONPATH": os.pathsep.join(
            [str(candidate_path / "src"), str(candidate_path)]
        ),
    }


def _state(
    candidate_path: Path, command_runner: CommandRunner
) -> tuple[str, list[str], str]:
    head = command_runner(
        ["git", "-C", str(candidate_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    changed_paths = collect_changed_paths(candidate_path, command_runner)
    snapshot = candidate_snapshot_sha256(
        candidate_path, head_commit=head, changed_paths=changed_paths
    )
    return head, changed_paths, snapshot


def audit_candidate_tests(
    *,
    candidate_audit_path: Path,
    policy_path: Path,
    python_executable: Path,
    command_runner: CommandRunner = _git,
    process_runner: ProcessRunner = _process,
) -> dict[str, Any]:
    """Execute trusted test kinds and preserve a compact, reproducible audit."""

    candidate_audit_path = Path(candidate_audit_path)
    audit = _object(candidate_audit_path, "candidate audit")
    policy_bytes = Path(policy_path).read_bytes()
    policy = json.loads(policy_bytes)
    if not isinstance(policy, dict):
        raise ValueError("candidate test policy must be a JSON object")
    if audit.get("schema_version") != "deepread-candidate-audit-v1":
        raise ValueError("unsupported candidate audit schema")
    if not audit.get("passed"):
        raise ValueError("candidate static audit must pass before tests")
    if policy.get("schema_version") != "deepread-candidate-test-policy-v1":
        raise ValueError("unsupported candidate test policy schema")
    _exact(policy, {"schema_version", "policy_id", "checks"}, "candidate test policy")
    policy_id = str(policy.get("policy_id") or "").strip()
    checks = policy.get("checks")
    if not policy_id or not isinstance(checks, list) or not checks:
        raise ValueError("candidate test policy requires policy_id and checks")
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if policy_id != audit.get("test_policy_id"):
        raise ValueError("candidate test policy_id does not match static audit")
    if policy_sha256 != audit.get("test_policy_sha256"):
        raise ValueError("candidate test policy hash does not match static audit")
    if any(not isinstance(item, Mapping) for item in checks):
        raise ValueError("candidate test check must be an object")
    check_ids = [str(item.get("check_id") or "") for item in checks]
    if not all(check_ids) or len(check_ids) != len(set(check_ids)):
        raise ValueError("candidate test check_ids must be non-empty and unique")

    candidate_path = Path(str(audit.get("candidate_path") or "")).resolve()
    if not candidate_path.is_dir():
        raise ValueError(f"candidate path is not a directory: {candidate_path}")
    head_before, paths_before, snapshot_before = _state(candidate_path, command_runner)
    if head_before != audit.get("head_commit"):
        raise ValueError("candidate HEAD changed after static audit")
    if paths_before != audit.get("changed_paths"):
        raise ValueError("candidate changed paths differ from static audit")
    if snapshot_before != audit.get("candidate_snapshot_sha256"):
        raise ValueError("candidate source snapshot changed after static audit")

    results = []
    with tempfile.TemporaryDirectory(prefix="deepread-candidate-tests-") as directory:
        environment = _safe_environment(candidate_path, Path(directory) / "pycache")
        for check in checks:
            _exact(
                check,
                {
                    "check_id", "kind", "start_directory", "pattern",
                    "timeout_seconds", "minimum_tests",
                },
                "candidate test check",
            )
            check_id = str(check["check_id"])
            if check.get("kind") != "unittest_discover":
                raise ValueError(f"unsupported candidate test kind: {check.get('kind')!r}")
            start_directory = _relative_directory(
                check.get("start_directory"), f"{check_id}.start_directory"
            )
            resolved_start = (candidate_path / start_directory).resolve()
            if candidate_path not in resolved_start.parents or not resolved_start.is_dir():
                raise ValueError(f"{check_id}.start_directory is not a candidate directory")
            pattern = str(check.get("pattern") or "")
            if not pattern or "/" in pattern or "\\" in pattern:
                raise ValueError(f"{check_id}.pattern must be a filename pattern")
            timeout = int(check.get("timeout_seconds") or 0)
            minimum_tests = int(check.get("minimum_tests") or 0)
            if timeout < 1 or timeout > 3600 or minimum_tests < 1:
                raise ValueError(f"{check_id} has invalid timeout or minimum_tests")
            command = [
                # Resolving a venv's python symlink selects the base interpreter
                # and silently loses that venv's installed dependencies.
                str(Path(python_executable).absolute()), "-m", "unittest", "discover",
                "-s", start_directory, "-p", pattern, "-v",
            ]
            started = time.monotonic()
            timed_out = False
            try:
                completed = process_runner(
                    command,
                    cwd=candidate_path,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                returncode = None
            except OSError as exc:
                stdout = ""
                stderr = f"{type(exc).__name__}: {exc}"
                returncode = None
            elapsed = time.monotonic() - started
            combined = stdout + "\n" + stderr
            matches = _RAN_TESTS.findall(combined)
            tests_run = int(matches[-1]) if matches else None
            passed = (
                not timed_out
                and returncode == 0
                and tests_run is not None
                and tests_run >= minimum_tests
            )
            failures = []
            if timed_out:
                failures.append("timeout")
            if returncode is None and not timed_out:
                failures.append("execution_error")
            if returncode not in (0, None):
                failures.append("nonzero_exit")
            if tests_run is None:
                failures.append("test_count_missing")
            elif tests_run < minimum_tests:
                failures.append("too_few_tests")
            results.append(
                {
                    "check_id": check_id,
                    "kind": "unittest_discover",
                    "command": command,
                    "timeout_seconds": timeout,
                    "minimum_tests": minimum_tests,
                    "tests_run": tests_run,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "duration_seconds": elapsed,
                    "passed": passed,
                    "failure_reasons": failures,
                    "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                    "stdout_tail": _bounded(stdout),
                    "stderr_tail": _bounded(stderr),
                }
            )

    head_after, paths_after, snapshot_after = _state(candidate_path, command_runner)
    state_preserved = (
        head_after == head_before
        and paths_after == paths_before
        and snapshot_after == snapshot_before
    )
    return {
        "schema_version": "deepread-candidate-test-audit-v1",
        "candidate_audit_sha256": hashlib.sha256(
            candidate_audit_path.read_bytes()
        ).hexdigest(),
        "candidate_id": audit.get("candidate_id"),
        "plan_id": audit.get("plan_id"),
        "plan_sha256": audit.get("plan_sha256"),
        "candidate_snapshot_sha256": snapshot_before,
        "test_policy_id": policy_id,
        "test_policy_sha256": policy_sha256,
        "python_executable": str(Path(python_executable).resolve()),
        "passed": state_preserved and all(item["passed"] for item in results),
        "candidate_state_preserved": state_preserved,
        "checks": results,
    }
