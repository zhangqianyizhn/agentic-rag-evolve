"""Read-only scope and static audit for isolated DeepRead candidates."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    )


def _nul_paths(output: str) -> set[str]:
    return {item for item in output.split("\0") if item}


def collect_changed_paths(
    candidate_path: Path, command_runner: CommandRunner = _run
) -> list[str]:
    candidate_path = Path(candidate_path).resolve()
    tracked = command_runner(
        [
            "git", "-C", str(candidate_path), "diff", "--name-only", "--no-renames",
            "-z", "HEAD",
        ]
    )
    untracked = command_runner(
        [
            "git", "-C", str(candidate_path), "ls-files", "--others",
            "--exclude-standard", "-z",
        ]
    )
    return sorted(_nul_paths(tracked.stdout) | _nul_paths(untracked.stdout))


def candidate_snapshot_sha256(
    candidate_path: Path, *, head_commit: str, changed_paths: Sequence[str]
) -> str:
    """Hash the audited source state without depending on Git diff formatting."""

    candidate_path = Path(candidate_path).resolve()
    entries = []
    for relative in sorted(changed_paths):
        path = candidate_path / relative
        if path.is_symlink():
            entry = {"path": relative, "kind": "symlink", "target": str(path.readlink())}
        elif not path.exists():
            entry = {"path": relative, "kind": "deleted"}
        elif path.is_file():
            entry = {
                "path": relative,
                "kind": "file",
                "mode": stat.S_IMODE(path.stat().st_mode),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        else:
            entry = {"path": relative, "kind": "unsupported"}
        entries.append(entry)
    payload = {"head_commit": head_commit, "changed_entries": entries}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _under(path: str, roots: Sequence[str]) -> bool:
    return any(
        path == root.rstrip("/") or path.startswith(root.rstrip("/") + "/")
        for root in roots
        if root
    )


def _load_inputs(
    manifest_path: Path, plan_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    plan_bytes = Path(plan_path).read_bytes()
    plan_document = json.loads(plan_bytes)
    if manifest.get("schema_version") != "deepread-candidate-manifest-v1":
        raise ValueError("unsupported candidate manifest schema")
    if plan_document.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")
    if hashlib.sha256(plan_bytes).hexdigest() != manifest.get("plan_sha256"):
        raise ValueError("modification plan hash does not match candidate manifest")
    matches = [
        item
        for item in plan_document.get("plans") or []
        if item.get("plan_id") == manifest.get("plan_id")
    ]
    if len(matches) != 1:
        raise ValueError("candidate plan_id is missing or ambiguous")
    return manifest, plan_document, matches[0]


def audit_candidate(
    *,
    manifest_path: Path,
    plan_path: Path,
    command_runner: CommandRunner = _run,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest, _, plan = _load_inputs(manifest_path, plan_path)
    candidate_path = Path(str(manifest["candidate_path"])).resolve()
    head = command_runner(
        ["git", "-C", str(candidate_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    changed_paths = collect_changed_paths(candidate_path, command_runner)
    snapshot_sha256 = candidate_snapshot_sha256(
        candidate_path, head_commit=head, changed_paths=changed_paths
    )
    scope = plan.get("edit_scope") or {}
    allowed_paths = set(str(item) for item in scope.get("allowed_paths") or [])
    forbidden_roots = [str(item) for item in scope.get("forbidden_roots") or []]
    max_files = int(scope.get("max_files_to_modify") or 0)

    out_of_scope = [path for path in changed_paths if path not in allowed_paths]
    forbidden = [path for path in changed_paths if _under(path, forbidden_roots)]
    symlinks = [
        path
        for path in changed_paths
        if (candidate_path / path).is_symlink()
    ]
    syntax_errors = []
    for relative in changed_paths:
        path = candidate_path / relative
        if not path.exists() or path.is_symlink() or path.suffix != ".py":
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            syntax_errors.append(f"{relative}: {type(exc).__name__}: {exc}")

    try:
        diff_check = command_runner(
            ["git", "-C", str(candidate_path), "diff", "--check", "HEAD"]
        )
    except subprocess.CalledProcessError as exc:
        diff_check = subprocess.CompletedProcess(
            exc.cmd,
            exc.returncode,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    violations = []
    if head != manifest.get("base_commit"):
        violations.append("candidate_head_changed")
    if len(changed_paths) > max_files:
        violations.append("changed_files_exceed_budget")
    if out_of_scope:
        violations.append("changed_files_outside_allowed_paths")
    if forbidden:
        violations.append("forbidden_roots_touched")
    if symlinks:
        violations.append("changed_symlinks")
    if syntax_errors:
        violations.append("python_syntax_errors")
    if diff_check.returncode != 0 or diff_check.stdout or diff_check.stderr:
        violations.append("git_diff_check_failed")
    return {
        "schema_version": "deepread-candidate-audit-v1",
        "candidate_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "candidate_id": manifest.get("candidate_id"),
        "plan_id": manifest.get("plan_id"),
        "plan_sha256": manifest.get("plan_sha256"),
        "test_policy_id": manifest.get("test_policy_id"),
        "test_policy_sha256": manifest.get("test_policy_sha256"),
        "base_commit": manifest.get("base_commit"),
        "head_commit": head,
        "candidate_path": str(candidate_path),
        "candidate_snapshot_sha256": snapshot_sha256,
        "passed": not violations,
        "changed_paths": changed_paths,
        "changed_file_count": len(changed_paths),
        "max_files_to_modify": max_files,
        "allowed_paths": sorted(allowed_paths),
        "out_of_scope": out_of_scope,
        "forbidden_touches": forbidden,
        "changed_symlinks": symlinks,
        "syntax_errors": syntax_errors,
        "diff_check_output": (diff_check.stdout + diff_check.stderr).strip(),
        "violations": violations,
    }
