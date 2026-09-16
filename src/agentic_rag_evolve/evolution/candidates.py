"""Create detached, reproducible candidate worktrees from approved plans."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    )


def _load_plan(path: Path, plan_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    data = Path(path).read_bytes()
    document = json.loads(data)
    if document.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")
    matches = [item for item in document.get("plans") or [] if item.get("plan_id") == plan_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one plan with id {plan_id!r}")
    plan = matches[0]
    if plan.get("decision") != "proceed":
        raise ValueError(f"plan {plan_id!r} is not approved to proceed")
    scope = plan.get("edit_scope") or {}
    if not scope.get("allowed_paths") or int(scope.get("max_files_to_modify") or 0) < 1:
        raise ValueError(f"plan {plan_id!r} has no editable scope")
    return document, plan, hashlib.sha256(data).hexdigest()


def _commit(repo_root: Path, revision: str, runner: CommandRunner) -> str:
    result = runner(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{revision}^{{commit}}"]
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"revision did not resolve to a full commit: {revision}")
    return commit


def create_candidate_worktree(
    *,
    repo_root: Path,
    plan_path: Path,
    plan_id: str,
    base_revision: str,
    candidate_path: Path,
    manifest_path: Path,
    command_runner: CommandRunner = _run,
) -> dict[str, Any]:
    """Create one detached worktree; never overwrite or clean an existing path."""

    repo_root = Path(repo_root).resolve()
    candidate_path = Path(candidate_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if candidate_path == repo_root or repo_root in candidate_path.parents:
        raise ValueError("candidate path must be outside the source repository")
    if candidate_path.exists():
        raise FileExistsError(f"candidate path already exists: {candidate_path}")
    if manifest_path.exists():
        raise FileExistsError(f"candidate manifest already exists: {manifest_path}")
    if manifest_path == candidate_path or candidate_path in manifest_path.parents:
        raise ValueError("candidate manifest must be stored outside the worktree")

    document, plan, plan_sha256 = _load_plan(Path(plan_path), plan_id)
    base_commit = _commit(repo_root, base_revision, command_runner)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    command_runner(
        [
            "git",
            "-C",
            str(repo_root),
            "worktree",
            "add",
            "--detach",
            str(candidate_path),
            base_commit,
        ]
    )
    manifest = {
        "schema_version": "deepread-candidate-manifest-v1",
        "candidate_id": f"{plan_id}-{base_commit[:12]}",
        "cohort_id": document.get("cohort_id"),
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "base_commit": base_commit,
        "source_repo": str(repo_root),
        "candidate_path": str(candidate_path),
        "isolation": "detached_git_worktree",
        "allowed_paths": list(plan["edit_scope"]["allowed_paths"]),
        "max_files_to_modify": int(plan["edit_scope"]["max_files_to_modify"]),
        "forbidden_roots": list(plan["edit_scope"].get("forbidden_roots") or []),
        "status": "created",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest
