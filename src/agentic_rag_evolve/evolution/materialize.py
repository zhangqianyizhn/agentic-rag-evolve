"""Materialize an accepted candidate snapshot as a detached Git commit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .audit import candidate_snapshot_sha256, collect_changed_paths


GitRunner = Callable[..., subprocess.CompletedProcess[str]]
ARTIFACT_KEYS = {
    "candidate_manifest",
    "modification_plan",
    "candidate_audit",
    "candidate_test_audit",
    "validation_suite",
    "validation_gate",
}


def _git(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True, **kwargs
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, hashlib.sha256(data).hexdigest()


def _verify_artifacts(outcome: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = outcome.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != ARTIFACT_KEYS:
        raise ValueError("candidate outcome has an incomplete artifact set")
    resolved = {}
    for name in sorted(ARTIFACT_KEYS):
        reference = artifacts.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"candidate outcome artifact {name} is invalid")
        path_value = reference.get("path")
        digest = str(reference.get("sha256") or "")
        if not isinstance(path_value, str) or len(digest) != 64:
            raise ValueError(f"candidate outcome artifact {name} is incomplete")
        path = Path(path_value).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"candidate outcome artifact changed: {name}")
        resolved[name] = path
    return resolved


def materialize_accepted_candidate(
    *, outcome_path: Path, git_runner: GitRunner = _git
) -> dict[str, Any]:
    """Commit the exact accepted diff in its detached worktree, without promotion."""

    outcome_path = Path(outcome_path).resolve()
    outcome, outcome_sha256 = _load_json(outcome_path, "candidate outcome")
    if outcome.get("schema_version") != "deepread-candidate-outcome-v1":
        raise ValueError("unsupported candidate outcome schema")
    if outcome.get("outcome") != "accepted" or outcome.get("candidate_status") != "accepted":
        raise ValueError("only an accepted candidate can be materialized")
    if outcome.get("promotion_status") != "eligible_for_materialization":
        raise ValueError("candidate is not eligible for materialization")
    if outcome.get("git_mutation_performed") is not False:
        raise ValueError("candidate outcome is not a pre-materialization record")
    artifacts = _verify_artifacts(outcome)
    gate_sha256 = str(outcome["artifacts"]["validation_gate"]["sha256"])
    basis = {
        "candidate_id": outcome.get("candidate_id"),
        "candidate_snapshot_sha256": outcome.get("candidate_snapshot_sha256"),
        "validation_gate_sha256": gate_sha256,
        "outcome": "accepted",
    }
    basis_sha256 = _canonical_sha256(basis)
    if outcome.get("decision_basis_sha256") != basis_sha256:
        raise ValueError("candidate outcome decision basis is invalid")
    if outcome.get("decision_id") != "outcome-" + basis_sha256[:20]:
        raise ValueError("candidate outcome decision_id is invalid")

    manifest, _ = _load_json(artifacts["candidate_manifest"], "candidate manifest")
    audit, _ = _load_json(artifacts["candidate_audit"], "candidate audit")
    if manifest.get("schema_version") != "deepread-candidate-manifest-v1":
        raise ValueError("unsupported candidate manifest schema")
    if audit.get("schema_version") != "deepread-candidate-audit-v1" or not audit.get("passed"):
        raise ValueError("candidate static audit is not passing")
    for field in ("candidate_id", "plan_id", "base_commit", "candidate_path"):
        expected = outcome.get(field)
        if manifest.get(field) != expected:
            raise ValueError(f"candidate manifest {field} does not match outcome")
    for field in ("candidate_id", "plan_id", "base_commit", "candidate_path"):
        if audit.get(field) != outcome.get(field):
            raise ValueError(f"candidate audit {field} does not match outcome")
    snapshot = str(outcome.get("candidate_snapshot_sha256") or "")
    if audit.get("candidate_snapshot_sha256") != snapshot:
        raise ValueError("candidate audit snapshot does not match outcome")
    changed_paths = audit.get("changed_paths")
    if not isinstance(changed_paths, list) or not changed_paths:
        raise ValueError("accepted candidate has no audited changes")

    candidate_path = Path(str(outcome.get("candidate_path") or "")).resolve()
    if not candidate_path.is_dir():
        raise ValueError(f"candidate worktree is missing: {candidate_path}")
    inside = git_runner(
        ["git", "-C", str(candidate_path), "rev-parse", "--is-inside-work-tree"]
    ).stdout.strip()
    if inside != "true":
        raise ValueError("candidate path is not a Git worktree")
    base_commit = str(outcome.get("base_commit") or "")
    epoch_text = git_runner(
        ["git", "-C", str(candidate_path), "show", "-s", "--format=%ct", base_commit]
    ).stdout.strip()
    try:
        commit_epoch = int(epoch_text) + 1
    except ValueError as exc:
        raise ValueError("base commit has no valid timestamp") from exc
    commit_date = f"{commit_epoch} +0000"
    identity = {
        "GIT_AUTHOR_NAME": "AgenticRAGEvolve",
        "GIT_AUTHOR_EMAIL": "agentic-rag-evolve@local",
        "GIT_AUTHOR_DATE": commit_date,
        "GIT_COMMITTER_NAME": "AgenticRAGEvolve",
        "GIT_COMMITTER_EMAIL": "agentic-rag-evolve@local",
        "GIT_COMMITTER_DATE": commit_date,
    }
    environment = os.environ.copy()
    environment.update(identity)
    message = (
        f"Materialize {outcome['candidate_id']} {snapshot[:12]}\n\n"
        f"Outcome: {outcome['decision_id']}"
    )

    def result(commit: str, tree: str) -> dict[str, Any]:
        materialization_basis = {
            "decision_id": outcome.get("decision_id"),
            "outcome_record_sha256": outcome_sha256,
            "materialized_commit": commit,
        }
        return {
            "schema_version": "deepread-candidate-materialization-v1",
            "materialization_id": (
                "materialization-" + _canonical_sha256(materialization_basis)[:20]
            ),
            "decision_id": outcome.get("decision_id"),
            "candidate_id": outcome.get("candidate_id"),
            "plan_id": outcome.get("plan_id"),
            "candidate_snapshot_sha256": snapshot,
            "outcome_record": str(outcome_path),
            "outcome_record_sha256": outcome_sha256,
            "base_commit": base_commit,
            "materialized_commit": commit,
            "tree": tree,
            "candidate_path": str(candidate_path),
            "commit_identity": identity,
            "commit_message": message,
            "status": "materialized_detached",
            "branch_created": False,
            "baseline_updated": False,
        }

    head = git_runner(
        ["git", "-C", str(candidate_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    if head != base_commit:
        parent_line = git_runner(
            ["git", "-C", str(candidate_path), "rev-list", "--parents", "-n", "1", head]
        ).stdout.strip().split()
        status = git_runner(
            ["git", "-C", str(candidate_path), "status", "--porcelain"]
        ).stdout.strip()
        current_snapshot = candidate_snapshot_sha256(
            candidate_path, head_commit=base_commit, changed_paths=changed_paths
        )
        committed_message = git_runner(
            ["git", "-C", str(candidate_path), "show", "-s", "--format=%B", head]
        ).stdout.strip()
        committed_paths_output = git_runner(
            [
                "git", "-C", str(candidate_path), "diff", "--name-only",
                "--no-renames", "-z", base_commit, head,
            ]
        ).stdout
        committed_paths = sorted(
            item for item in committed_paths_output.split("\0") if item
        )
        tree = git_runner(
            ["git", "-C", str(candidate_path), "show", "-s", "--format=%T", head]
        ).stdout.strip()
        expected_commit = git_runner(
            [
                "git", "-C", str(candidate_path), "commit-tree", tree,
                "-p", base_commit, "-m", message,
            ],
            env=environment,
        ).stdout.strip()
        if (
            parent_line != [head, base_commit]
            or status
            or current_snapshot != snapshot
            or committed_message != message
            or committed_paths != changed_paths
            or expected_commit != head
        ):
            raise ValueError("candidate HEAD changed before materialization")
        return result(head, tree)
    current_paths = collect_changed_paths(candidate_path, git_runner)
    if current_paths != changed_paths:
        raise ValueError("candidate changed paths differ from accepted audit")
    current_snapshot = candidate_snapshot_sha256(
        candidate_path, head_commit=head, changed_paths=current_paths
    )
    if current_snapshot != snapshot:
        raise ValueError("candidate source snapshot changed before materialization")

    git_runner(
        ["git", "-C", str(candidate_path), "add", "-A", "--", *changed_paths]
    )
    staged = git_runner(
        [
            "git", "-C", str(candidate_path), "diff", "--cached", "--name-only",
            "--no-renames", "-z", base_commit,
        ]
    ).stdout
    staged_paths = sorted(item for item in staged.split("\0") if item)
    if staged_paths != changed_paths:
        raise ValueError("staged candidate paths differ from accepted audit")
    tree = git_runner(
        ["git", "-C", str(candidate_path), "write-tree"]
    ).stdout.strip()
    materialized_commit = git_runner(
        [
            "git", "-C", str(candidate_path), "commit-tree", tree,
            "-p", base_commit, "-m", message,
        ],
        env=environment,
    ).stdout.strip()
    parent_line = git_runner(
        [
            "git", "-C", str(candidate_path), "rev-list", "--parents", "-n", "1",
            materialized_commit,
        ]
    ).stdout.strip().split()
    if parent_line != [materialized_commit, base_commit]:
        raise ValueError("materialized commit parent is not the accepted base")
    git_runner(
        ["git", "-C", str(candidate_path), "reset", "--hard", materialized_commit]
    )
    final_head = git_runner(
        ["git", "-C", str(candidate_path), "rev-parse", "HEAD"]
    ).stdout.strip()
    status = git_runner(
        ["git", "-C", str(candidate_path), "status", "--porcelain"]
    ).stdout.strip()
    if final_head != materialized_commit or status:
        raise RuntimeError("candidate worktree did not reach a clean materialized state")
    return result(materialized_commit, tree)
