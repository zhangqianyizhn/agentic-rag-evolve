"""Immutable linear baseline registry for materialized DeepRead candidates."""

from __future__ import annotations

import hashlib
import json
import fcntl
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


GitRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _git(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    )


@contextmanager
def _registry_lock(registry_root: Path):
    registry_root.mkdir(parents=True, exist_ok=True)
    with (registry_root / ".baseline-registry.lock").open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _full_commit(repo: Path, revision: str, runner: GitRunner) -> str:
    commit = runner(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"]
    ).stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"revision did not resolve to a full commit: {revision}")
    return commit


def _tree(repo: Path, commit: str, runner: GitRunner) -> str:
    return runner(
        ["git", "-C", str(repo), "show", "-s", "--format=%T", commit]
    ).stdout.strip()


def _ensure_durable_ref(
    repo: Path, *, baseline_id: str, commit: str, runner: GitRunner
) -> str:
    reference = f"refs/agentic-rag-evolve/baselines/{baseline_id}"
    try:
        existing = runner(
            ["git", "-C", str(repo), "rev-parse", "--verify", reference]
        ).stdout.strip()
    except subprocess.CalledProcessError:
        runner(
            ["git", "-C", str(repo), "update-ref", reference, commit, ""]
        )
    else:
        if existing != commit:
            raise ValueError(f"baseline durable ref points to another commit: {reference}")
    return reference


def _basis(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generation": entry.get("generation"),
        "commit": entry.get("commit"),
        "tree": entry.get("tree"),
        "parent_baseline_id": entry.get("parent_baseline_id"),
        "parent_commit": entry.get("parent_commit"),
        "materialization_id": entry.get("materialization_id"),
        "materialization_path": entry.get("materialization_path"),
        "materialization_sha256": entry.get("materialization_sha256"),
        "source_repo": entry.get("source_repo"),
        "status": entry.get("status"),
    }


def _validated_entries(registry_root: Path) -> list[dict[str, Any]]:
    paths = sorted((Path(registry_root) / "entries").glob("*.json"))
    entries = []
    identifiers = set()
    for path in paths:
        entry, _ = _object(path, "baseline entry")
        if entry.get("schema_version") != "deepread-baseline-entry-v1":
            raise ValueError(f"unsupported baseline entry schema: {path}")
        basis_hash = _canonical_sha256(_basis(entry))
        expected_id = f"baseline-{int(entry.get('generation')):04d}-{basis_hash[:16]}"
        if entry.get("entry_basis_sha256") != basis_hash or entry.get("baseline_id") != expected_id:
            raise ValueError(f"baseline entry identity is invalid: {path}")
        expected_ref = f"refs/agentic-rag-evolve/baselines/{expected_id}"
        if entry.get("durable_ref") != expected_ref:
            raise ValueError(f"baseline durable ref is invalid: {path}")
        if entry["baseline_id"] in identifiers:
            raise ValueError(f"duplicate baseline_id: {entry['baseline_id']}")
        identifiers.add(entry["baseline_id"])
        entries.append(entry)
    entries.sort(key=lambda item: int(item["generation"]))
    for index, entry in enumerate(entries):
        if int(entry["generation"]) != index:
            raise ValueError("baseline registry generations must be contiguous")
        if index == 0:
            if entry.get("parent_baseline_id") is not None or entry.get("parent_commit") is not None:
                raise ValueError("genesis baseline cannot have a parent")
            if entry.get("materialization_sha256") is not None:
                raise ValueError("genesis baseline cannot reference a materialization")
        else:
            parent = entries[index - 1]
            if entry.get("parent_baseline_id") != parent.get("baseline_id"):
                raise ValueError("baseline registry is not a linear chain")
            if entry.get("parent_commit") != parent.get("commit"):
                raise ValueError("baseline parent commit is inconsistent")
            if not entry.get("materialization_sha256"):
                raise ValueError("advanced baseline has no materialization hash")
    return entries


def _write_entry(destination: Path, entry: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"baseline entry already exists: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(entry, ensure_ascii=False, indent=2) + "\n"
    if temporary.exists():
        if temporary.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"conflicting baseline entry temporary file: {temporary}")
    else:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
    temporary.replace(destination)


def current_baseline(registry_root: Path) -> dict[str, Any]:
    """Return the unique registry tip after validating the entire chain."""

    entries = _validated_entries(Path(registry_root).resolve())
    if not entries:
        raise ValueError("baseline registry is not initialized")
    return entries[-1]


def _initialize_baseline_registry(
    *,
    registry_root: Path,
    repo_root: Path,
    revision: str,
    git_runner: GitRunner = _git,
) -> tuple[dict[str, Any], Path]:
    """Create generation zero from one explicit full-repository commit."""

    registry_root = Path(registry_root).resolve()
    repo_root = Path(repo_root).resolve()
    if _validated_entries(registry_root):
        raise FileExistsError("baseline registry is already initialized")
    commit = _full_commit(repo_root, revision, git_runner)
    tree = _tree(repo_root, commit, git_runner)
    entry: dict[str, Any] = {
        "schema_version": "deepread-baseline-entry-v1",
        "generation": 0,
        "commit": commit,
        "tree": tree,
        "parent_baseline_id": None,
        "parent_commit": None,
        "materialization_id": None,
        "materialization_path": None,
        "materialization_sha256": None,
        "source_repo": str(repo_root),
        "status": "registered",
    }
    basis_hash = _canonical_sha256(_basis(entry))
    entry["entry_basis_sha256"] = basis_hash
    entry["baseline_id"] = f"baseline-0000-{basis_hash[:16]}"
    entry["durable_ref"] = _ensure_durable_ref(
        repo_root,
        baseline_id=entry["baseline_id"],
        commit=commit,
        runner=git_runner,
    )
    destination = registry_root / "entries" / f"0000-{basis_hash[:16]}.json"
    _write_entry(destination, entry)
    return entry, destination


def _advance_baseline_registry(
    *,
    registry_root: Path,
    materialization_path: Path,
    git_runner: GitRunner = _git,
) -> tuple[dict[str, Any], Path]:
    """Append one verified materialized commit to the current baseline chain."""

    registry_root = Path(registry_root).resolve()
    entries = _validated_entries(registry_root)
    if not entries:
        raise ValueError("baseline registry is not initialized")
    parent = entries[-1]
    materialization_path = Path(materialization_path).resolve()
    materialization, materialization_sha = _object(
        materialization_path, "candidate materialization"
    )
    if materialization.get("schema_version") != "deepread-candidate-materialization-v1":
        raise ValueError("unsupported candidate materialization schema")
    if materialization.get("status") != "materialized_detached":
        raise ValueError("candidate materialization is not complete")
    if materialization.get("branch_created") is not False:
        raise ValueError("candidate materialization unexpectedly created a branch")
    if materialization.get("baseline_updated") is not False:
        raise ValueError("candidate materialization already claims baseline mutation")
    if materialization.get("base_commit") != parent.get("commit"):
        raise ValueError("materialization does not extend the current baseline")
    materialized_commit = str(materialization.get("materialized_commit") or "")
    candidate_path = Path(str(materialization.get("candidate_path") or "")).resolve()
    commit = _full_commit(candidate_path, materialized_commit, git_runner)
    if commit != materialized_commit:
        raise ValueError("materialized commit is not canonical")
    parent_line = git_runner(
        ["git", "-C", str(candidate_path), "rev-list", "--parents", "-n", "1", commit]
    ).stdout.strip().split()
    if parent_line != [commit, parent["commit"]]:
        raise ValueError("materialized commit does not have the current baseline as parent")
    tree = _tree(candidate_path, commit, git_runner)
    if tree != materialization.get("tree"):
        raise ValueError("materialized commit tree does not match artifact")
    materialization_basis = {
        "decision_id": materialization.get("decision_id"),
        "outcome_record_sha256": materialization.get("outcome_record_sha256"),
        "materialized_commit": commit,
    }
    expected_materialization_id = (
        "materialization-" + _canonical_sha256(materialization_basis)[:20]
    )
    if materialization.get("materialization_id") != expected_materialization_id:
        raise ValueError("candidate materialization identity is invalid")
    outcome_path_value = materialization.get("outcome_record")
    if not isinstance(outcome_path_value, str):
        raise ValueError("candidate materialization has no outcome record")
    outcome, outcome_sha = _object(Path(outcome_path_value), "candidate outcome")
    if outcome_sha != materialization.get("outcome_record_sha256"):
        raise ValueError("candidate outcome changed after materialization")
    if (
        outcome.get("schema_version") != "deepread-candidate-outcome-v1"
        or outcome.get("outcome") != "accepted"
        or outcome.get("decision_id") != materialization.get("decision_id")
    ):
        raise ValueError("materialization does not reference an accepted outcome")

    generation = int(parent["generation"]) + 1
    entry = {
        "schema_version": "deepread-baseline-entry-v1",
        "generation": generation,
        "commit": commit,
        "tree": tree,
        "parent_baseline_id": parent["baseline_id"],
        "parent_commit": parent["commit"],
        "materialization_id": materialization["materialization_id"],
        "materialization_path": str(materialization_path),
        "materialization_sha256": materialization_sha,
        "source_repo": str(candidate_path),
        "status": "registered",
    }
    basis_hash = _canonical_sha256(_basis(entry))
    entry["entry_basis_sha256"] = basis_hash
    entry["baseline_id"] = f"baseline-{generation:04d}-{basis_hash[:16]}"
    entry["durable_ref"] = _ensure_durable_ref(
        candidate_path,
        baseline_id=entry["baseline_id"],
        commit=commit,
        runner=git_runner,
    )
    destination = (
        registry_root / "entries" / f"{generation:04d}-{basis_hash[:16]}.json"
    )
    _write_entry(destination, entry)
    return entry, destination


def initialize_baseline_registry(
    *,
    registry_root: Path,
    repo_root: Path,
    revision: str,
    git_runner: GitRunner = _git,
) -> tuple[dict[str, Any], Path]:
    registry_root = Path(registry_root).resolve()
    with _registry_lock(registry_root):
        return _initialize_baseline_registry(
            registry_root=registry_root,
            repo_root=repo_root,
            revision=revision,
            git_runner=git_runner,
        )


def advance_baseline_registry(
    *,
    registry_root: Path,
    materialization_path: Path,
    git_runner: GitRunner = _git,
) -> tuple[dict[str, Any], Path]:
    registry_root = Path(registry_root).resolve()
    with _registry_lock(registry_root):
        return _advance_baseline_registry(
            registry_root=registry_root,
            materialization_path=materialization_path,
            git_runner=git_runner,
        )
