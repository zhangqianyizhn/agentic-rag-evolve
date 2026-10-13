"""Read-only validation and byte-identical import of a completed baseline."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from agentic_rag_evolve.deepread_runner import load_financebench_queries
from agentic_rag_evolve.providers import ProviderSettings
from agentic_rag_evolve.store_build import verify_store_manifest
from .experiment import ExperimentConfig, _read_json, _sha256, _path_digest, _write_json, _document_manifest

_FILES = ("manifest.json", "predictions.jsonl", "deepread_trace.jsonl", "summary.json")


def validate_baseline_reuse(config: ExperimentConfig) -> dict:
    """No API calls, no writes, and no dependence on old diagnoses/plans."""
    original = config.reuse_baseline_experiment
    if original is None:
        raise ValueError("baseline reuse requires an input experiment")
    if original == config.output or original in config.output.parents or config.output in original.parents:
        raise ValueError("replay output must not overlap the original experiment")
    manifest = _read_json(original / "experiment_manifest.json")
    state = _read_json(original / "experiment_state.json")
    stage = (state.get("stages") or {}).get("baseline") or {}
    if stage.get("status") != "completed":
        raise ValueError("original baseline stage must be completed")
    root = original / "baseline"
    refs = list((stage.get("artifacts") or {}).values())
    for name in _FILES:
        path = root / name
        matches = [ref for ref in refs if Path(ref["path"]).resolve() == path.resolve()]
        if len(matches) != 1 or _sha256(path) != matches[0]["sha256"]:
            raise ValueError(f"original baseline artifact is unverified or changed: {path}")
    for key, path in (("dataset", config.dataset), ("documents", config.documents)):
        if _sha256(path) != manifest["inputs"][key + "_sha256"]:
            raise ValueError(f"baseline reuse {key} hash mismatch")
    run = _read_json(root / "manifest.json")
    if run.get("candidate_source"):
        raise ValueError("replay input must be a baseline, not a repaired candidate")
    if run.get("dataset_sha256") != _sha256(config.dataset):
        raise ValueError("baseline run dataset hash mismatch")
    runtime = run.get("config") or {}
    if runtime.get("max_rounds") != config.max_rounds or runtime.get("retrieval_topk") != config.retrieval_topk:
        raise ValueError("baseline reuse target runtime config mismatch")
    queries = load_financebench_queries(config.dataset)
    if config.task_ids:
        by_id = {query.task_id: query for query in queries}
        queries = tuple(by_id[task] for task in config.task_ids)
    if config.limit is not None:
        queries = queries[:config.limit]
    if [query.task_id for query in queries] != run.get("task_ids"):
        raise ValueError("baseline reuse task selection/order mismatch")
    settings = ProviderSettings.from_env(config.env_file)
    if run.get("providers") != {"chat": settings.llm_model, "embedding": settings.embedding_model,
                                 "reranker": settings.rerank_model}:
        raise ValueError("baseline reuse target provider models changed")
    store_manifest = config.store / "STORE_MANIFEST.json"
    store = verify_store_manifest(config.store, store_manifest)
    if _sha256(store_manifest) != (run.get("store_build") or {}).get("sha256"):
        raise ValueError("baseline reuse store provenance mismatch")
    documents = {item["document_id"]: _sha256(Path(item["path"]))
                 for item in _document_manifest(config.documents)}
    if documents != {item["document_id"]: item["sha256"] for item in store["documents"]}:
        raise ValueError("baseline reuse Markdown document content changed")
    # Framework can advance; the complete target tree must remain frozen, even
    # if the old experiment predates diagnostic bundle construction.
    revision = str(manifest["inputs"]["source_revision"])
    names = subprocess.run(["git", "-C", str(config.source_root), "ls-tree", "-r", "--name-only",
                            revision, "--", "systems/deepread"], check=True,
                           capture_output=True, text=True).stdout.splitlines()
    if not names:
        raise ValueError("original revision has no frozen DeepRead source")
    current_names = subprocess.run(
        ["git", "-C", str(config.source_root), "ls-files", "--", "systems/deepread"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    extra_names = subprocess.run(
        ["git", "-C", str(config.source_root), "ls-files", "--others", "--exclude-standard",
         "--", "systems/deepread"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if set(current_names) != set(names) or extra_names:
        raise ValueError("baseline reuse target source file set changed")
    for name in names:
        data = subprocess.run(["git", "-C", str(config.source_root), "show", f"{revision}:{name}"],
                              check=True, capture_output=True).stdout
        path = config.source_root / name
        if not path.is_file() or hashlib.sha256(data).hexdigest() != _sha256(path):
            raise ValueError(f"baseline reuse target source changed: {name}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"baseline import does not accept symlinks: {path}")
        if path.is_file():
            files.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    return {"schema_version": "deepread-baseline-reuse-v1", "original_experiment": str(original),
            "original_manifest_sha256": _sha256(original / "experiment_manifest.json"),
            "original_source_revision": revision, "files": files,
            "baseline_sha256": _path_digest(root), "task_count": len(queries)}


def import_baseline(config: ExperimentConfig) -> tuple[dict, dict]:
    provenance = validate_baseline_reuse(config)
    destination = config.output / "baseline"
    receipt = config.output / "baseline_reuse.json"
    if destination.exists():
        if not receipt.is_file() or _read_json(receipt) != provenance:
            raise ValueError("baseline reuse receipt missing or changed")
        if _path_digest(destination) != provenance["baseline_sha256"]:
            raise ValueError("imported baseline bytes changed")
    else:
        attempts = config.output / "baseline_imports"
        number = len(list(attempts.glob("attempt-*"))) + 1
        attempt = attempts / f"attempt-{number:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        staging = attempt / "baseline"
        shutil.copytree(config.reuse_baseline_experiment / "baseline", staging)
        if _path_digest(staging) != provenance["baseline_sha256"]:
            raise ValueError("baseline changed during import; partial attempt retained")
        _write_json(attempt / "receipt.json", provenance)
        # Write receipt before atomic publication; resume can validate either state.
        _write_json(receipt, provenance)
        staging.rename(destination)
    summary = _read_json(destination / "summary.json")
    return {"primary": destination / "manifest.json", "predictions": destination / "predictions.jsonl",
            "trace": destination / "deepread_trace.jsonl", "summary": destination / "summary.json",
            "reuse_receipt": receipt}, {**summary, "baseline_reused": True,
                                        "original_experiment": provenance["original_experiment"]}
