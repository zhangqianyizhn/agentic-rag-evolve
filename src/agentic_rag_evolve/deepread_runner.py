"""Stable single-dataset runner for an injected DeepRead target."""

from __future__ import annotations

import json
import hashlib
import inspect
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agentic_rag_evolve.providers import ProviderBundle
from agentic_rag_evolve.evolution.audit import verify_candidate_snapshot
from systems.deepread.runtime import DeepReadConfig, GlobalDeepReadRuntime
from agentic_rag_evolve.store_build import verify_store_manifest


@dataclass(frozen=True, slots=True)
class QueryInput:
    task_id: str
    sample_id: str
    question: str
    source_index: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store_fingerprint(store_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(store_path).glob("*_corpus.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def load_financebench_queries(dataset_path: Path) -> tuple[QueryInput, ...]:
    queries: list[QueryInput] = []
    with Path(dataset_path).open("r", encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            queries.append(
                QueryInput(
                    task_id=str(record.get("financebench_id") or source_index),
                    sample_id=str(record["doc_name"]),
                    question=str(record["question"]).strip(),
                    source_index=source_index,
                )
            )
    if not queries:
        raise ValueError(f"dataset contains no queries: {dataset_path}")
    return tuple(queries)


def run_financebench(
    *,
    dataset_path: Path,
    store_path: Path,
    output_path: Path,
    providers: ProviderBundle,
    config: DeepReadConfig,
    limit: int | None = None,
    task_ids: Sequence[str] | None = None,
    store_manifest_path: Path | None = None,
    candidate_audit_path: Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    queries = load_financebench_queries(dataset_path)
    if task_ids is not None:
        requested = [str(task_id) for task_id in task_ids]
        if not requested:
            raise ValueError("task_ids must not be empty when provided")
        if len(set(requested)) != len(requested):
            raise ValueError("task_ids contains duplicates")
        by_id = {query.task_id: query for query in queries}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"task_ids not found in dataset: {missing}")
        queries = tuple(by_id[task_id] for task_id in requested)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        queries = queries[:limit]

    store_build = None
    if store_manifest_path is not None:
        store_manifest_path = Path(store_manifest_path).resolve()
        store_build_value = verify_store_manifest(store_path, store_manifest_path)
        store_build = {
            "path": str(store_manifest_path),
            "sha256": _sha256_file(store_manifest_path),
            "candidate": store_build_value.get("candidate"),
        }

    candidate_source = None
    if candidate_audit_path is not None:
        candidate_audit_path = Path(candidate_audit_path).resolve()
        audit_bytes = candidate_audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        if audit.get("schema_version") != "deepread-candidate-audit-v1" or not audit.get("passed"):
            raise ValueError("a passing candidate audit is required for a candidate run")
        candidate_root = Path(str(audit.get("candidate_path") or "")).resolve()
        verify_candidate_snapshot(candidate_root, audit)
        loaded_runtime = Path(inspect.getfile(GlobalDeepReadRuntime)).resolve()
        expected_runtime = (candidate_root / "systems/deepread/runtime.py").resolve()
        if loaded_runtime != expected_runtime:
            raise ValueError(
                "loaded DeepRead runtime does not come from the audited candidate checkout"
            )
        candidate_source = {
            "candidate_id": audit.get("candidate_id"),
            "plan_id": audit.get("plan_id"),
            "candidate_snapshot_sha256": audit.get("candidate_snapshot_sha256"),
            "candidate_audit_path": str(candidate_audit_path),
            "candidate_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        }
        store_candidate = (store_build or {}).get("candidate")
        if store_candidate is not None and store_candidate.get(
            "candidate_snapshot_sha256"
        ) != candidate_source["candidate_snapshot_sha256"]:
            raise ValueError("candidate store and runtime snapshots do not match")
        if store_candidate is not None and store_candidate.get(
            "candidate_audit_sha256"
        ) != candidate_source["candidate_audit_sha256"]:
            raise ValueError("candidate store and runtime audits do not match")

    created_at = datetime.now(timezone.utc)
    run_id = f"run_{created_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    trace_path = output_path / "deepread_trace.jsonl"
    runtime = GlobalDeepReadRuntime(
        store_path,
        trace_path,
        config,
        chat_model=providers.chat,
        embedding_model=providers.embedding,
        reranker=providers.reranker,
        run_id=run_id,
    )
    runtime.load_index()

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "dataset_file": Path(dataset_path).name,
        "dataset_sha256": _sha256_file(dataset_path),
        "store_name": Path(store_path).name,
        "store_fingerprint": _store_fingerprint(store_path),
        "query_count": len(queries),
        "task_ids": [query.task_id for query in queries],
        "config": asdict(config),
        "providers": {
            "chat": providers.chat.model_name,
            "embedding": providers.embedding.model_name,
            "reranker": providers.reranker.model_name if providers.reranker else None,
        },
        "store_build": store_build,
        "candidate_source": candidate_source,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction_path = output_path / "predictions.jsonl"
    completed = 0
    with prediction_path.open("x", encoding="utf-8") as predictions:
        for query in queries:
            started = time.perf_counter()
            try:
                result = runtime.query(query.question, task_id=query.task_id)
                record = {
                    **asdict(query),
                    "run_id": run_id,
                    "status": "ok",
                    "answer": result.answer,
                    "termination_reason": result.termination_reason,
                    "rounds_completed": result.rounds_completed,
                    "retrieved_texts": list(result.retrieved_texts),
                    "token_usage": {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    "latency_seconds": time.perf_counter() - started,
                }
                completed += 1
            except Exception as exc:
                record = {
                    **asdict(query),
                    "run_id": run_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_seconds": time.perf_counter() - started,
                }
            predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions.flush()

    summary = {"query_count": len(queries), "completed": completed, "failed": len(queries) - completed}
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
