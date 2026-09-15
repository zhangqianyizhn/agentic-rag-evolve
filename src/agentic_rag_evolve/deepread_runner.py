"""Stable single-dataset runner for an injected DeepRead target."""

from __future__ import annotations

import json
import hashlib
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agentic_rag_evolve.providers import ProviderBundle
from systems.deepread.runtime import DeepReadConfig, GlobalDeepReadRuntime


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
        "config": asdict(config),
        "providers": {
            "chat": providers.chat.model_name,
            "embedding": providers.embedding.model_name,
            "reranker": providers.reranker.model_name if providers.reranker else None,
        },
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
