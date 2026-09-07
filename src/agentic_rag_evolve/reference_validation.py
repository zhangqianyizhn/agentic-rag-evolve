"""Read-only validation of historical DeepRead stores and run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StoreValidation:
    document_count: int
    node_count: int
    vector_count: int
    vector_dimension: int
    model_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "node_count": self.node_count,
            "vector_count": self.vector_count,
            "vector_dimension": self.vector_dimension,
            "model_names": list(self.model_names),
        }


def _artifact_path(corpus_path: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    if path.is_file():
        return path
    local = corpus_path.parent / path.name
    if local.is_file():
        return local
    raise FileNotFoundError(f"referenced vector artifact not found: {configured}")


def validate_store(store_path: Path) -> StoreValidation:
    store_path = Path(store_path)
    corpus_paths = tuple(sorted(store_path.glob("*_corpus.json")))
    if not corpus_paths:
        raise FileNotFoundError(f"no corpus files found in {store_path}")

    node_count = 0
    vector_count = 0
    dimensions: set[int] = set()
    models: set[str] = set()
    for corpus_path in corpus_paths:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        nodes = corpus.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"invalid nodes in {corpus_path.name}")
        node_count += len(nodes)
        nodes_by_id = {str(node.get("id")): node for node in nodes}

        vector_store = corpus.get("vector_store")
        if not isinstance(vector_store, dict):
            raise ValueError(f"missing vector_store in {corpus_path.name}")
        matrix_path = _artifact_path(corpus_path, str(vector_store["matrix_path"]))
        idmap_path = _artifact_path(corpus_path, str(vector_store["id_map_path"]))
        matrix = np.load(matrix_path, mmap_mode="r")
        idmap = json.loads(idmap_path.read_text(encoding="utf-8"))
        if matrix.ndim != 2 or matrix.shape[0] != len(idmap):
            raise ValueError(f"matrix/idmap mismatch in {corpus_path.name}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite embedding in {matrix_path.name}")
        for entry in idmap:
            node_id = str(entry.get("node_id"))
            paragraph_index = int(entry.get("paragraph_index", -1))
            node = nodes_by_id.get(node_id)
            if node is None or not 0 <= paragraph_index < len(node.get("paragraphs", [])):
                raise ValueError(f"invalid idmap reference in {idmap_path.name}")
        vector_count += int(matrix.shape[0])
        dimensions.add(int(matrix.shape[1]))
        models.add(str(vector_store.get("model_name", "")))

    if len(dimensions) != 1:
        raise ValueError(f"inconsistent vector dimensions: {sorted(dimensions)}")
    return StoreValidation(
        document_count=len(corpus_paths),
        node_count=node_count,
        vector_count=vector_count,
        vector_dimension=dimensions.pop(),
        model_names=tuple(sorted(models)),
    )


def validate_historical_run(run_path: Path) -> dict[str, Any]:
    run_path = Path(run_path)
    generated = json.loads((run_path / "generated_answers.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_path / "benchmark_metrics_report.json").read_text(encoding="utf-8"))
    results = generated.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("generated_answers.json has no results")
    required = {"question", "sample_id", "llm", "retrieval", "token_usage"}
    for index, result in enumerate(results):
        missing = required - result.keys()
        if missing:
            raise ValueError(f"historical result {index} missing fields: {sorted(missing)}")
    return {
        "result_count": len(results),
        "dataset": generated.get("summary", {}).get("dataset"),
        "average_accuracy": metrics.get("Performance Metrics", {}).get(
            "Average Accuracy (normalization)"
        ),
        "files": sorted(path.name for path in run_path.iterdir() if path.is_file()),
    }
