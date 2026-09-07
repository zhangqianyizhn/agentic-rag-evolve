from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .retrieval import DocIndex


def _resolve_image_paths(nodes: List[Dict[str, Any]], corpus_dir: Path) -> None:
    for node in nodes:
        for paragraph in node.get("paragraphs", []):
            if not isinstance(paragraph, dict) or paragraph.get("type") != "image":
                continue
            image_path = paragraph.get("image_path")
            if not image_path:
                continue
            image_file = Path(str(image_path)).expanduser()
            if image_file.is_absolute() or str(image_path).startswith(("http://", "https://", "data:")):
                continue
            paragraph["image_path"] = str((corpus_dir / image_file).resolve())


def load_corpus(paths: List[str], neighbor_window: Optional[Tuple[int, int]]) -> DocIndex:
    all_nodes: List[Dict[str, Any]] = []
    matrices: List[np.ndarray] = []
    idmaps: List[Dict[str, Any]] = []
    models: List[str] = []
    normalized_flags: List[bool] = []
    doc_id_map: Dict[str, str] = {}

    for idx, path in enumerate(paths):
        corpus_path = Path(path).expanduser().resolve()
        with corpus_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("corpus JSON must contain a non-empty 'nodes' list")

        doc_id = str(idx + 1)
        # 从 corpus 文件名提取 human-readable 文档名：3M_2015_10K_corpus.json -> 3M_2015_10K
        doc_name = corpus_path.stem
        if doc_name.endswith("_corpus"):
            doc_name = doc_name[:-7]
        doc_id_map[doc_id] = doc_name

        for node in nodes:
            node["doc_id"] = doc_id
        _resolve_image_paths(nodes, corpus_path.parent)
        all_nodes.extend(nodes)

        vector_store = data.get("vector_store")
        if not isinstance(vector_store, dict):
            continue

        matrix_path = vector_store.get("matrix_path")
        id_map_path = vector_store.get("id_map_path")
        if not matrix_path or not id_map_path:
            continue

        matrix_file = Path(matrix_path).expanduser()
        id_map_file = Path(id_map_path).expanduser()
        if not matrix_file.is_absolute():
            matrix_file = corpus_path.parent / matrix_file
        if not id_map_file.is_absolute():
            id_map_file = corpus_path.parent / id_map_file
        if not matrix_file.exists() or not id_map_file.exists():
            continue

        matrix = np.load(matrix_file, mmap_mode="r").astype(np.float32)
        with id_map_file.open("r", encoding="utf-8") as f_id:
            id_map = json.load(f_id) or []
        for entry in id_map:
            entry["doc_id"] = doc_id

        matrices.append(matrix)
        idmaps.extend(id_map)
        models.append(str(vector_store.get("model_name")))
        normalized_flags.append(bool(vector_store.get("normalized", False)))

    doc_index = DocIndex(all_nodes, neighbor_window=neighbor_window)
    doc_index.doc_id_map = doc_id_map

    if matrices:
        doc_index._vec_matrix = np.concatenate(matrices, axis=0)
        doc_index._vec_idmap = idmaps
        doc_index._vec_model_name = models[0] if models else None
        doc_index._vec_normalized = all(normalized_flags) if normalized_flags else False

    return doc_index
