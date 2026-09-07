from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ports import EmbeddingModel
from .utils import _round_score


def vector_search(
    doc_index: Any,
    query: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    top_k: int = 2,
    include_images: bool = True,
    embedding_model: EmbeddingModel | None = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    if not query:
        return {"ok": False, "error": "empty query"}
    if doc_index._vec_matrix is None or not len(doc_index._vec_idmap):
        return {"ok": False, "error": "vector_store not available"}
    if embedding_model is None:
        return {"ok": False, "error": "embedding_model not configured"}

    q_vec = np.asarray(embedding_model.embed(query), dtype=np.float32)
    if q_vec.ndim != 1 or not q_vec.size:
        return {"ok": False, "error": "embedding_failed"}
    if q_vec.shape[0] != doc_index._vec_matrix.shape[1]:
        return {
            "ok": False,
            "error": "embedding_dimension_mismatch",
            "expected": int(doc_index._vec_matrix.shape[1]),
            "actual": int(q_vec.shape[0]),
        }
    q_norm = float(np.linalg.norm(q_vec)) + 1e-12

    idxs = list(range(len(doc_index._vec_idmap)))
    if scope == "doc" and doc_id is not None:
        did = str(doc_id)
        idxs = [
            i
            for i, meta in enumerate(doc_index._vec_idmap)
            if str(meta.get("doc_id") or doc_index.node_to_doc_id.get(str(meta.get("node_id")), "")) == did
        ]
        if not idxs:
            return {
                "ok": False,
                "error": f"no embeddings under doc '{did}'",
                "query": query,
                "scope": scope,
                "doc_id": did,
                "results": [],
            }

    matrix = doc_index._vec_matrix
    sims: List[float] = []
    for i in idxs:
        v = np.asarray(matrix[i], dtype=np.float32)
        v_norm = 1.0 if doc_index._vec_normalized else (float(np.linalg.norm(v)) + 1e-12)
        sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
        sims.append(max(-1.0, min(1.0, sim)))

    order = np.argsort(sims)[::-1]
    k = max(1, int(top_k))

    hits: List[Dict[str, Any]] = []
    for rank in order[:k]:
        global_idx = idxs[int(rank)]
        meta = doc_index._vec_idmap[global_idx]
        nid = str(meta.get("node_id"))
        did = str(meta.get("doc_id") or doc_index.node_to_doc_id.get(nid, ""))
        paragraph_index = int(meta.get("paragraph_index", 0))

        node = (doc_index.nodes_by_doc.get(did) or {}).get(nid) or {"paragraphs": []}
        text = ""
        paragraphs = node.get("paragraphs", [])
        if 0 <= paragraph_index < len(paragraphs):
            paragraph = paragraphs[paragraph_index]
            if isinstance(paragraph, str):
                text = paragraph
            elif isinstance(paragraph, dict):
                text = paragraph.get("content", "")
            else:
                text = str(paragraph)

        neighbors = doc_index._neighbor_context_for(
            did,
            nid,
            paragraph_index,
            include_images=include_images,
            neighbor_window=neighbor_window,
        )
        index_set = {paragraph_index}
        for item in neighbors:
            try:
                index_set.add(int(item["paragraph_index"]))
            except Exception:
                continue
        paragraph_indexes = sorted(index_set)

        hits.append(
            {
                "score": _round_score(sims[int(rank)]),
                "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": paragraph_indexes},
                "text": text,
                "neighbors": neighbors,
            }
        )

    return {
        "ok": True,
        "query": query,
        "scope": scope,
        "doc_id": str(doc_id) if doc_id is not None else None,
        "results": hits,
    }
