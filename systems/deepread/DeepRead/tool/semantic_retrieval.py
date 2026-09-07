from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..ports import EmbeddingModel, Reranker
from .utils import _round_score


def _with_neighbors(
    doc_index: Any,
    candidate: Dict[str, Any],
    *,
    include_images: bool,
    neighbor_window: Optional[Tuple[int, int]],
    score: float,
) -> Dict[str, Any]:
    ref = candidate.get("ref") or {}
    did = str(ref.get("doc_id"))
    nid = str(ref.get("node_id"))
    para_list = ref.get("paragraph_indexes") or []
    paragraph_index = int(para_list[0]) if para_list else 0

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

    return {
        "score": _round_score(score),
        "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": paragraph_indexes},
        "text": candidate.get("text", ""),
        "neighbors": neighbors,
    }


def semantic_retrieval(
    doc_index: Any,
    query: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    stage1_method: str = "vector",
    top_k1: int = 30,
    top_k2: int = 5,
    stage1_hybrid_topk_bm25: int = 50,
    stage1_hybrid_topk_vec: int = 50,
    include_images: bool = True,
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
    hybrid_bm25_weight: float = 0.5,
    hybrid_vector_weight: float = 0.5,
) -> Dict[str, Any]:
    if not query:
        return {"ok": False, "error": "empty query"}

    k1 = max(1, int(top_k1))
    k2 = max(1, int(top_k2))
    if k1 < k2:
        k1 = k2

    stage1 = (stage1_method or "vector").lower().strip()
    if stage1 not in ("vector", "bm25", "hybrid"):
        stage1 = "vector"

    no_neighbor: Optional[Tuple[int, int]] = (0, 0)
    if stage1 == "bm25":
        candidates_res = doc_index.bm25_search(
            query=query,
            scope=scope,
            doc_id=doc_id,
            top_k=k1,
            include_images=False,
            neighbor_window=no_neighbor,
        )
    elif stage1 == "hybrid":
        candidates_res = doc_index.hybrid_search(
            query=query,
            scope=scope,
            doc_id=doc_id,
            top_k=k1,
            bm25_weight=hybrid_bm25_weight,
            vector_weight=hybrid_vector_weight,
            top_k_bm25=max(k1, int(stage1_hybrid_topk_bm25)),
            top_k_vec=max(k1, int(stage1_hybrid_topk_vec)),
            include_images=False,
            embedding_model=embedding_model,
            neighbor_window=no_neighbor,
        )
    else:
        candidates_res = doc_index.vector_search(
            query=query,
            scope=scope,
            doc_id=doc_id,
            top_k=k1,
            include_images=False,
            embedding_model=embedding_model,
            neighbor_window=no_neighbor,
        )

    if not candidates_res.get("ok", False):
        return {
            "ok": False,
            "error": f"stage1_{stage1}_failed",
            "query": query,
            "scope": scope,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "stage1_method": stage1,
            "top_k1": k1,
            "top_k2": k2,
            "results": [],
        }

    candidates = candidates_res.get("results", []) or []
    docs = [str(c.get("text", "")) for c in candidates]
    if not candidates or not any(docs):
        return {
            "ok": True,
            "query": query,
            "scope": scope,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "stage1_method": stage1,
            "top_k1": k1,
            "top_k2": k2,
            "results": [],
        }

    try:
        if reranker is None:
            raise RuntimeError("reranker not configured")
        rerank_data = reranker.rerank(query, docs, top_n=-1)
    except Exception as exc:
        hits = [
            _with_neighbors(
                doc_index,
                candidate,
                include_images=include_images,
                neighbor_window=neighbor_window,
                score=float(candidate.get("score", 0.0)),
            )
            for candidate in candidates[:k2]
        ]
        return {
            "ok": True,
            "query": query,
            "scope": scope,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "stage1_method": stage1,
            "top_k1": k1,
            "top_k2": k2,
            "rerank_ok": False,
            "rerank_error": str(exc),
            "results": hits,
        }

    hits: List[Dict[str, Any]] = []
    for item in (rerank_data.get("results", []) or [])[:k2]:
        try:
            idx = int(item.get("index", -1))
        except Exception:
            idx = -1
        if idx < 0 or idx >= len(candidates):
            continue

        try:
            score = float(item.get("relevance_score", 0.0))
        except Exception:
            score = 0.0
        hits.append(
            _with_neighbors(
                doc_index,
                candidates[idx],
                include_images=include_images,
                neighbor_window=neighbor_window,
                score=score,
            )
        )

    return {
        "ok": True,
        "query": query,
        "scope": scope,
        "doc_id": str(doc_id) if doc_id is not None else None,
        "stage1_method": stage1,
        "top_k1": k1,
        "top_k2": k2,
        "rerank_ok": True,
        "rerank_model": reranker.model_name,
        "results": hits,
    }
