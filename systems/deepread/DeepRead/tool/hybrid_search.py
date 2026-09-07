from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .utils import _round_score


def hybrid_search(
    doc_index: Any,
    query: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    top_k: int = 2,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k_bm25: int = 20,
    top_k_vec: int = 20,
    include_images: bool = True,
    embed_api_key: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    embed_model: Optional[str] = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    no_neighbor: Optional[Tuple[int, int]] = (0, 0)
    bm25_res = doc_index.bm25_search(
        query=query,
        scope=scope,
        doc_id=doc_id,
        top_k=top_k_bm25,
        include_images=include_images,
        neighbor_window=no_neighbor,
    )
    vec_res = doc_index.vector_search(
        query=query,
        scope=scope,
        doc_id=doc_id,
        top_k=top_k_vec,
        include_images=include_images,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        neighbor_window=no_neighbor,
    )

    if not bm25_res.get("ok", False) and not vec_res.get("ok", False):
        return {
            "ok": False,
            "error": "both bm25_search and vector_search failed",
            "query": query,
            "scope": scope,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "results": [],
        }

    bm25_map: Dict[Tuple[str, str, int], float] = {}
    bm25_max = 0.0
    for result in bm25_res.get("results", []):
        ref = result.get("ref") or {}
        did = str(ref.get("doc_id"))
        nid = str(ref.get("node_id"))
        para_list = ref.get("paragraph_indexes") or []
        if not para_list:
            continue
        paragraph_index = int(para_list[0])
        score_val = float(result.get("score", 0.0))
        key = (did, nid, paragraph_index)
        bm25_map[key] = score_val
        bm25_max = max(bm25_max, score_val)

    vec_map: Dict[Tuple[str, str, int], float] = {}
    vec_scores: List[float] = []
    for result in vec_res.get("results", []):
        ref = result.get("ref") or {}
        did = str(ref.get("doc_id"))
        nid = str(ref.get("node_id"))
        para_list = ref.get("paragraph_indexes") or []
        if not para_list:
            continue
        paragraph_index = int(para_list[0])
        score_val = float(result.get("score", 0.0))
        key = (did, nid, paragraph_index)
        vec_map[key] = score_val
        vec_scores.append(score_val)

    vec_min, vec_max = (min(vec_scores), max(vec_scores)) if vec_scores else (0.0, 1.0)

    fused: Dict[Tuple[str, str, int], float] = {}
    keys = set(bm25_map.keys()) | set(vec_map.keys())

    for key in keys:
        bm25_score = bm25_map.get(key, 0.0)
        vec_score = vec_map.get(key, 0.0)
        bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0
        vec_norm = ((vec_score - vec_min) / (vec_max - vec_min)) if (vec_max - vec_min) > 1e-12 else 0.0
        fused[key] = bm25_weight * bm25_norm + vector_weight * vec_norm

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    k = max(1, int(top_k))

    hits: List[Dict[str, Any]] = []
    for (did, nid, paragraph_index), fused_score in ordered[:k]:
        node = (doc_index.nodes_by_doc.get(did) or {}).get(nid) or {"paragraphs": []}
        paragraphs = node.get("paragraphs", [])
        text = ""
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
                "score": _round_score(fused_score),
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
