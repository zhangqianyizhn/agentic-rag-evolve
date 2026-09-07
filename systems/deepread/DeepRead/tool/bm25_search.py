from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .utils import _round_score, simple_tokenize


def bm25_search(
    doc_index: Any,
    query: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    top_k: int = 2,
    k1: float = 1.5,
    b: float = 0.75,
    include_images: bool = True,
    neighbor_window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    if not query:
        return {"ok": False, "error": "empty query"}

    q_terms = simple_tokenize(query)
    if not q_terms:
        return {"ok": False, "error": "empty query"}

    docs = doc_index.par_docs
    if scope == "doc" and doc_id is not None:
        did = str(doc_id)
        docs = [d for d in doc_index.par_docs if d["doc_id"] == did]
        if not docs:
            return {
                "ok": False,
                "error": f"no paragraphs under doc '{did}'",
                "query": query,
                "scope": scope,
                "doc_id": did,
                "results": [],
            }

    scores: List[Tuple[float, Dict[str, Any]]] = []
    for doc in docs:
        score_val = 0.0
        doc_len = doc["len"]
        term_counts: Dict[str, int] = {}
        for term in doc["tokens"]:
            term_counts[term] = term_counts.get(term, 0) + 1
        for term in q_terms:
            if term not in term_counts:
                continue
            idf_val = doc_index.idf.get(term, 0.0)
            tf = term_counts[term]
            denom = tf + k1 * (1 - b + b * doc_len / doc_index.avgdl)
            score_val += idf_val * (tf * (k1 + 1)) / (denom + 1e-9)
        if score_val > 0:
            scores.append((score_val, doc))

    scores.sort(key=lambda x: x[0], reverse=True)

    hits: List[Dict[str, Any]] = []
    k = max(1, int(top_k))

    for raw_score, doc in scores[:k]:
        did = doc["doc_id"]
        nid = doc["node_id"]
        paragraph_index = doc["p_idx"]

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
                "score": _round_score(raw_score),
                "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": paragraph_indexes},
                "text": doc["text"],
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
