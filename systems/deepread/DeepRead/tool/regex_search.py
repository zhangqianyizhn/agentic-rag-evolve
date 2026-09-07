from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import _round_score


def regex_search(
    doc_index: Any,
    pattern: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    top_k: int = 2,
    include_images: bool = True,
    neighbor_window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    if not pattern:
        return {"ok": False, "error": "empty pattern"}
    try:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        return {"ok": False, "error": f"invalid regex pattern: {exc}"}

    docs = doc_index.par_docs
    if scope == "doc" and doc_id is not None:
        did = str(doc_id)
        docs = [d for d in doc_index.par_docs if d["doc_id"] == did]
        if not docs:
            return {
                "ok": False,
                "error": f"no paragraphs under doc '{did}'",
                "pattern": pattern,
                "scope": scope,
                "doc_id": did,
                "results": [],
            }

    matches: List[Tuple[int, Dict[str, Any]]] = []
    for doc in docs:
        found = regex.findall(doc["text"])
        if found:
            matches.append((len(found), doc))
    matches.sort(key=lambda x: x[0], reverse=True)

    hits: List[Dict[str, Any]] = []
    k = max(1, int(top_k))

    for count, doc in matches[:k]:
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
                "score": _round_score(float(count)),
                "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": paragraph_indexes},
                "text": doc["text"],
                "neighbors": neighbors,
            }
        )

    return {
        "ok": True,
        "pattern": pattern,
        "scope": scope,
        "doc_id": str(doc_id) if doc_id is not None else None,
        "results": hits,
    }
