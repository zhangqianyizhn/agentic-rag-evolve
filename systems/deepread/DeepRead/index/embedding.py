from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def http_embed(
    model: str,
    inputs: List[str],
    *,
    base_url: str,
    api_key: str,
) -> List[List[float]]:
    import requests

    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": inputs}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return [item.get("embedding") for item in data.get("data", [])]


def extract_embedding_inputs(corpus: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    id_map: List[Dict[str, Any]] = []

    for node in corpus.get("nodes", []):
        node_id = node.get("id")
        for paragraph_index, paragraph in enumerate(node.get("paragraphs", [])):
            if isinstance(paragraph, str):
                text = paragraph.strip()
            elif isinstance(paragraph, dict):
                text = str(paragraph.get("content", "")).strip()
            else:
                text = str(paragraph).strip()

            if not text:
                continue
            texts.append(text)
            id_map.append({"node_id": node_id, "paragraph_index": paragraph_index})

    return texts, id_map


def build_embeddings(
    corpus: Dict[str, Any],
    *,
    output_dir: Path,
    basename: str,
    embedding_model: str,
    embedding_batch_size: int,
    embed_base_url: str,
    embed_api_key: str,
) -> Dict[str, Any]:
    texts, id_map = extract_embedding_inputs(corpus)
    if not texts:
        return corpus

    batch_size = max(1, int(embedding_batch_size))
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(
            http_embed(
                embedding_model,
                batch,
                base_url=embed_base_url,
                api_key=embed_api_key,
            )
        )

    arr = np.asarray(embeddings, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / (norm + 1e-12)

    emb_path = output_dir / f"{basename}_emb.npy"
    idmap_path = output_dir / f"{basename}_idmap.json"

    np.save(emb_path, arr.astype(np.float16))
    with idmap_path.open("w", encoding="utf-8") as f_id:
        json.dump(id_map, f_id, ensure_ascii=False)

    corpus["vector_store"] = {
        "matrix_path": str(emb_path),
        "id_map_path": str(idmap_path),
        "model_name": embedding_model,
        "normalized": True,
        "dtype": "float16",
        "embed_base_url": embed_base_url,
    }
    return corpus
