"""Markdown-only ingestion for the DeepRead v0 target system."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from agentic_rag_evolve.contracts import DocumentInput

from .DeepRead.index import parse_markdown_to_corpus


class EmbeddingProvider(Protocol):
    """Minimal embedding boundary used by ingestion.

    Provider-specific clients and credentials stay outside the corpus builder.
    """

    model_name: str
    base_url: str
    normalized: bool

    def embed(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    document_id: str
    source_path: Path
    markdown_path: Path
    corpus_path: Path
    embedding_path: Path | None
    id_map_path: Path | None
    paragraph_count: int


def _safe_basename(path: Path) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", path.stem)[:120]
    return name or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _embedding_inputs(corpus: dict) -> tuple[list[str], list[dict[str, object]]]:
    texts: list[str] = []
    id_map: list[dict[str, object]] = []
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


class MarkdownIngestor:
    """Build the flat corpus layout expected by ``GlobalDeepReadRuntime``."""

    def __init__(self, store_path: Path, embedder: EmbeddingProvider | None = None) -> None:
        self.store_path = Path(store_path)
        self.embedder = embedder

    def ingest(self, documents: Sequence[DocumentInput]) -> tuple[IngestedDocument, ...]:
        documents = tuple(documents)
        if not documents:
            raise ValueError("documents must contain at least one Markdown document")

        basenames = [_safe_basename(document.path) for document in documents]
        if len(basenames) != len(set(basenames)):
            raise ValueError("document paths produce duplicate output basenames")

        for document in documents:
            if document.path.suffix.lower() not in {".md", ".markdown"}:
                raise ValueError(f"unsupported document format (Markdown required): {document.path}")
            if not document.path.is_file():
                raise FileNotFoundError(f"Markdown document not found: {document.path}")

        self.store_path.mkdir(parents=True, exist_ok=True)
        results = [
            self._ingest_one(document, basename)
            for document, basename in zip(documents, basenames, strict=True)
        ]
        return tuple(results)

    def _ingest_one(self, document: DocumentInput, basename: str) -> IngestedDocument:
        markdown_path = self.store_path / f"{basename}.md"
        if document.path.resolve() != markdown_path.resolve():
            shutil.copy2(document.path, markdown_path)

        corpus = parse_markdown_to_corpus(str(markdown_path))
        texts, id_map = _embedding_inputs(corpus)
        embedding_path: Path | None = None
        id_map_path: Path | None = None

        if self.embedder is not None and texts:
            vectors = [list(self.embedder.embed(text)) for text in texts]
            dimensions = {len(vector) for vector in vectors}
            if not dimensions or 0 in dimensions or len(dimensions) != 1:
                raise ValueError("embedding provider returned empty or inconsistent vectors")

            embedding_path = self.store_path / f"{basename}_emb.npy"
            id_map_path = self.store_path / f"{basename}_idmap.json"
            np.save(embedding_path, np.asarray(vectors, dtype=np.float16))
            id_map_path.write_text(
                json.dumps(id_map, ensure_ascii=False),
                encoding="utf-8",
            )
            corpus["vector_store"] = {
                "matrix_path": str(embedding_path.resolve()),
                "id_map_path": str(id_map_path.resolve()),
                "model_name": self.embedder.model_name,
                "normalized": bool(self.embedder.normalized),
                "dtype": "float16",
                "embed_base_url": self.embedder.base_url,
            }

        corpus_path = self.store_path / f"{basename}_corpus.json"
        corpus_path.write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return IngestedDocument(
            document_id=document.document_id,
            source_path=document.path,
            markdown_path=markdown_path,
            corpus_path=corpus_path,
            embedding_path=embedding_path,
            id_map_path=id_map_path,
            paragraph_count=len(texts),
        )
