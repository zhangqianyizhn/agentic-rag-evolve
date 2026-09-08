"""DeepRead tool dispatch without telemetry or provider transport concerns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..ports import EmbeddingModel, Reranker
from .retrieval import DocIndex


@dataclass(slots=True)
class DeepReadToolExecutor:
    doc_index: DocIndex
    embedding_model: EmbeddingModel | None = None
    reranker: Reranker | None = None
    enable_multimodal: bool = False
    neighbor_window: tuple[int, int] | None = None
    bm25_topk: int = 1
    regex_topk: int = 1
    vector_topk: int = 1
    hybrid_topk: int = 1
    hybrid_topk_bm25: int = 30
    hybrid_topk_vec: int = 30
    hybrid_bm25_weight: float = 0.5
    hybrid_vector_weight: float = 0.5
    semantic_stage1_method: str = "vector"
    semantic_topk1: int = 30
    semantic_topk2: int = 1
    semantic_stage1_hybrid_topk_bm25: int = 30
    semantic_stage1_hybrid_topk_vec: int = 30

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        args = arguments
        neighbor_window = (
            self.neighbor_window
            if self.neighbor_window is not None
            else self.doc_index.neighbor_window
        )
        if name == "get_doc_structure":
            raw_ids = args.get("doc_id")
            doc_ids = [str(doc_id) for doc_id in raw_ids] if isinstance(raw_ids, list) else None
            return self.doc_index.get_doc_structure(doc_ids=doc_ids)
        if name == "read_section":
            return self.doc_index.read_section(
                doc_id=args.get("doc_id"),
                node_id=args.get("node_id"),
                start_paragraph=int(args.get("start_paragraph", 0)),
                end_paragraph=int(args.get("end_paragraph", -1)),
                include_images=self.enable_multimodal,
            )
        if name == "bm25_search":
            return self.doc_index.bm25_search(
                query=args.get("query", ""),
                scope=args.get("scope", "full"),
                doc_id=args.get("doc_id"),
                top_k=int(self.bm25_topk),
                include_images=self.enable_multimodal,
                neighbor_window=neighbor_window,
            )
        if name == "regex_search":
            return self.doc_index.regex_search(
                pattern=args.get("pattern", ""),
                scope=args.get("scope", "full"),
                doc_id=args.get("doc_id"),
                top_k=int(self.regex_topk),
                include_images=self.enable_multimodal,
                neighbor_window=neighbor_window,
            )
        if name == "vector_search":
            return self.doc_index.vector_search(
                query=args.get("query", ""),
                scope=args.get("scope", "full"),
                doc_id=args.get("doc_id"),
                top_k=int(self.vector_topk),
                include_images=self.enable_multimodal,
                embedding_model=self.embedding_model,
                neighbor_window=neighbor_window,
            )
        if name == "hybrid_search":
            return self.doc_index.hybrid_search(
                query=args.get("query", ""),
                scope=args.get("scope", "full"),
                doc_id=args.get("doc_id"),
                top_k=int(self.hybrid_topk),
                bm25_weight=float(self.hybrid_bm25_weight),
                vector_weight=float(self.hybrid_vector_weight),
                top_k_bm25=int(self.hybrid_topk_bm25),
                top_k_vec=int(self.hybrid_topk_vec),
                include_images=self.enable_multimodal,
                embedding_model=self.embedding_model,
                neighbor_window=neighbor_window,
            )
        if name == "semantic_retrieval":
            return self.doc_index.semantic_retrieval(
                query=args.get("query", ""),
                scope=args.get("scope", "full"),
                doc_id=args.get("doc_id"),
                stage1_method=str(self.semantic_stage1_method),
                top_k1=int(self.semantic_topk1),
                top_k2=int(self.semantic_topk2),
                stage1_hybrid_topk_bm25=int(self.semantic_stage1_hybrid_topk_bm25),
                stage1_hybrid_topk_vec=int(self.semantic_stage1_hybrid_topk_vec),
                include_images=self.enable_multimodal,
                embedding_model=self.embedding_model,
                reranker=self.reranker,
                neighbor_window=neighbor_window,
                hybrid_bm25_weight=float(self.hybrid_bm25_weight),
                hybrid_vector_weight=float(self.hybrid_vector_weight),
            )
        return {"ok": False, "error": f"Tool '{name}' not implemented"}
