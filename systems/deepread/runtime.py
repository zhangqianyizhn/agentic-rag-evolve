"""Standalone global-mode adapter for the frozen DeepRead v0 implementation."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag_evolve.telemetry import JsonlTraceWriter, ScopedTraceWriter

from .DeepRead.agent import run_agent
from .DeepRead.ports import ChatModel, EmbeddingModel, Reranker
from .DeepRead.runtime_state import token_tracker
from .DeepRead.tool import DocIndex, load_corpus
from .DeepRead.tool.utils import _normalize_neighbor_window


@dataclass(frozen=True, slots=True)
class DeepReadConfig:
    temperature: float = 0.0
    enable_vector: bool = True
    enable_hybrid: bool = False
    enable_semantic: bool = False
    neighbor_window: tuple[int, int] | None = (1, -1)
    max_rounds: int = 50
    retrieval_topk: int = 5

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.retrieval_topk < 1:
            raise ValueError("retrieval_topk must be at least 1")
        _normalize_neighbor_window(self.neighbor_window)


@dataclass(frozen=True, slots=True)
class DeepReadQueryResult:
    answer: str
    retrieved_texts: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    trace_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieved_texts": list(self.retrieved_texts),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "trace_path": str(self.trace_path),
        }


class GlobalDeepReadRuntime:
    """Load one shared corpus index and execute DeepRead queries against it.

    Corpus discovery intentionally matches ruc-ov-eval ``fb8a301c``: only
    ``*_corpus.json`` files directly under ``store_path`` are loaded, sorted by
    filename. Supporting recursive layouts would change document ids and is
    therefore deferred until it can be introduced as an explicit migration.
    """

    def __init__(
        self,
        store_path: Path,
        trace_path: Path,
        config: DeepReadConfig,
        *,
        chat_model: ChatModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
        run_id: str | None = None,
    ) -> None:
        self.store_path = Path(store_path)
        self.trace_path = Path(trace_path)
        self.config = config
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.run_id = run_id
        self._logger: JsonlTraceWriter | None = None
        self._logger_lock = threading.Lock()
        self._doc_index: DocIndex | None = None
        self._index_lock = threading.Lock()

    def discover_corpus_paths(self) -> tuple[Path, ...]:
        if not self.store_path.is_dir():
            raise FileNotFoundError(f"DeepRead store directory not found: {self.store_path}")
        paths = tuple(sorted(self.store_path.glob("*_corpus.json")))
        if not paths:
            raise FileNotFoundError(
                f"No *_corpus.json files found directly under: {self.store_path}"
            )
        return paths

    def load_index(self) -> DocIndex:
        if self._doc_index is not None:
            return self._doc_index
        with self._index_lock:
            if self._doc_index is None:
                paths = self.discover_corpus_paths()
                self._doc_index = load_corpus(
                    [str(path) for path in paths],
                    neighbor_window=self.config.neighbor_window,
                )
        return self._doc_index

    def invalidate_index(self) -> None:
        with self._index_lock:
            self._doc_index = None

    def _get_logger(self) -> JsonlTraceWriter:
        if self._logger is not None:
            return self._logger
        with self._logger_lock:
            if self._logger is None:
                self._logger = JsonlTraceWriter(self.trace_path)
        return self._logger

    def write_document_map(self, output_path: Path) -> None:
        doc_index = self.load_index()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(doc_index.doc_id_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def query(self, question: str, *, task_id: str | None = None) -> DeepReadQueryResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if self.chat_model is None:
            raise RuntimeError("chat_model is required to execute a query")

        token_tracker.reset()
        retrieved_texts: list[str] = []
        logger = ScopedTraceWriter(
            self._get_logger(),
            {"run_id": self.run_id, "task_id": task_id},
        )

        answer = run_agent(
            chat_model=self.chat_model,
            doc_index=self.load_index(),
            user_question=question,
            logger=logger,
            max_rounds=self.config.max_rounds,
            temperature=self.config.temperature,
            enable_vector=self.config.enable_vector,
            enable_hybrid=self.config.enable_hybrid,
            enable_semantic=self.config.enable_semantic,
            disable_bm25=False,
            disable_regex=False,
            disable_read=False,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            neighbor_window=self.config.neighbor_window,
            bm25_topk=self.config.retrieval_topk,
            regex_topk=self.config.retrieval_topk,
            vector_topk=self.config.retrieval_topk,
            hybrid_topk=self.config.retrieval_topk,
            semantic_topk1=30,
            semantic_topk2=1,
            collected_texts=retrieved_texts,
        )
        usage = token_tracker.get()
        return DeepReadQueryResult(
            answer=answer,
            retrieved_texts=tuple(retrieved_texts),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            trace_path=self.trace_path,
        )
