from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

from .logger import JsonlLogger
from .runner import run_agent
from ..tool.corpus import load_corpus
from ..tool.utils import _normalize_neighbor_window


def add_ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("doc", nargs="+", help="one or more DeepRead *_corpus.json files")
    parser.add_argument("question", help="question to ask")
    parser.add_argument("--log", default="run_log.jsonl", help="jsonl log output path")
    parser.add_argument("--max-rounds", "--max_rounds", dest="max_rounds", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or os.getenv("OPENROUTER_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--enable-multimodal", action="store_true", help="enable image content in retrieved context")

    parser.add_argument("--tool-fallback", dest="tool_fallback", action="store_true", default=True, help="enable text fallback parsing for tool calls")
    parser.add_argument("--no-tool-fallback", dest="tool_fallback", action="store_false", help="disable text fallback parsing for tool calls")

    parser.add_argument("--enable-reasoning", dest="enable_reasoning", action="store_true", default=True, help="request provider reasoning/thinking if supported")
    parser.add_argument("--disable-reasoning", dest="enable_reasoning", action="store_false", help="disable provider reasoning/thinking")

    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    parser.add_argument("--embed-base-url", default=os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--embed-api-key", default=os.getenv("EMBED_API_KEY", ""))

    parser.add_argument("--enable-vector", action="store_true")
    parser.add_argument("--enable-hybrid", action="store_true")
    parser.add_argument("--enable-semantic", action="store_true")
    parser.add_argument(
        "--retrieval",
        choices=["bm25", "vector", "hybrid", "semantic"],
        default=None,
        help="shortcut for common retrieval modes",
    )
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--disable-regex", action="store_true")
    parser.add_argument("--disable-read", action="store_true")

    parser.add_argument("--bm25-topk", type=int, default=int(os.getenv("BM25_TOPK", "1")))
    parser.add_argument("--regex-topk", type=int, default=int(os.getenv("REGEX_TOPK", "1")))
    parser.add_argument("--vector-topk", type=int, default=int(os.getenv("VECTOR_TOPK", "1")))
    parser.add_argument("--hybrid-topk", type=int, default=int(os.getenv("HYBRID_TOPK", "1")))

    parser.add_argument("--hybrid-topk-bm25", type=int, default=int(os.getenv("HYBRID_TOPK_BM25", "30")))
    parser.add_argument("--hybrid-topk-vec", type=int, default=int(os.getenv("HYBRID_TOPK_VEC", "30")))
    parser.add_argument("--hybrid-bm25-weight", type=float, default=float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")))
    parser.add_argument("--hybrid-vector-weight", type=float, default=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.5")))

    parser.add_argument("--semantic-stage1", default=os.getenv("SEMANTIC_STAGE1", "vector"), choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--semantic-topk1", type=int, default=int(os.getenv("SEMANTIC_TOPK1", "30")))
    parser.add_argument("--semantic-topk2", type=int, default=int(os.getenv("SEMANTIC_TOPK2", "2")))
    parser.add_argument("--semantic-stage1-hybrid-topk-bm25", type=int, default=int(os.getenv("SEMANTIC_STAGE1_HYBRID_TOPK_BM25", "30")))
    parser.add_argument("--semantic-stage1-hybrid-topk-vec", type=int, default=int(os.getenv("SEMANTIC_STAGE1_HYBRID_TOPK_VEC", "30")))

    parser.add_argument("--rerank-api-key", default=os.getenv("RERANK_API_KEY", "") or os.getenv("SILICONFLOW_API_KEY", ""))
    parser.add_argument("--rerank-base-url", default=os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--rerank-model", default=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"))

    parser.add_argument(
        "--neighbor-window",
        default=os.getenv("NEIGHBOR_WINDOW", "1,-1"),
        help="neighbor window as 'up,down' where up>=0 and down<=0; use '0,0' to disable",
    )


def _parse_neighbor_window(value: str) -> Optional[tuple[int, int]]:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError("--neighbor-window must be in 'up,down' form, e.g. '1,-1' or '0,0'")
    return _normalize_neighbor_window((int(parts[0]), int(parts[1])))


def _openrouter_headers(base_url: Optional[str]) -> Optional[Dict[str, str]]:
    if base_url and "openrouter.ai" in base_url:
        return {
            "HTTP-Referer": os.getenv("ORIGIN", "http://localhost"),
            "X-Title": os.getenv("APP_NAME", "deepread"),
        }
    return None


def run_ask(args: argparse.Namespace) -> str:
    try:
        neighbor_window = _parse_neighbor_window(args.neighbor_window)
    except Exception as exc:
        raise SystemExit(f"Invalid --neighbor-window: {exc}") from exc

    if not args.api_key:
        raise RuntimeError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY, or pass --api-key")
    if not args.model:
        raise RuntimeError("Please set OPENAI_MODEL or OPENROUTER_MODEL, or pass --model")

    logger = JsonlLogger(args.log)
    doc_index = load_corpus(args.doc, neighbor_window=neighbor_window)

    if args.retrieval == "vector":
        args.enable_vector = True
        args.disable_bm25 = True
        args.disable_regex = True
    elif args.retrieval == "hybrid":
        args.enable_hybrid = True
    elif args.retrieval == "semantic":
        args.enable_semantic = True
    elif args.retrieval == "bm25":
        args.disable_bm25 = False

    answer = run_agent(
        model=args.model,
        base_url=args.base_url,
        doc_index=doc_index,
        user_question=args.question,
        logger=logger,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        api_key=args.api_key,
        default_headers=_openrouter_headers(args.base_url),
        enable_multimodal=args.enable_multimodal,
        enable_vector=args.enable_vector,
        enable_hybrid=args.enable_hybrid,
        enable_semantic=args.enable_semantic,
        disable_bm25=args.disable_bm25,
        disable_regex=args.disable_regex,
        disable_read=args.disable_read,
        embed_api_key=args.embed_api_key,
        embed_base_url=args.embed_base_url,
        embedding_model=args.embedding_model,
        neighbor_window=neighbor_window,
        bm25_topk=args.bm25_topk,
        regex_topk=args.regex_topk,
        vector_topk=args.vector_topk,
        hybrid_topk=args.hybrid_topk,
        hybrid_topk_bm25=args.hybrid_topk_bm25,
        hybrid_topk_vec=args.hybrid_topk_vec,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_vector_weight=args.hybrid_vector_weight,
        semantic_stage1_method=args.semantic_stage1,
        semantic_topk1=args.semantic_topk1,
        semantic_topk2=args.semantic_topk2,
        semantic_stage1_hybrid_topk_bm25=args.semantic_stage1_hybrid_topk_bm25,
        semantic_stage1_hybrid_topk_vec=args.semantic_stage1_hybrid_topk_vec,
        rerank_api_key=args.rerank_api_key,
        rerank_base_url=args.rerank_base_url,
        rerank_model=args.rerank_model,
        tool_fallback=args.tool_fallback,
        enable_reasoning=args.enable_reasoning,
    )
    print("\n==== Final Answer ====")
    print(answer)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over DeepRead corpus files.")
    add_ask_arguments(parser)
    run_ask(parser.parse_args())


if __name__ == "__main__":
    main()
