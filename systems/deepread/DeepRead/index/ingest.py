from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from .embedding import build_embeddings
from .markdown_parser import parse_markdown_to_corpus
from .pdf_parser import run_pdf_ocr

MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _safe_name(name: str) -> str:
    cleaned = Path(name).stem.strip()
    if not cleaned:
        raise ValueError("document name cannot be empty")
    return cleaned


def _output_dir(input_path: Path, output: Optional[str]) -> Path:
    if output:
        return Path(output)
    return input_path.with_suffix("")


def parse_document(
    input_file: str,
    *,
    output: Optional[str] = None,
    name: Optional[str] = None,
    build_embedding_index: bool = False,
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    embedding_batch_size: int = 64,
    embed_base_url: str = "http://127.0.0.1:8756/v1",
    embed_api_key: str = "",
    paddle_vl_rec_backend: str = "vllm-server",
    paddle_vl_rec_server_url: str = "http://127.0.0.1:8956/v1",
) -> Path:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_file}")

    suffix = input_path.suffix.lower()
    if suffix not in MARKDOWN_SUFFIXES and suffix != ".pdf":
        raise ValueError("unsupported input format; use .pdf, .md, or .markdown")

    basename = _safe_name(name or input_path.stem)
    output_dir = _output_dir(input_path, output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".pdf":
        print(f"Running OCR for PDF: {input_path}")
        md_path = run_pdf_ocr(
            input_path,
            output_dir=output_dir,
            basename=basename,
            paddle_vl_rec_backend=paddle_vl_rec_backend,
            paddle_vl_rec_server_url=paddle_vl_rec_server_url,
        )
    else:
        md_path = output_dir / f"{basename}.md"
        if input_path.resolve() != md_path.resolve():
            shutil.copyfile(input_path, md_path)

    print(f"Parsing Markdown: {md_path}")
    corpus = parse_markdown_to_corpus(str(md_path))

    if build_embedding_index:
        print("Building embedding index")
        corpus = build_embeddings(
            corpus,
            output_dir=output_dir,
            basename=basename,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embed_base_url=embed_base_url,
            embed_api_key=embed_api_key,
        )

    corpus_path = output_dir / f"{basename}_corpus.json"
    with corpus_path.open("w", encoding="utf-8") as f_corpus:
        json.dump(corpus, f_corpus, indent=2, ensure_ascii=False)

    print(f"Corpus saved: {corpus_path}")
    return corpus_path


def add_parse_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="PDF or Markdown file")
    parser.add_argument("-o", "--output", default=None, help="output directory; default is input path without suffix")
    parser.add_argument("--name", default=None, help="output file basename; default is input filename stem")
    parser.add_argument("--build-embeddings", action="store_true", help="build *_emb.npy and *_idmap.json")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
        help="embedding model name/path",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        help="embedding request batch size",
    )
    parser.add_argument(
        "--embed-api-key",
        default=os.getenv("EMBED_API_KEY", ""),
        help="embedding API key",
    )
    parser.add_argument(
        "--embed-base-url",
        default=os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8756/v1"),
        help="embedding API base URL",
    )
    parser.add_argument(
        "--paddle-vl-rec-backend",
        default="vllm-server",
        help="PaddleOCRVL vl_rec_backend",
    )
    parser.add_argument(
        "--paddle-vl-rec-server-url",
        default="http://127.0.0.1:8956/v1",
        help="PaddleOCRVL vl_rec_server_url",
    )


def run_parse(args: argparse.Namespace) -> Path:
    return parse_document(
        args.input,
        output=args.output,
        name=args.name,
        build_embedding_index=args.build_embeddings,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        paddle_vl_rec_backend=args.paddle_vl_rec_backend,
        paddle_vl_rec_server_url=args.paddle_vl_rec_server_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF or Markdown into a DeepRead corpus.")
    add_parse_arguments(parser)
    run_parse(parser.parse_args())


if __name__ == "__main__":
    main()
