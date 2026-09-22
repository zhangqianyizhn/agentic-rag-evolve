#!/usr/bin/env python3
"""Build a provenance-bound DeepRead Markdown store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.providers import load_embedding_model
from agentic_rag_evolve.store_build import build_markdown_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fresh DeepRead Markdown store")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()

    manifest = build_markdown_store(
        document_manifest_path=args.documents,
        output_path=args.output,
        source_root=args.source_root,
        embedder=None if args.lexical_only else load_embedding_model(args.env_file),
        candidate_audit_path=args.candidate_audit,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
