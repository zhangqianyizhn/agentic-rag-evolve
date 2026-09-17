#!/usr/bin/env python3
"""Initialize the immutable baseline registry at one explicit commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import initialize_baseline_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the DeepRead baseline registry")
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    entry, path = initialize_baseline_registry(
        registry_root=args.registry_root,
        repo_root=args.repo_root,
        revision=args.revision,
    )
    print(json.dumps({"entry_path": str(path), "entry": entry}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
