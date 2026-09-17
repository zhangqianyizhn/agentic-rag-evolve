#!/usr/bin/env python3
"""Advance the baseline registry with one materialized candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import advance_baseline_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Advance the DeepRead baseline registry")
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    args = parser.parse_args()
    entry, path = advance_baseline_registry(
        registry_root=args.registry_root,
        materialization_path=args.materialization,
    )
    print(json.dumps({"entry_path": str(path), "entry": entry}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
