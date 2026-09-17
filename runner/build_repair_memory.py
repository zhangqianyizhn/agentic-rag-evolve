#!/usr/bin/env python3
"""Convert one rejected candidate outcome into immutable repair memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import build_repair_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rejected-candidate repair memory")
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    args = parser.parse_args()
    record, path = build_repair_memory(
        outcome_path=args.outcome, memory_root=args.memory_root
    )
    print(json.dumps({"memory_path": str(path), "memory": record}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
