#!/usr/bin/env python3
"""Record constraints and gains from an accepted, registered baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import build_preservation_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Build accepted-candidate preservation memory")
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--baseline-entry", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    args = parser.parse_args()
    record, path = build_preservation_memory(
        outcome_path=args.outcome,
        materialization_path=args.materialization,
        baseline_entry_path=args.baseline_entry,
        memory_root=args.memory_root,
    )
    print(
        json.dumps(
            {"preservation_memory_path": str(path), "memory": record},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
