#!/usr/bin/env python3
"""Build a bounded failure-memory context for one hypothesis set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import build_planning_memory_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Select relevant repair memory")
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-entries-per-hypothesis", type=int, default=12)
    args = parser.parse_args()
    result = build_planning_memory_context(
        memory_root=args.memory_root,
        cohort_path=args.cohort,
        hypotheses_path=args.hypotheses,
        output_path=args.output,
        max_entries_per_hypothesis=args.max_entries_per_hypothesis,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
