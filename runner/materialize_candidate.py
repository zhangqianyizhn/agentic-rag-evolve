#!/usr/bin/env python3
"""Materialize one accepted candidate as a detached commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import materialize_accepted_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize an accepted candidate")
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite materialization: {args.output}")
    result = materialize_accepted_candidate(outcome_path=args.outcome)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
