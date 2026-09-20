#!/usr/bin/env python3
"""Audit candidate diff and scope without modifying the worktree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import audit_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an isolated DeepRead candidate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--modification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate audit: {args.output}")
    result = audit_candidate(
        manifest_path=args.manifest,
        plan_path=args.plan,
        modification_path=args.modification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
