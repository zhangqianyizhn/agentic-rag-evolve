#!/usr/bin/env python3
"""Apply paired development/holdout/cross-dataset candidate gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.validation import evaluate_validation_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a DeepRead candidate validation gate")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation gate: {args.output}")
    result = evaluate_validation_gate(
        suite_path=args.suite,
        candidate_audit_path=args.candidate_audit,
        plan_path=args.plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
