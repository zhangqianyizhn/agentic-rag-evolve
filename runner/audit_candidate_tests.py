#!/usr/bin/env python3
"""Run a fixed test policy against one statically audited candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_rag_evolve.evolution import audit_candidate_tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed tests for a DeepRead candidate")
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate test audit: {args.output}")
    result = audit_candidate_tests(
        candidate_audit_path=args.candidate_audit,
        policy_path=args.policy,
        python_executable=args.python,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
