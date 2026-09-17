#!/usr/bin/env python3
"""Authorize candidate rediagnosis for development regressions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.validation import build_regression_feedback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build sealed-boundary development regression feedback"
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feedback = build_regression_feedback(
        gate_path=args.gate,
        suite_path=args.suite,
        candidate_audit_path=args.candidate_audit,
        output_path=args.output,
    )
    print(json.dumps(feedback, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
