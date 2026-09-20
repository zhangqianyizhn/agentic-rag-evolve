#!/usr/bin/env python3
"""Build a compact report for one terminal evolution iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.reporting import build_iteration_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact iteration report")
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--terminal-memory", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--regression-feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_iteration_report(
        outcome_path=args.outcome,
        terminal_memory_path=args.terminal_memory,
        cohort_path=args.cohort,
        hypotheses_path=args.hypotheses,
        regression_feedback_path=args.regression_feedback,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
