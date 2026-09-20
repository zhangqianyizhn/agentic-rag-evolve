#!/usr/bin/env python3
"""Build a terminal report for an iteration with no proceeding candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.reporting import build_no_candidate_iteration_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a no-candidate iteration report")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_no_candidate_iteration_report(
        cohort_path=args.cohort,
        hypotheses_path=args.hypotheses,
        plan_path=args.plan,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
