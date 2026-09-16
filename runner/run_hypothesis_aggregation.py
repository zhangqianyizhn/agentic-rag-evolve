#!/usr/bin/env python3
"""Aggregate a repair cohort into grounded improvement hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import run_hypothesis_aggregation
from agentic_rag_evolve.providers import load_chat_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate DeepRead improvement hypotheses")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--request-retry-max-seconds", type=float, default=120.0)
    args = parser.parse_args()
    report = run_hypothesis_aggregation(
        cohort_path=args.cohort,
        output_path=args.output,
        model=load_chat_model(
            args.env_file,
            timeout=args.request_timeout,
            max_retries=args.request_max_retries,
            retry_base_seconds=args.request_retry_base_seconds,
            retry_max_seconds=args.request_retry_max_seconds,
        ),
        max_output_tokens=args.max_output_tokens,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0 if report.status in {"ok", "no_eligible_diagnoses"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
