#!/usr/bin/env python3
"""Create a bounded modification plan from improvement hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import run_modification_planning
from agentic_rag_evolve.providers import load_chat_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan bounded DeepRead modifications")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--memory-context", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--request-retry-max-seconds", type=float, default=120.0)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    args = parser.parse_args()
    report = run_modification_planning(
        cohort_path=args.cohort,
        hypotheses_path=args.hypotheses,
        output_path=args.output,
        model=load_chat_model(
            args.env_file,
            timeout=args.request_timeout,
            max_retries=args.request_max_retries,
            retry_base_seconds=args.request_retry_base_seconds,
            retry_max_seconds=args.request_retry_max_seconds,
        ),
        max_output_tokens=args.max_output_tokens,
        memory_context_path=args.memory_context,
        source_root=args.source_root,
        candidate_audit_path=args.candidate_audit,
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0 if report.status in {"ok", "no_plannable_hypotheses"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
