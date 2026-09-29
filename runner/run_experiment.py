#!/usr/bin/env python3
"""Run or resume the initial DeepRead experiment through one stable entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_rag_evolve.orchestration.experiment import ExperimentConfig, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a recoverable DeepRead baseline or diagnosis experiment"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--mode", choices=("baseline", "diagnose"), default="baseline")
    parser.add_argument(
        "--judge", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--diagnosis-max-rounds", type=int, default=12)
    parser.add_argument("--diagnosis-max-tool-calls", type=int)
    parser.add_argument("--diagnosis-max-output-tokens", type=int)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--request-retry-max-seconds", type=float, default=120.0)
    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        source_root=args.source_root,
        documents=args.documents,
        dataset=args.dataset,
        store=args.store,
        output=args.output,
        env_file=args.env_file,
        mode=args.mode,
        judge=args.judge,
        limit=args.limit,
        task_ids=tuple(args.task_id),
        max_rounds=args.max_rounds,
        retrieval_topk=args.retrieval_topk,
        diagnosis_max_rounds=args.diagnosis_max_rounds,
        diagnosis_max_tool_calls=args.diagnosis_max_tool_calls,
        diagnosis_max_output_tokens=args.diagnosis_max_output_tokens,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        request_retry_base_seconds=args.request_retry_base_seconds,
        request_retry_max_seconds=args.request_retry_max_seconds,
    )
    state = run_experiment(
        config,
        resume=args.resume,
        progress=lambda message: print(f"[experiment] {message}", file=sys.stderr, flush=True),
    )
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
