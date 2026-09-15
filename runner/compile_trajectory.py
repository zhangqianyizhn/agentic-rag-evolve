#!/usr/bin/env python3
"""Compile raw DeepRead events into one normalized trajectory per task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.trajectory import DEFAULT_INLINE_RESULT_BYTES, compile_trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile DeepRead JSONL trajectories")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inline-result-bytes",
        type=int,
        default=DEFAULT_INLINE_RESULT_BYTES,
        help="Externalize tool results larger than this many UTF-8 JSON bytes",
    )
    args = parser.parse_args()
    report = compile_trajectories(
        trace_path=args.trace,
        prediction_path=args.predictions,
        output_path=args.output,
        inline_result_bytes=args.inline_result_bytes,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 1 if report.unassigned_event_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
