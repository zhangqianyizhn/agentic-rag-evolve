#!/usr/bin/env python3
"""Compile raw DeepRead events into one normalized trajectory per task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.trajectory import compile_trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile DeepRead JSONL trajectories")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_trajectories(
        trace_path=args.trace,
        prediction_path=args.predictions,
        output_path=args.output,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 1 if report.unassigned_event_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
