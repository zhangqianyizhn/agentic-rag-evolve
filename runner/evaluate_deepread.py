#!/usr/bin/env python3
"""Framework-owned evaluator entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evaluation.comparison import compare_with_historical
from agentic_rag_evolve.evaluation.evaluator import evaluate_financebench, write_evaluation
from agentic_rag_evolve.providers import load_chat_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DeepRead FinanceBench predictions")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", action="store_true", help="Use the configured LLM as a 0-4 judge")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--historical", type=Path)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"evaluation output directory must be empty: {args.output}")
    judge_model = load_chat_model(args.env_file) if args.judge else None
    details, summary = evaluate_financebench(
        dataset_path=args.dataset,
        prediction_path=args.predictions,
        judge_model=judge_model,
    )
    write_evaluation(args.output, details, summary)
    if args.historical:
        comparison = compare_with_historical(details, args.historical)
        (args.output / "historical_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary.to_dict(), ensure_ascii=False))
    return 0 if not summary.prediction_errors and not summary.judge_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
