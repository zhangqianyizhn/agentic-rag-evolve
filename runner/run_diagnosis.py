#!/usr/bin/env python3
"""Run one evidence-grounded DeepRead diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.diagnosis import run_diagnosis
from agentic_rag_evolve.providers import load_chat_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DeepRead diagnosis agent")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()
    report = run_diagnosis(
        bundle_path=args.bundle,
        source_root=args.source_root,
        output_path=args.output,
        model=load_chat_model(args.env_file),
        max_rounds=args.max_rounds,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0 if report.status in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
