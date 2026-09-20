#!/usr/bin/env python3
"""Run or resume configured steps for one evolution iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.orchestration import execute_iteration


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume an evolution iteration")
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    report = execute_iteration(
        ledger_root=args.ledger_root,
        runbook_path=args.runbook,
        max_steps=args.max_steps,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 1 if report.status == "step_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
