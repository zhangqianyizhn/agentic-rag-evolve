#!/usr/bin/env python3
"""Validate the user-provided historical DeepRead artifacts without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.reference_validation import validate_historical_run, validate_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "store": validate_store(args.store).to_dict(),
        "historical_run": validate_historical_run(args.run),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
