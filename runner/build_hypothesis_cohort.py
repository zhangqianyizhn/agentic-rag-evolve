#!/usr/bin/env python3
"""Build the deterministic input cohort for M5 hypothesis aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.planning import write_hypothesis_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a repair-eligible diagnosis cohort")
    parser.add_argument("--diagnosis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = write_hypothesis_cohort(args.diagnosis, args.output)
    print(json.dumps({"output": str(args.output), **cohort["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
