#!/usr/bin/env python3
"""Build a compact report from one or more DeepRead diagnostic bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.diagnostics import build_bad_case_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a DeepRead bad-case report")
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        action="append",
        help="Bundle JSON file or its containing directory; repeat for multiple tasks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_bad_case_report(bundle_paths=args.bundle, output_path=args.output)
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
