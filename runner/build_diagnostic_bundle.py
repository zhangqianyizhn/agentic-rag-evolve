#!/usr/bin/env python3
"""Build one validated DeepRead diagnosis-agent input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.diagnostics import build_diagnostic_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a DeepRead diagnostic input bundle")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_diagnostic_bundle(
        trajectory_path=args.trajectory,
        evaluation_path=args.evaluation,
        run_manifest_path=args.run_manifest,
        store_path=args.store,
        source_root=args.source_root,
        output_path=args.output,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
