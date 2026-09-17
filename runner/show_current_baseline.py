#!/usr/bin/env python3
"""Show the validated tip of the baseline registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import current_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the current DeepRead baseline")
    parser.add_argument("--registry-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(current_baseline(args.registry_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
