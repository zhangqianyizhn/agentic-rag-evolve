#!/usr/bin/env python3
"""Initialize, inspect, or advance an append-only evolution iteration ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.orchestration import (
    append_iteration_event,
    initialize_iteration,
    read_iteration_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage one recoverable evolution iteration")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--iteration-id", required=True)
    init.add_argument("--baseline-id", required=True)
    init.add_argument("--baseline-commit", required=True)
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    record = commands.add_parser("record")
    record.add_argument("--root", type=Path, required=True)
    record.add_argument("--step", required=True)
    record.add_argument("--status", choices=("completed", "failed"), required=True)
    record.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    record.add_argument("--error")
    args = parser.parse_args()
    if args.command == "init":
        result = initialize_iteration(
            root=args.root,
            iteration_id=args.iteration_id,
            baseline_id=args.baseline_id,
            baseline_commit=args.baseline_commit,
        )
    elif args.command == "status":
        result = read_iteration_status(root=args.root)
    else:
        artifacts = {}
        for item in args.artifact:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("artifact must use NAME=PATH")
            artifacts[name] = Path(path)
        result = append_iteration_event(
            root=args.root,
            step=args.step,
            status=args.status,
            artifacts=artifacts,
            error=args.error,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
