#!/usr/bin/env python3
"""Exercise the restricted readers exposed to a future diagnosis agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.diagnostics import DiagnosticArtifactReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Read an allowlisted diagnostic artifact")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-sources")
    list_parser.add_argument("--component")

    source_parser = subparsers.add_parser("read-source")
    source_parser.add_argument("--path", required=True)
    source_parser.add_argument("--start-line", type=int, default=1)
    source_parser.add_argument("--end-line", type=int, default=240)

    payload_parser = subparsers.add_parser("read-payload")
    payload_parser.add_argument("--path", required=True)
    payload_parser.add_argument("--offset-chars", type=int, default=0)
    payload_parser.add_argument("--limit-chars", type=int, default=12_000)

    args = parser.parse_args()
    reader = DiagnosticArtifactReader(bundle_path=args.bundle, source_root=args.source_root)
    if args.command == "list-sources":
        result = reader.list_sources(args.component)
    elif args.command == "read-source":
        result = reader.read_source(
            args.path,
            start_line=args.start_line,
            end_line=args.end_line,
        )
    else:
        result = reader.read_payload(
            args.path,
            offset_chars=args.offset_chars,
            limit_chars=args.limit_chars,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
