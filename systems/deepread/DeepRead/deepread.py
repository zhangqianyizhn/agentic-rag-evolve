#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .agent.cli import add_ask_arguments, run_ask
from .index.ingest import add_parse_arguments, run_parse


def main() -> None:
    parser = argparse.ArgumentParser(prog="deepread", description="Parse documents and ask questions with DeepRead.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="convert PDF or Markdown into a DeepRead corpus")
    add_parse_arguments(parse_parser)
    parse_parser.set_defaults(func=run_parse)

    ask_parser = subparsers.add_parser("ask", help="ask questions over one or more DeepRead corpus files")
    add_ask_arguments(ask_parser)
    ask_parser.set_defaults(func=run_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
