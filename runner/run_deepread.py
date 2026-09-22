#!/usr/bin/env python3
"""Framework-owned DeepRead benchmark entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.deepread_runner import run_financebench
from agentic_rag_evolve.providers import load_provider_bundle
from systems.deepread.runtime import DeepReadConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen DeepRead global baseline")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--store-manifest",
        type=Path,
        help="Verify and bind a STORE_MANIFEST.json produced by build_deepread_store",
    )
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        help="Bind this run to an audited candidate checkout",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--task-id",
        action="append",
        help="Run only this exact dataset task ID; repeat to preserve a chosen order",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    args = parser.parse_args()

    summary = run_financebench(
        dataset_path=args.dataset,
        store_path=args.store,
        output_path=args.output,
        providers=load_provider_bundle(args.env_file),
        config=DeepReadConfig(
            max_rounds=args.max_rounds,
            retrieval_topk=args.retrieval_topk,
        ),
        limit=args.limit,
        task_ids=args.task_id,
        store_manifest_path=args.store_manifest,
        candidate_audit_path=args.candidate_audit,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
