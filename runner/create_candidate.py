#!/usr/bin/env python3
"""Create a detached candidate worktree for one proceeding plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import create_candidate_worktree


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an isolated DeepRead candidate")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_candidate_worktree(
        repo_root=args.repo_root,
        plan_path=args.plan,
        plan_id=args.plan_id,
        base_revision=args.base_revision,
        candidate_path=args.candidate_path,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
