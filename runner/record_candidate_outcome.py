#!/usr/bin/env python3
"""Append one accepted/rejected candidate outcome to the experiment registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_evolve.evolution import record_candidate_outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a terminal candidate outcome")
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-modification", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--candidate-test-audit", type=Path, required=True)
    parser.add_argument("--validation-suite", type=Path, required=True)
    parser.add_argument("--validation-gate", type=Path, required=True)
    args = parser.parse_args()
    record, path = record_candidate_outcome(
        registry_root=args.registry_root,
        candidate_manifest_path=args.candidate_manifest,
        plan_path=args.plan,
        candidate_modification_path=args.candidate_modification,
        candidate_audit_path=args.candidate_audit,
        candidate_test_audit_path=args.candidate_test_audit,
        validation_suite_path=args.validation_suite,
        validation_gate_path=args.validation_gate,
    )
    print(json.dumps({"record_path": str(path), "record": record}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
