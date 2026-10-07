"""Continue all proceeding plans through repair, evaluation and optional evolution."""

import argparse
import json
import sys
from pathlib import Path

from agentic_rag_evolve.orchestration.repairs import RepairConfig, run_repairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--base-revision")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--allow-model-change", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = RepairConfig(**{k: v for k, v in vars(args).items() if k not in {"resume", "dry_run"}})
    result = run_repairs(config, resume=args.resume, dry_run=args.dry_run,
                         progress=lambda s: print(f"[repairs] {s}", file=sys.stderr, flush=True))
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
