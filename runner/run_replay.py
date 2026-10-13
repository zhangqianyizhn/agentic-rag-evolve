"""Replay all evolution stages with full logs, reusing a verified old baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_rag_evolve.orchestration.baseline_reuse import validate_baseline_reuse
from agentic_rag_evolve.orchestration.experiment import ExperimentConfig, _read_json, run_experiment
from agentic_rag_evolve.orchestration.repairs import RepairConfig, run_repairs
from agentic_rag_evolve.providers import load_chat_model


def replay_configs(args) -> tuple[ExperimentConfig, RepairConfig]:
    original = args.baseline_experiment.resolve()
    output = args.output.resolve()
    repo = args.repo_root.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("replay output must be outside the source repository")
    if output == original or original in output.parents or output in original.parents:
        raise ValueError("replay output must not overlap the original experiment")
    old = _read_json(original / "experiment_manifest.json")["config"]
    experiment = ExperimentConfig(
        experiment_id=output.name, source_root=repo, env_file=args.env_file,
        documents=Path(old["documents"]), dataset=Path(old["dataset"]), store=Path(old["store"]),
        output=output / "experiment", mode="diagnose", judge=True,
        limit=old.get("limit"), task_ids=tuple(old.get("task_ids") or ()),
        max_rounds=old["max_rounds"], retrieval_topk=old["retrieval_topk"],
        diagnosis_max_rounds=args.diagnosis_max_rounds,
        diagnosis_max_tool_calls=args.diagnosis_max_tool_calls,
        diagnosis_max_output_tokens=args.diagnosis_max_output_tokens,
        request_timeout=args.request_timeout, request_max_retries=args.request_max_retries,
        reuse_baseline_experiment=original,
    )
    repairs = RepairConfig(
        experiment=experiment.output, output=output / "repairs", repo_root=repo,
        env_file=args.env_file, validation_config=args.validation_config,
        workers=args.workers, max_iterations=args.max_iterations,
        modification_max_rounds=args.modification_max_rounds,
        modification_max_tool_calls=args.modification_max_tool_calls,
        request_timeout=args.request_timeout, request_max_retries=args.request_max_retries,
    )
    return experiment, repairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New experiment root outside the repository")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--diagnosis-max-rounds", type=int, default=20)
    parser.add_argument("--diagnosis-max-tool-calls", type=int)
    parser.add_argument("--diagnosis-max-output-tokens", type=int)
    parser.add_argument("--modification-max-rounds", type=int, default=100)
    parser.add_argument("--modification-max-tool-calls", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment, repairs = replay_configs(args)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("replay output is non-empty; choose a new path or use --resume")
    # Preflight before creating any experiment checkpoint/worktree or calling APIs.
    provenance = validate_baseline_reuse(experiment)
    model = load_chat_model(args.env_file, timeout=args.request_timeout,
                            max_retries=args.request_max_retries)
    if args.validation_config is not None and not args.validation_config.is_file():
        raise FileNotFoundError(args.validation_config)
    if args.dry_run:
        report = {"status": "dry_run", "baseline": provenance,
                  "evolution_model": model.model_name,
                  "experiment": str(experiment.output), "repairs": str(repairs.output),
                  "stages": list(experiment.stages), "baseline_model_calls": 0,
                  "mode": "promotion" if args.validation_config else "screening",
                  "other_stages": "fresh evaluation, diagnosis, hypotheses, plans, all proceeding repairs and evaluations"}
        if (experiment.output / "experiment_state.json").is_file():
            if _read_json(experiment.output / "experiment_state.json").get("status") == "completed":
                report["repair_preflight"] = run_repairs(repairs, resume=args.resume, dry_run=True)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    progress = lambda message: print(f"[replay] {message}", file=sys.stderr, flush=True)
    run_experiment(experiment, resume=args.resume, progress=progress)
    result = run_repairs(repairs, resume=args.resume, progress=progress)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
