"""Resume a planned experiment through isolated repairs and paired validation.

Sibling plans never share a mutable checkout. Only a promotion-gated winner may
become the source of the next round; screening is deliberately non-promoting.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from agentic_rag_evolve.deepread_runner import load_financebench_queries
from agentic_rag_evolve.evolution import (
    advance_baseline_registry,
    current_baseline,
    initialize_baseline_registry,
    materialize_accepted_candidate,
    record_candidate_outcome,
    verify_candidate_snapshot,
)
from agentic_rag_evolve.planning import (
    build_preservation_memory,
    build_repair_memory,
)
from agentic_rag_evolve.reporting import build_iteration_report
from agentic_rag_evolve.providers.config import llm_environment, provider_environment
from agentic_rag_evolve.store_build import verify_store_manifest
from agentic_rag_evolve.validation.gate import _pair_cohort

from .experiment import (
    ExperimentCheckpoint,
    ExperimentConfig,
    _path_digest,
    _document_manifest,
    _read_json,
    _sha256,
    _verify_frozen_diagnostic_sources,
    _write_json,
)


@dataclass(frozen=True)
class RepairConfig:
    experiment: Path
    output: Path
    repo_root: Path
    env_file: Path
    validation_config: Path | None = None
    base_revision: str | None = None
    workers: int = 1
    max_iterations: int = 1
    modification_max_rounds: int = 60
    modification_max_tool_calls: int = 120
    request_timeout: int = 1800
    request_max_retries: int = 3
    allow_model_change: bool = False

    def __post_init__(self) -> None:
        for name in ("experiment", "output", "repo_root", "env_file", "validation_config"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        if self.workers < 1 or self.max_iterations < 1:
            raise ValueError("workers and max_iterations must be positive")
        if self.modification_max_rounds < 1 or self.modification_max_tool_calls < 1:
            raise ValueError("modification budgets must be positive")
        if self.request_timeout < 1 or self.request_max_retries < 0:
            raise ValueError("invalid request timeout/retries")
        if self.max_iterations > 1 and self.validation_config is None:
            raise ValueError("multiple evolution rounds require promotion validation cohorts")
        if self.output == self.repo_root or self.repo_root in self.output.parents:
            raise ValueError("repair output/worktrees must be outside the source repository")
        if (self.output == self.experiment or self.output in self.experiment.parents
                or self.experiment in self.output.parents):
            raise ValueError("repair output must not overlap the input experiment")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def proceeding_plans(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != "deepread-modification-plan-v1":
        raise ValueError("unsupported modification plan schema")
    plans = list(document.get("plans") or [])
    ids = [str(plan.get("plan_id") or "") for plan in plans]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError("plan IDs must be unique safe path components")
    # Existing model order is retained; no new taxonomy or manual selector.
    return [plan for plan in plans if plan.get("decision") == "proceed"]


class RepairCheckpoint:
    """Independent stage attempts; partial outputs are retained, never overwritten."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "state.json"
        self.state = _read_json(self.path) if self.path.is_file() else {"stages": {}}
        self.verify_completed()

    def verify_completed(self) -> None:
        for record in self.state["stages"].values():
            if record.get("status") == "completed":
                for ref in record["artifacts"].values():
                    if _path_digest(Path(ref["path"])) != ref["sha256"]:
                        raise ValueError(f"completed repair artifact changed: {ref['path']}")

    def run(self, name: str, action: Callable[[Path], tuple[dict, dict]]) -> dict:
        previous = self.state["stages"].get(name, {})
        if previous.get("status") == "completed":
            for ref in previous["artifacts"].values():
                if _path_digest(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError(f"completed repair artifact changed: {ref['path']}")
            return previous["result"]
        attempt = int(previous.get("attempt", 0)) + 1
        directory = self.root / name / f"attempt-{attempt:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "attempt": attempt}
        self.state["stages"][name] = record
        _write_json(self.path, self.state)
        try:
            artifacts, result = action(directory)
            record.update(
                status="completed", result=result,
                artifacts={name: {"path": str(Path(path).resolve()),
                                  "sha256": _path_digest(Path(path))}
                           for name, path in artifacts.items()},
            )
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _write_json(self.path, self.state)
        return result


def _command(
    config: RepairConfig, directory: Path, module: str, arguments: list[str],
    *, cwd: Path | None = None, allowed_codes: tuple[int, ...] = (0,),
) -> dict:
    source = cwd or config.repo_root
    env = provider_environment(config.env_file)
    llm_models = {role: llm_environment(env, role)["MODEL"]
                  for role in ("deepread", "evolution")}
    # Providers/orchestration belong to the current stable framework; systems
    # belong to the candidate. Old frozen checkouts must also use dual-role config.
    env["PYTHONPATH"] = os.pathsep.join([str(config.repo_root / "src"), str(source),
                                      str(config.repo_root)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = directory / "stdout.log"
    stderr = directory / "stderr.log"
    with ExitStack() as stack:
        out = stack.enter_context(stdout.open("w"))
        err = stack.enter_context(stderr.open("w"))
        if module == "runner.create_candidate":
            git_lock = stack.enter_context((config.output / ".worktree.lock").open("a"))
            fcntl.flock(git_lock, fcntl.LOCK_EX)
        value = subprocess.run(
            [sys.executable, "-m", module, *map(str, arguments)],
            cwd=source, env=env, stdout=out, stderr=err,
        )
    _write_json(directory / "command.json", {
        "module": module, "arguments": list(map(str, arguments)),
        "cwd": str(source), "returncode": value.returncode,
        "llm_models": llm_models,
    })
    if value.returncode not in allowed_codes:
        raise RuntimeError(f"{module} failed ({value.returncode}); see {stderr}")
    lines = stdout.read_text().splitlines()
    return json.loads(lines[-1]) if lines else {}


def _cli_stage(
    checkpoint: RepairCheckpoint, config: RepairConfig, name: str, module: str,
    arguments: Callable[[Path], list], products: Callable[[Path, dict], dict],
    *, cwd: Path | None = None, allowed_codes: tuple[int, ...] = (0,),
) -> dict:
    def action(directory: Path):
        result = _command(config, directory, module, arguments(directory),
                          cwd=cwd, allowed_codes=allowed_codes)
        files = products(directory, result)
        files.update(command=directory / "command.json")
        return files, {**result, "paths": {k: str(v) for k, v in files.items()}}
    return checkpoint.run(name, action)


def _evaluation(
    checkpoint: RepairCheckpoint, config: RepairConfig, name: str,
    dataset: Path, predictions: Path,
) -> Path:
    result = _cli_stage(
        checkpoint, config, name, "runner.evaluate_deepread",
        lambda d: ["--dataset", dataset, "--predictions", predictions,
                   "--output", d / "evaluation", "--judge", "--env-file", config.env_file,
                   "--request-timeout", str(config.request_timeout),
                   "--request-max-retries", str(config.request_max_retries)],
        lambda d, _: {"evaluation": d / "evaluation"},
    )
    return Path(result["paths"]["evaluation"]) / "evaluation.json"


def _inference(
    checkpoint: RepairCheckpoint, config: RepairConfig, name: str, *,
    source: Path, dataset: Path, store: Path, task_ids: list[str],
    runtime_config: dict, audit: Path | None = None,
) -> Path:
    result = _cli_stage(
        checkpoint, config, name, "runner.run_deepread",
        lambda d: ["--dataset", dataset, "--store", store,
                   "--store-manifest", store / "STORE_MANIFEST.json",
                   "--output", d / "run", "--env-file", config.env_file,
                   "--max-rounds", str(runtime_config["max_rounds"]),
                   "--retrieval-topk", str(runtime_config["retrieval_topk"]),
                   *(["--candidate-audit", audit] if audit else []),
                   *[arg for task in task_ids for arg in ["--task-id", task]]],
        lambda d, _: {"run": d / "run"}, cwd=source,
    )
    return Path(result["paths"]["run"])


def _cohorts(config: RepairConfig, experiment: Path) -> list[dict]:
    manifest = _read_json(experiment / "experiment_manifest.json")["config"]
    task_ids = _read_json(experiment / "baseline" / "manifest.json")["task_ids"]
    development = {
        "name": "development", "dataset": "financebench", "role": "development",
        "dataset_path": manifest["dataset"], "documents": manifest["documents"],
        "store": manifest["store"], "task_ids": task_ids,
        "min_mean_delta": 0.0, "min_improved": 1, "max_regressions": 0,
        "max_token_cost_ratio": 2.0,
    }
    if config.validation_config is None:
        return [development]
    document = _read_json(config.validation_config)
    if document.get("schema_version") != "deepread-repair-validation-v1":
        raise ValueError("unsupported repair validation config schema")
    if set(document) - {"schema_version", "development_thresholds", "cohorts"}:
        raise ValueError("unknown repair validation config fields")
    development.update(document.get("development_thresholds") or {})
    # Only threshold overrides are allowed, never role/task changes.
    if set(document.get("development_thresholds") or {}) - {
        "min_mean_delta", "min_improved", "max_regressions", "max_token_cost_ratio"
    }:
        raise ValueError("development overrides must only contain thresholds")
    extras = list(document.get("cohorts") or [])
    names = {development["name"]}
    roles = set()
    for index, raw in enumerate(extras):
        if set(raw) - {"name", "dataset", "role", "dataset_path", "documents", "store", "task_ids",
                       "min_mean_delta", "min_improved", "max_regressions", "max_token_cost_ratio"}:
            raise ValueError("unknown validation cohort fields")
        if raw.get("role") not in {"holdout", "cross_dataset"}:
            raise ValueError("additional cohorts must be holdout or cross_dataset, never test")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(raw.get("name") or "")) or raw["name"] in names:
            raise ValueError("cohort names must be unique safe path components")
        names.add(raw["name"])
        roles.add(raw["role"])
        if raw["role"] == "holdout" and raw.get("dataset") != development["dataset"]:
            raise ValueError("holdout must belong to the development dataset")
        if raw["role"] == "cross_dataset" and raw.get("dataset") == development["dataset"]:
            raise ValueError("cross_dataset must be a different dataset")
        ids = list(raw.get("task_ids") or [])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("validation task IDs must be non-empty and unique")
        if raw.get("dataset") == development["dataset"] and set(ids) & set(task_ids):
            raise ValueError("already diagnosed experiment tasks cannot become holdout")
        value = {"min_mean_delta": 0.0, "min_improved": 0, "max_regressions": 0,
                 "max_token_cost_ratio": 2.0, **raw}
        if value["min_improved"] != 0:
            raise ValueError("min_improved is development-only")
        for field in ("dataset_path", "documents", "store"):
            path = Path(value[field])
            value[field] = str((config.validation_config.parent / path).resolve())
        extras[index] = value
    if roles != {"holdout", "cross_dataset"}:
        raise ValueError("promotion requires both holdout and cross_dataset cohorts")
    result = [development, *extras]
    for cohort in result:
        for key in ("min_mean_delta", "max_token_cost_ratio"):
            if not math.isfinite(float(cohort[key])):
                raise ValueError(f"non-finite threshold: {key}")
        if float(cohort["max_token_cost_ratio"]) <= 0:
            raise ValueError("max_token_cost_ratio must be positive")
        for key in ("min_improved", "max_regressions"):
            if not isinstance(cohort[key], int) or cohort[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
    return result


def _validate_experiment(experiment: Path) -> None:
    manifest = _read_json(experiment / "experiment_manifest.json")
    value = ExperimentConfig(**manifest["config"])
    state = _read_json(experiment / "experiment_state.json")
    if value.mode != "diagnose" or state.get("status") != "completed":
        raise ValueError("repair continuation requires a completed diagnose experiment")
    checkpoint = ExperimentCheckpoint(value, resume=True)
    for stage in value.stages:
        checkpoint._verify_completed(stage)


def _store_for(
    checkpoint: RepairCheckpoint, config: RepairConfig, cohort: dict,
    *, source: Path, rebuild: bool, audit: Path | None = None,
) -> Path:
    if not rebuild:
        store = Path(cohort["store"])
        manifest = verify_store_manifest(store, store / "STORE_MANIFEST.json")
        if manifest["document_manifest"]["sha256"] != _sha256(Path(cohort["documents"])):
            raise ValueError("baseline store belongs to different documents")
        for item in manifest["source_files"]:
            if _sha256(source / item["path"]) != item["sha256"]:
                raise ValueError("baseline store was built from different ingestion/parser code")
        return store
    result = _cli_stage(
        checkpoint, config, f"store-{cohort['name']}", "runner.build_deepread_store",
        lambda d: ["--documents", cohort["documents"], "--output", d / "store",
                   "--source-root", source, "--env-file", config.env_file,
                   *(["--candidate-audit", audit] if audit else [])],
        lambda d, _: {"store": d / "store"}, cwd=source,
    )
    return Path(result["paths"]["store"])


def _prepare_baselines(
    checkpoint: RepairCheckpoint, config: RepairConfig, cohorts: list[dict],
    experiment: Path, source: Path, runtime_config: dict,
) -> list[dict]:
    result = []
    for cohort in cohorts:
        value = dict(cohort)
        if cohort["role"] == "development":
            value["baseline_evaluation"] = str(experiment / "evaluation" / "evaluation.json")
        else:
            store = _store_for(checkpoint, config, cohort, source=source,
                               rebuild=not (Path(cohort["store"]) / "STORE_MANIFEST.json").is_file())
            run = _inference(checkpoint, config, f"baseline-{cohort['name']}",
                             source=source, dataset=Path(cohort["dataset_path"]),
                             store=store, task_ids=cohort["task_ids"], runtime_config=runtime_config)
            evaluation = _evaluation(checkpoint, config, f"baseline-eval-{cohort['name']}",
                                     Path(cohort["dataset_path"]), run / "predictions.jsonl")
            value["baseline_evaluation"] = str(evaluation)
            value["store"] = str(store)
        result.append(value)
    return result


def _suite_cohort(cohort: dict, evaluation: Path, run: Path) -> dict:
    return {
        **{key: cohort[key] for key in (
            "name", "dataset", "role", "baseline_evaluation", "task_ids",
            "min_mean_delta", "min_improved", "max_regressions", "max_token_cost_ratio",
        )},
        "candidate_evaluation": str(evaluation),
        "candidate_run_manifest": str(run / "manifest.json"),
        "primary_metric": "accuracy_normalized",
    }


def _run_plan(
    config: RepairConfig, round_root: Path, plan: dict, plan_path: Path,
    base_commit: str, cohorts: list[dict], runtime_config: dict,
    progress: Callable[[str], None],
) -> dict:
    root = round_root / "candidates" / plan["plan_id"]
    # A failed/interrupted editor may have mutated the checkout. Preserve it and
    # start a new isolated candidate instead of resetting or replaying edits.
    attempts = sorted(root.glob("attempt-*"))
    attempt = attempts[-1] if attempts else root / "attempt-0001"
    checkpoint = RepairCheckpoint(attempt)
    editing = checkpoint.state["stages"].get("modify", {})
    creating = checkpoint.state["stages"].get("create", {})
    if (editing and editing.get("status") != "completed") or (
        creating and creating.get("status") != "completed" and (attempt / "worktree").exists()
    ):
        attempt = root / f"attempt-{len(attempts) + 1:04d}"
        checkpoint = RepairCheckpoint(attempt)
    try:
        progress(f"plan={plan['plan_id']} stage=create")
        created = _cli_stage(
            checkpoint, config, "create", "runner.create_candidate",
            lambda d: ["--repo-root", config.repo_root, "--plan", plan_path,
                       "--test-policy", config.repo_root / "config/candidate-test-policy.json",
                       "--plan-id", plan["plan_id"], "--base-revision", base_commit,
                       "--candidate-path", attempt / "worktree", "--manifest", d / "candidate.json"],
            lambda d, _: {"manifest": d / "candidate.json"},
        )
        manifest = Path(created["paths"]["manifest"])
        source = Path(created["candidate_path"])
        progress(f"plan={plan['plan_id']} stage=modify")
        modified = _cli_stage(
            checkpoint, config, "modify", "runner.run_candidate_modification",
            lambda d: ["--manifest", manifest, "--plan", plan_path,
                       "--output", d / "modification.json", "--env-file", config.env_file,
                       "--request-timeout", str(config.request_timeout),
                       "--request-max-retries", str(config.request_max_retries),
                       "--max-rounds", str(config.modification_max_rounds),
                       "--max-tool-calls", str(config.modification_max_tool_calls)],
            lambda d, _: {"modification": d / "modification.json"},
        )
        modification = Path(modified["paths"]["modification"])
        audit = _cli_stage(
            checkpoint, config, "audit", "runner.audit_candidate",
            lambda d: ["--manifest", manifest, "--plan", plan_path,
                       "--modification", modification, "--output", d / "audit.json"],
            lambda d, _: {"audit": d / "audit.json"}, allowed_codes=(0, 1),
        )
        audit_path = Path(audit["paths"]["audit"])
        if not audit.get("passed"):
            return {"plan_id": plan["plan_id"], "status": "static_audit_failed"}
        verify_candidate_snapshot(source, audit)
        tests = _cli_stage(
            checkpoint, config, "tests", "runner.audit_candidate_tests",
            lambda d: ["--candidate-audit", audit_path,
                       "--policy", config.repo_root / "config/candidate-test-policy.json",
                       "--output", d / "tests.json"],
            lambda d, _: {"tests": d / "tests.json"}, allowed_codes=(0, 1),
        )
        if not tests.get("passed"):
            return {"plan_id": plan["plan_id"], "status": "fixed_tests_failed"}
        rebuild = bool(plan["edit_scope"].get("requires_store_rebuild"))
        specs = []
        for cohort in cohorts:
            progress(f"plan={plan['plan_id']} stage=run cohort={cohort['name']}")
            store = _store_for(checkpoint, config, cohort, source=source,
                               rebuild=rebuild, audit=audit_path)
            run = _inference(checkpoint, config, f"run-{cohort['name']}", source=source,
                             dataset=Path(cohort["dataset_path"]), store=store,
                             task_ids=cohort["task_ids"], runtime_config=runtime_config,
                             audit=audit_path)
            evaluation = _evaluation(checkpoint, config, f"eval-{cohort['name']}",
                                     Path(cohort["dataset_path"]), run / "predictions.jsonl")
            specs.append(_suite_cohort(cohort, evaluation, run))
        suite_path = attempt / "validation-suite.json"
        _write_json(suite_path, {
            "schema_version": "deepread-validation-suite-v1", "candidate_id": audit["candidate_id"],
            "plan_id": plan["plan_id"], "candidate_snapshot_sha256": audit["candidate_snapshot_sha256"],
            "gate_level": "promotion", "comparison_epsilon": 1e-9, "cohorts": specs,
        })
        if config.validation_config is None:
            comparison = _pair_cohort(
                specs[0], suite_root=attempt, epsilon=1e-9,
                requires_store_rebuild=rebuild,
                candidate_snapshot_sha256=audit["candidate_snapshot_sha256"],
                candidate_audit_sha256=_sha256(audit_path),
            )
            _write_json(attempt / "screening.json", {
                "schema_version": "deepread-repair-screening-v1",
                "promotion_performed": False, "cohort": comparison,
            })
            report = {"plan_id": plan["plan_id"], "status": "screened",
                      "comparison": str(attempt / "screening.json"), "cohorts": [comparison]}
        else:
            gate = _cli_stage(
                checkpoint, config, "gate", "runner.check_validation_gate",
                lambda d: ["--suite", suite_path, "--candidate-audit", audit_path,
                           "--candidate-test-audit", tests["paths"]["tests"],
                           "--plan", plan_path, "--output", d / "gate.json"],
                lambda d, _: {"gate": d / "gate.json", "suite": suite_path}, allowed_codes=(0, 1),
            )
            outcome = _outcome_stage(
                checkpoint, config, candidate_manifest_path=manifest, plan_path=plan_path,
                candidate_modification_path=modification, candidate_audit_path=audit_path,
                candidate_test_audit_path=Path(tests["paths"]["tests"]),
                validation_suite_path=suite_path, validation_gate_path=Path(gate["paths"]["gate"]),
            )
            report = {"plan_id": plan["plan_id"], "status": outcome["record"]["outcome"],
                      "outcome": outcome["record_path"], "cohorts": gate["cohorts"]}
        # Summary contains aggregates only; sealed task-level comparisons stay on disk.
        report["cohorts"] = [_cohort_summary(c) for c in report["cohorts"]]
        _write_json(attempt / "report.json", report)
        return report
    except Exception as exc:
        report = {"plan_id": plan["plan_id"], "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
        _write_json(attempt / "failure.json", report)
        return report


def _cohort_summary(cohort: dict) -> dict:
    pairs = cohort["pairs"]
    return {
        **{key: cohort[key] for key in (
            "name", "role", "task_count", "mean_delta", "baseline_answer_tokens",
            "candidate_answer_tokens", "token_cost_ratio", "passed",
        )},
        "improved_count": len(cohort["improved_task_ids"]),
        "regression_count": len(cohort["regressed_task_ids"]),
        "baseline_mean": sum(p["baseline"] for p in pairs) / len(pairs),
        "candidate_mean": sum(p["candidate"] for p in pairs) / len(pairs),
        "baseline_full_score_count": sum(p["baseline"] == 1.0 for p in pairs),
        "candidate_full_score_count": sum(p["candidate"] == 1.0 for p in pairs),
    }


def _outcome_stage(checkpoint: RepairCheckpoint, config: RepairConfig, **inputs) -> dict:
    def action(directory: Path):
        # Registry writes are serial even when candidate investigation is parallel.
        with (config.output / ".outcomes.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            audit = _read_json(inputs["candidate_audit_path"])
            matches = [path for path in (config.output / "outcomes").glob("*/*.json")
                       if (record := _read_json(path)).get("candidate_id") == audit["candidate_id"]
                       and record.get("candidate_snapshot_sha256") == audit["candidate_snapshot_sha256"]]
            if matches:
                if len(matches) != 1:
                    raise ValueError("ambiguous existing candidate outcome")
                path = matches[0]
                record = _read_json(path)
                for name, ref in record["artifacts"].items():
                    # Map outcome artifact names to the public registry function's args.
                    field = "plan_path" if name == "modification_plan" else name + "_path"
                    if Path(ref["path"]) != inputs[field] or ref["sha256"] != _sha256(inputs[field]):
                        raise ValueError("existing outcome lineage changed")
            else:
                record, path = record_candidate_outcome(registry_root=config.output / "outcomes", **inputs)
        return {"outcome": path}, {"record_path": str(path), "record": record}
    return checkpoint.run("outcome", action)


def _experiment_stage(checkpoint: RepairCheckpoint, config: RepairConfig,
                      name: str, arguments: list, source: Path) -> dict:
    """Resume the existing experiment checkpoint instead of duplicating API work."""
    previous = checkpoint.state["stages"].get(name, {})
    previous_root = Path(previous.get("experiment_path") or (
        checkpoint.root / name / f"attempt-{previous.get('attempt', 0):04d}" / "experiment"
    ))
    def action(directory: Path):
        output = previous_root if (previous_root / "experiment_manifest.json").is_file() else directory / "experiment"
        checkpoint.state["stages"][name]["experiment_path"] = str(output)
        _write_json(checkpoint.path, checkpoint.state)
        result = _command(config, directory, "runner.run_experiment",
                          [*arguments, "--output", output,
                           *(["--resume"] if (output / "experiment_manifest.json").is_file() else [])],
                          cwd=source)
        return {"experiment": output, "command": directory / "command.json"}, {
            **result, "paths": {"experiment": str(output)},
        }
    return checkpoint.run(name, action)


def select_winner(reports: list[dict]) -> dict | None:
    """Select only promotion-eligible candidates, using aggregate sealed metrics."""
    eligible = [r for r in reports if r["status"] == "accepted"]
    def score(report: dict):
        sealed = [c for c in report["cohorts"] if c["role"] != "development"]
        mean = sum(c["mean_delta"] * c["task_count"] for c in sealed) / sum(c["task_count"] for c in sealed)
        development = next(c["mean_delta"] for c in report["cohorts"] if c["role"] == "development")
        return (-mean, -development, report["plan_id"])
    return min(eligible, key=score) if eligible else None


def _verify_repair_resume(config: RepairConfig, description: dict, frozen: dict) -> None:
    existing_rounds = [int(path.name.removeprefix("round-"))
                       for path in config.output.glob("round-*")
                       if re.fullmatch(r"round-\d+", path.name)]
    if config.max_iterations < max([frozen["config"]["max_iterations"], *existing_rounds]):
        raise ValueError("max_iterations cannot be reduced below existing repair rounds/budget")
    # Execution policy/framework models may change for pending stages.
    # Completed stage artifacts, target models and inputs remain frozen.
    for key in ("workers", "max_iterations", "modification_max_rounds",
                "modification_max_tool_calls", "request_timeout", "request_max_retries"):
        frozen["config"][key] = description["config"][key]
    frozen["evolution_model"] = description["evolution_model"]
    # Legacy dotenv_values recorded absent optional settings as "" instead of null.
    frozen["models"] = {key: value or None for key, value in frozen["models"].items()}
    if frozen != description:
        raise ValueError("repair config/frozen inputs/models changed since start")


def run_repairs(
    config: RepairConfig, *, resume: bool = False, dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    _validate_experiment(config.experiment)
    initial = _read_json(config.experiment / "experiment_manifest.json")["config"]
    _verify_frozen_diagnostic_sources(config.experiment, config.repo_root)
    frozen_manifest = config.output / "manifest.json"
    revision = config.base_revision or (
        _read_json(frozen_manifest)["base_commit"] if resume and frozen_manifest.is_file() else "HEAD"
    )
    base = _git(config.repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    # The chosen commit must contain the target that generated the frozen evidence.
    catalog = _read_json(config.experiment / "diagnostic_bundles" / "index.json")["items"]
    from agentic_rag_evolve.diagnostics import DiagnosticArtifactReader
    reader = DiagnosticArtifactReader(bundle_path=Path(catalog[0]["bundle"]), source_root=config.repo_root)
    for source in reader.list_sources():
        content = subprocess.run(["git", "-C", str(config.repo_root), "show", f"{base}:{source['path']}"],
                                 check=True, capture_output=True).stdout
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("base commit target differs from frozen diagnostic sources")
    cohorts = _cohorts(config, config.experiment)
    used_keys = set()
    for cohort in cohorts:
        queries = {q.task_id: q for q in load_financebench_queries(Path(cohort["dataset_path"]))}
        documents = {d["document_id"] for d in _document_manifest(Path(cohort["documents"]))}
        if set(cohort["task_ids"]) - set(queries):
            raise ValueError(f"unknown validation task IDs in {cohort['name']}")
        if {queries[t].sample_id for t in cohort["task_ids"]} - documents:
            raise ValueError(f"missing documents for cohort {cohort['name']}")
        keys = {(cohort["dataset"], task) for task in cohort["task_ids"]}
        if used_keys & keys:
            raise ValueError("validation cohorts must be pairwise task-disjoint")
        used_keys |= keys
    plan_path = config.experiment / "planning" / "plan.json"
    plans = proceeding_plans(_read_json(plan_path))
    env = provider_environment(config.env_file)
    target_llm = llm_environment(env, "deepread")
    evolution_llm = llm_environment(env, "evolution")
    baseline_models = _read_json(config.experiment / "baseline/manifest.json")["providers"]
    current_models = {"chat": target_llm["MODEL"] or None,
                      "embedding": env.get("EMBEDDING_MODEL_NAME") or None,
                      "reranker": env.get("RERANK_MODEL") or None}
    model_changed = any(current_models[name] != model for name, model in baseline_models.items())
    if model_changed and not config.allow_model_change:
        raise ValueError("provider models differ from baseline; rerun control or explicitly use --allow-model-change")
    description = {
        "schema_version": "deepread-repair-run-v1",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "base_commit": base, "plan_ids": [p["plan_id"] for p in plans],
        # Keep the historical target key so old runs can resume without migration.
        "models": {"LLM_MODEL": target_llm["MODEL"] or None,
                   **{key: env.get(key) or None for key in (
                       "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION", "RERANK_MODEL")}},
        "evolution_model": evolution_llm["MODEL"] or None,
        "model_changed_from_control": model_changed,
        "control_models": baseline_models,
        "inputs": {str(path): _sha256(path) for path in (
            config.experiment / "experiment_state.json", plan_path,
            config.repo_root / "config/candidate-test-policy.json",
            *([config.validation_config] if config.validation_config else []),
            *[Path(c[field]) for c in cohorts for field in ("dataset_path", "documents")],
            *[Path(document["path"]) for c in cohorts
              for document in _document_manifest(Path(c["documents"]))],
            *[Path(c["store"]) / "STORE_MANIFEST.json" for c in cohorts
              if (Path(c["store"]) / "STORE_MANIFEST.json").is_file()],
        )},
        "cohort_counts": [{"name": c["name"], "role": c["role"], "count": len(c["task_ids"])} for c in cohorts],
        "final_test_accessed": False,
    }
    if dry_run:
        if resume and frozen_manifest.is_file():
            _verify_repair_resume(config, description, _read_json(frozen_manifest))
        return {**description, "status": "dry_run", "writes_performed": False}
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = config.output / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("repair run exists; use --resume")
            _verify_repair_resume(config, description, _read_json(manifest))
        else:
            if any(p.name != ".run.lock" for p in config.output.iterdir()):
                raise FileExistsError("repair output is non-empty without a manifest")
            _write_json(manifest, description)
        # Preserve the initial manifest; record each invocation's effective policy
        # separately, including upgrades from legacy single-model manifests.
        sessions = config.output / "execution"
        sessions.mkdir(exist_ok=True)
        session = sessions / f"attempt-{len(list(sessions.glob('attempt-*.json'))) + 1:04d}.json"
        _write_json(session, {"resume": resume, "config": description["config"],
                              "models": description["models"],
                              "evolution_model": description["evolution_model"]})
        return _run_rounds(config, description, initial, cohorts, progress)


def _run_rounds(config: RepairConfig, description: dict, initial: dict,
                cohorts: list[dict], progress: Callable[[str], None]) -> dict:
    checkpoint = RepairCheckpoint(config.output / "control")
    registry = config.output / "baselines"
    if config.validation_config is not None and not (registry / "entries").exists():
        initialize_baseline_registry(registry_root=registry, repo_root=config.repo_root,
                                     revision=description["base_commit"])
    source = config.repo_root
    base = description["base_commit"]
    experiment = config.experiment
    rounds = []
    for number in range(1, config.max_iterations + 1):
        root = config.output / f"round-{number:04d}"
        round_checkpoint = RepairCheckpoint(root / "control")
        done = round_checkpoint.state["stages"].get("terminal", {})
        if done.get("status") == "completed":
            summary = round_checkpoint.run("terminal", lambda _: ({}, {}))
        else:
            runtime = {key: initial[key] for key in ("max_rounds", "retrieval_topk")}
            if number == 1:
                # Run controls from the chosen clean commit, never a dirty main checkout.
                baseline = _cli_stage(
                    checkpoint, config, "baseline-checkout", "runner.create_baseline_checkout",
                    lambda d: ["--repo-root", config.repo_root, "--revision", base,
                               "--output", d / "worktree", "--manifest", d / "checkout.json"],
                    lambda d, _: {"manifest": d / "checkout.json"},
                )
                source = Path(baseline["source_root"])
            plan_path = experiment / "planning/plan.json"
            if number > 1:
                memory = _cli_stage(
                    round_checkpoint, config, "planning-memory", "runner.build_planning_memory",
                    lambda d: ["--memory-root", config.output / "memory",
                               "--cohort", experiment / "planning/cohort.json",
                               "--hypotheses", experiment / "planning/hypotheses.json",
                               "--output", d / "memory.json"],
                    lambda d, _: {"memory": d / "memory.json"},
                )
                planned = _cli_stage(
                    round_checkpoint, config, "replan", "runner.run_modification_planning",
                    lambda d: ["--cohort", experiment / "planning/cohort.json",
                               "--hypotheses", experiment / "planning/hypotheses.json",
                               "--source-root", source, "--memory-context", memory["paths"]["memory"],
                               "--env-file", config.env_file, "--output", d / "plan.json",
                               "--request-timeout", str(config.request_timeout),
                               "--request-max-retries", str(config.request_max_retries)],
                    lambda d, _: {"plan": d / "plan.json"}, cwd=source,
                )
                plan_path = Path(planned["paths"]["plan"])
            plans = proceeding_plans(_read_json(plan_path))
            baseline_checkpoint = RepairCheckpoint(config.output / "controls" / base)
            prepared = _prepare_baselines(baseline_checkpoint, config, cohorts, experiment, source, runtime) if plans else []
            progress(f"round={number} proceeding_plans={len(plans)} workers={config.workers}")
            def candidates_action(directory: Path):
                with ThreadPoolExecutor(max_workers=config.workers) as pool:
                    futures = [pool.submit(_run_plan, config, root, plan, plan_path,
                                           base, prepared, runtime, progress) for plan in plans]
                    reports = [future.result() for future in futures]
                _write_json(directory / "results.json", reports)
                if any(r["status"] == "failed" for r in reports):
                    raise RuntimeError(f"candidate stage failed; see {directory / 'results.json'}")
                return {"results": directory / "results.json"}, {"candidates": reports}
            try:
                reports = round_checkpoint.run("candidates", candidates_action)["candidates"]
            except RuntimeError as exc:
                stage = round_checkpoint.state["stages"]["candidates"]
                results_path = root / "control/candidates" / f"attempt-{stage['attempt']:04d}" / "results.json"
                reports = _read_json(results_path) if results_path.is_file() else []
                summary = {"round": number, "status": "failed", "candidates": reports,
                           "error": str(exc)}
                _write_json(root / "report.json", summary)
                result = {"status": "failed", "rounds": [*rounds, summary], "resume_required": True}
                _write_json(config.output / "report.json", result)
                return result
            if any(r["status"] == "failed" for r in reports):
                summary = {"round": number, "status": "failed", "candidates": reports}
                _write_json(root / "report.json", summary)
                result = {"status": "failed", "rounds": [*rounds, summary], "resume_required": True}
                _write_json(config.output / "report.json", result)
                return result
            winner = select_winner(reports)
            lifecycle = _finish_round(round_checkpoint, config, reports, winner, experiment)
            summary = {"round": number, "status": "completed" if plans else "no_candidate", "candidates": reports,
                       "selected_plan_id": winner["plan_id"] if winner else None,
                       "promotion_performed": winner is not None, **lifecycle}
            _write_json(root / "report.json", summary)
            round_checkpoint.run("terminal", lambda _: ({"report": root / "report.json"}, summary))
        rounds.append(summary)
        if config.validation_config is None or not summary["candidates"]:
            break
        if summary.get("promotion_performed"):
            base = summary["next_commit"]
            source = Path(summary["next_source"])
            cohorts = [{**c, "store": str(root / "next-stores" / c["name"])} for c in cohorts]
        if number < config.max_iterations and summary.get("promotion_performed"):
            fresh = _experiment_stage(
                checkpoint, config, f"diagnose-round-{number + 1}",
                ["--experiment-id", f"repair-round-{number + 1}", "--mode", "diagnose",
                           "--documents", initial["documents"], "--dataset", initial["dataset"],
                           "--store", cohorts[0]["store"],
                           "--source-root", source, "--env-file", config.env_file,
                           "--max-rounds", str(initial["max_rounds"]),
                           "--retrieval-topk", str(initial["retrieval_topk"]),
                           "--request-timeout", str(config.request_timeout),
                           "--request-max-retries", str(config.request_max_retries),
                           *[arg for task in cohorts[0]["task_ids"] for arg in ["--task-id", task]]],
                source,
            )
            experiment = Path(fresh["paths"]["experiment"])
        # Rejected rounds keep baseline/evidence, but replan with newly saved memory.
    result = {"schema_version": "deepread-repair-report-v1", "status": "completed",
              "mode": "promotion" if config.validation_config else "screening",
              "rounds": rounds, "workers": config.workers,
              "model_changed_from_control": description.get("model_changed_from_control", False),
              "final_test_accessed": False}
    _write_json(config.output / "report.json", result)
    return result


def _finish_round(checkpoint: RepairCheckpoint, config: RepairConfig,
                  reports: list[dict], winner: dict | None, experiment: Path) -> dict:
    for report in reports:
        if report["status"] != "rejected":
            continue
        def rejected_action(directory: Path, report=report):
            memory = _memory(build_repair_memory, outcome_path=Path(report["outcome"]),
                             memory_root=config.output / "memory")
            result = build_iteration_report(
                outcome_path=Path(report["outcome"]), terminal_memory_path=memory,
                cohort_path=experiment / "planning/cohort.json",
                hypotheses_path=experiment / "planning/hypotheses.json",
                output_path=directory / "iteration-report.json",
            )
            return {"report": directory / "iteration-report.json", "memory": memory}, result
        checkpoint.run(f"rejected-{report['plan_id']}", rejected_action)
    if winner is None:
        return {}
    def promote(directory: Path):
        materialization = materialize_accepted_candidate(outcome_path=Path(winner["outcome"]))
        path = directory / "materialization.json"
        _write_json(path, materialization)
        return {"materialization": path}, materialization
    materialization = checkpoint.run("materialize", promote)
    path = Path(checkpoint.state["stages"]["materialize"]["artifacts"]["materialization"]["path"])
    def advance(directory: Path):
        current = current_baseline(config.output / "baselines")
        if current.get("materialization_id") == materialization["materialization_id"]:
            entries = sorted((config.output / "baselines/entries").glob("*.json"))
            entry, entry_path = current, entries[-1]
        else:
            entry, entry_path = advance_baseline_registry(registry_root=config.output / "baselines",
                                                         materialization_path=path)
        memory = _memory(build_preservation_memory,
            outcome_path=Path(winner["outcome"]), materialization_path=path,
            baseline_entry_path=entry_path, memory_root=config.output / "memory",
        )
        build_iteration_report(
            outcome_path=Path(winner["outcome"]), terminal_memory_path=memory,
            cohort_path=experiment / "planning/cohort.json",
            hypotheses_path=experiment / "planning/hypotheses.json",
            output_path=directory / "iteration-report.json",
        )
        return {"baseline": entry_path, "memory": memory,
                "report": directory / "iteration-report.json"}, entry
    checkpoint.run("advance", advance)
    outcome = _read_json(Path(winner["outcome"]))
    return {"next_commit": materialization["materialized_commit"],
            "next_source": outcome["candidate_path"]}


def _memory(builder: Callable, **kwargs) -> Path:
    """An interrupted terminal stage may already have written immutable memory."""
    try:
        _, path = builder(**kwargs)
        return path
    except FileExistsError as exc:
        if not exc.filename:
            raise
        path = Path(exc.filename)
        record = _read_json(path)
        outcome = Path(kwargs["outcome_path"])
        reference = record["artifacts"]["candidate_outcome"]
        if Path(reference["path"]) != outcome or reference["sha256"] != _sha256(outcome):
            raise ValueError("existing memory does not match the frozen outcome") from exc
        # The downstream iteration report also revalidates the memory derivation.
        return path
