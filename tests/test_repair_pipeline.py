import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentic_rag_evolve.evolution import (
    audit_candidate, audit_candidate_tests, create_candidate_worktree,
    run_candidate_modification,
)
from agentic_rag_evolve.orchestration.repairs import (
    RepairCheckpoint, RepairConfig, _cohorts, _command, _run_rounds,
    proceeding_plans, select_winner, run_repairs,
)
from agentic_rag_evolve.diagnostics.policy import DIAGNOSTIC_SOURCE_POLICY
from agentic_rag_evolve.orchestration.experiment import ExperimentConfig
from agentic_rag_evolve.validation import evaluate_validation_gate


SOURCE = "systems/deepread/DeepRead/prompt/system.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EditingModel:
    model_name = "fixture"

    def __init__(self, old, new):
        self.calls = iter([
            ("read_source", {"path": SOURCE, "start_line": 1, "end_line": 10}),
            ("replace_text", {"path": SOURCE, "old_text": old, "new_text": new}),
            ("show_diff", {}),
            ("submit_modification", {"summary": "verify", "implemented_behavior_delta": "verify",
                                     "preservation_notes": [], "uncertainties": []}),
        ])

    def complete(self, payload):
        name, arguments = next(self.calls)
        return {"choices": [{"message": {"tool_calls": [{"id": name, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


class FixtureCommands:
    """Real Git/editor/audits/gates; only model/inference outputs are controlled."""

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.holdout_regress = False

    def __call__(self, config, directory, module, arguments, *, cwd=None, allowed_codes=(0,)):
        arguments = list(map(str, arguments))
        args = dict(zip(arguments[::2], arguments[1::2]))
        self.calls.append(module)
        write(directory / "command.json", {"module": module})
        if module == "runner.create_baseline_checkout":
            subprocess.run(["git", "-C", str(config.repo_root), "worktree", "add", "--detach",
                            args["--output"], args["--revision"]], check=True, capture_output=True)
            result = {"source_root": args["--output"], "commit": args["--revision"]}
            write(Path(args["--manifest"]), result)
            return result
        if module == "runner.create_candidate":
            return create_candidate_worktree(
                repo_root=Path(args["--repo-root"]), plan_path=Path(args["--plan"]),
                test_policy_path=Path(args["--test-policy"]), plan_id=args["--plan-id"],
                base_revision=args["--base-revision"], candidate_path=Path(args["--candidate-path"]),
                manifest_path=Path(args["--manifest"]),
            )
        if module == "runner.run_candidate_modification":
            manifest = json.loads(Path(args["--manifest"]).read_text())
            old = (Path(manifest["candidate_path"]) / SOURCE).read_text()
            new = "VALUE = 2\n" if manifest["plan_id"].startswith("plan-1") else "VALUE = 3\n"
            if old == new:
                new += "# verified\n"
            return run_candidate_modification(
                manifest_path=Path(args["--manifest"]), plan_path=Path(args["--plan"]),
                output_path=Path(args["--output"]), model=EditingModel(old, new),
            ).to_dict()
        if module == "runner.audit_candidate":
            result = audit_candidate(manifest_path=Path(args["--manifest"]), plan_path=Path(args["--plan"]),
                                     modification_path=Path(args["--modification"]))
            write(Path(args["--output"]), result)
            return result
        if module == "runner.audit_candidate_tests":
            result = audit_candidate_tests(candidate_audit_path=Path(args["--candidate-audit"]),
                                           policy_path=Path(args["--policy"]), python_executable=Path(sys.executable))
            write(Path(args["--output"]), result)
            return result
        if module == "runner.run_deepread":
            output = Path(args["--output"])
            ids = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "--task-id"]
            rows = [json.loads(line) for line in Path(args["--dataset"]).read_text().splitlines()]
            code = (cwd / SOURCE).read_text()
            score = 0.25 if "VALUE = 1" in code else 1.0 if "VALUE = 2" in code else 0.75
            records = [{"task_id": row["financebench_id"], "question": row["question"],
                        "sample_id": row["doc_name"], "run_id": str(output),
                        "prediction": {"status": "ok"}, "token_usage": {"input_tokens": 90, "output_tokens": 10},
                        "metrics": {"accuracy_normalized": score}} for row in rows if row["financebench_id"] in ids]
            output.mkdir(parents=True)
            (output / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            audit = json.loads(Path(args["--candidate-audit"]).read_text()) if "--candidate-audit" in args else {}
            if self.holdout_regress and audit and "held" in ids:
                for record in records:
                    record["metrics"]["accuracy_normalized"] = 0.0
                (output / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            write(output / "manifest.json", {"schema_version": 1, "run_id": str(output),
                                              "task_ids": ids, "candidate_source": audit})
            return {"failed": 0}
        if module == "runner.evaluate_deepread":
            items = [json.loads(line) for line in Path(args["--predictions"]).read_text().splitlines()]
            output = Path(args["--output"])
            write(output / "evaluation.json", items)
            write(output / "evaluation_summary.json", {"total": len(items)})
            return {"total": len(items)}
        if module == "runner.check_validation_gate":
            result = evaluate_validation_gate(
                suite_path=Path(args["--suite"]), candidate_audit_path=Path(args["--candidate-audit"]),
                candidate_test_audit_path=Path(args["--candidate-test-audit"]), plan_path=Path(args["--plan"]),
            )
            write(Path(args["--output"]), result)
            return result
        if module == "runner.build_planning_memory":
            write(Path(args["--output"]), {})
            return {}
        if module == "runner.build_deepread_store":
            store = Path(args["--output"])
            write(store / "corpus.json", {})
            manifest = {"schema_version": "deepread-store-build-v1",
                        "document_manifest": {"sha256": sha(Path(args["--documents"]))},
                        "source_files": [{"path": "systems/deepread/ingestion.py",
                                          "sha256": sha(cwd / "systems/deepread/ingestion.py")}],
                        "artifacts": [{"path": "corpus.json", "sha256": sha(store / "corpus.json")}]}
            write(store / "STORE_MANIFEST.json", manifest)
            return manifest
        if module == "runner.run_experiment":
            output = Path(args["--output"])
            initial = json.loads((self.config.experiment / "experiment_manifest.json").read_text())
            initial["config"]["store"] = args["--store"]
            write(output / "experiment_manifest.json", initial)
            ids = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "--task-id"]
            write(output / "baseline/manifest.json", {"task_ids": ids})
            old_eval = json.loads((self.config.experiment / "evaluation/evaluation.json").read_text())
            for item in old_eval:
                item["metrics"]["accuracy_normalized"] = 1.0
            write(output / "evaluation/evaluation.json", old_eval)
            for name in ["cohort.json", "hypotheses.json", "plan.json"]:
                write(output / "planning" / name, json.loads((self.config.experiment / "planning" / name).read_text()))
            store = Path(args["--store"])
            self(config, directory, "runner.build_deepread_store",
                 ["--documents", args["--documents"], "--output", str(store)], cwd=cwd)
            return {"status": "completed"}
        if module == "runner.run_modification_planning":
            plan = json.loads((self.config.experiment / "planning/plan.json").read_text())
            for item in plan["plans"]:
                item["change_contract"]["required_behavior_delta"] += " with memory"
                item["plan_id"] += "-retry"
            write(Path(args["--output"]), plan)
            return {"status": "ok"}
        raise AssertionError(module)


class RepairPipelineTest(unittest.TestCase):
    def fixture(self, root, promotion=False, iterations=1):
        repo = root / "repo"
        repo.mkdir()
        (repo / SOURCE).parent.mkdir(parents=True)
        (repo / SOURCE).write_text("VALUE = 1\n")
        ingestion = repo / "systems/deepread/ingestion.py"
        ingestion.write_text("INDEX = 1\n")
        for spec in DIAGNOSTIC_SOURCE_POLICY:
            path = repo / spec.path
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n")
        (repo / "tests").mkdir()
        (repo / "tests/test_fixture.py").write_text("import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n")
        write(repo / "config/candidate-test-policy.json", {
            "schema_version": "deepread-candidate-test-policy-v1", "policy_id": "fixture",
            "checks": [{"check_id": "fixture", "kind": "unittest_discover", "start_directory": "tests",
                        "pattern": "test_*.py", "timeout_seconds": 30, "minimum_tests": 1}],
        })
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=f@x",
                        "commit", "-qm", "baseline"], check=True)
        base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        docs = root / "documents.jsonl"
        (root / "report.md").write_text("# Report\n\nEvidence\n")
        docs.write_text(json.dumps({"document_id": "report", "path": "report.md"}) + "\n")
        dataset = root / "dataset.jsonl"
        rows = [{"financebench_id": task, "doc_name": "report", "question": task + "?"}
                for task in ["dev1", "dev2", "held", "cross"]]
        dataset.write_text("\n".join(json.dumps(r) for r in rows))
        store = root / "store"
        write(store / "corpus.json", {})
        write(store / "STORE_MANIFEST.json", {"schema_version": "deepread-store-build-v1",
              "document_manifest": {"sha256": sha(docs)},
              "source_files": [{"path": "systems/deepread/ingestion.py", "sha256": sha(ingestion)}],
              "artifacts": [{"path": "corpus.json", "sha256": sha(store / "corpus.json")}]})
        experiment = root / "experiment"
        initial = {"dataset": str(dataset), "documents": str(docs), "store": str(store),
                   "max_rounds": 50, "retrieval_topk": 5}
        write(experiment / "experiment_manifest.json", {"config": initial})
        write(experiment / "baseline/manifest.json", {"task_ids": ["dev1", "dev2"], "providers": {"chat": "fixture"}})
        write(experiment / "evaluation/evaluation.json", [
            {"task_id": r["financebench_id"], "sample_id": "report", "question": r["question"],
             "prediction": {"status": "ok"}, "token_usage": {"input_tokens": 90, "output_tokens": 10},
             "metrics": {"accuracy_normalized": 0.25}} for r in rows[:2]
        ])
        plans = [{"plan_id": f"plan-{i}", "hypothesis_id": "hyp-1", "decision": "proceed",
                  "edit_scope": {"allowed_paths": [SOURCE], "max_files_to_modify": 1,
                                 "requires_store_rebuild": False, "forbidden_roots": ["tests/"]},
                  "change_contract": {"current_behavior": "one", "required_behavior_delta": f"verify-{i}",
                                      "must_preserve": ["retrieval"], "non_goals": ["providers"]},
                  "risk": {"level": "medium", "regression_scenarios": ["answers"]}} for i in [1, 2]]
        write(experiment / "planning/plan.json", {"schema_version": "deepread-modification-plan-v1",
              "cohort_id": "cohort", "hypothesis_set_status": "ready", "plans": plans})
        write(experiment / "planning/cohort.json", {"schema_version": "deepread-hypothesis-cohort-v1",
              "cohort_id": "cohort", "eligible_diagnoses": [{"task_id": "dev1"}, {"task_id": "dev2"}],
              "excluded_diagnoses": []})
        write(experiment / "planning/hypotheses.json", {"schema_version": "deepread-improvement-hypotheses-v1",
              "cohort_id": "cohort", "status": "ready", "hypotheses": [{"hypothesis_id": "hyp-1",
              "maturity": "recurring", "task_ids": ["dev1", "dev2"]}]})
        validation = root / "validation.json" if promotion else None
        if validation:
            write(validation, {"schema_version": "deepread-repair-validation-v1", "cohorts": [
                {"name": name, "dataset": ds, "role": role, "dataset_path": str(dataset),
                 "documents": str(docs), "store": str(store), "task_ids": [task]}
                for name, ds, role, task in [("hold", "financebench", "holdout", "held"),
                                              ("cross", "other", "cross_dataset", "cross")]]})
        config = RepairConfig(experiment=experiment, output=root / "output", repo_root=repo,
                              env_file=root / ".env", validation_config=validation,
                              workers=2, max_iterations=iterations)
        config.output.mkdir()
        return config, initial, base

    def freeze_experiment(self, config, initial, base):
        config.env_file.write_text("LLM_MODEL=fixture\nLLM_API_KEY=very-secret-key\n")
        value = ExperimentConfig(experiment_id="fixture", source_root=config.repo_root,
                                 documents=Path(initial["documents"]), dataset=Path(initial["dataset"]),
                                 store=Path(initial["store"]), output=config.experiment,
                                 env_file=config.env_file, mode="diagnose")
        write(config.experiment / "experiment_manifest.json", {
            "schema_version": "deepread-experiment-v1", "experiment_id": "fixture",
            "config": value.manifest_config(), "stages": list(value.stages),
            "inputs": {"documents_sha256": sha(value.documents), "dataset_sha256": sha(value.dataset),
                       "source_revision": base},
        })
        artifact = config.experiment / "planning/plan.json"
        write(config.experiment / "experiment_state.json", {"schema_version": "deepread-experiment-state-v1",
              "status": "completed", "stages": {stage: {"status": "completed", "artifacts": {
              "primary": {"path": str(artifact), "sha256": sha(artifact)}}} for stage in value.stages}})
        bundle = write(config.experiment / "diagnostic_bundles/dev1/bundle.json", {
            "schema_version": "deepread-diagnostic-input-v1", "trajectory": {"turns": []},
            "access": {"payloads": [], "sources": [{"path": spec.path,
                       "sha256": sha(config.repo_root / spec.path)} for spec in DIAGNOSTIC_SOURCE_POLICY]},
        })
        write(config.experiment / "diagnostic_bundles/index.json", {"items": [{"bundle": str(bundle)}]})

    def test_top_level_dry_run_checks_real_frozen_git_sources_without_writes_or_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            with patch.dict(os.environ, {"LLM_MODEL": "fixture"}):
                report = run_repairs(config, dry_run=True)
            self.assertEqual(report["plan_ids"], ["plan-1", "plan-2"])
            self.assertFalse(report["writes_performed"])
            self.assertFalse((config.output / "manifest.json").exists())
            self.assertNotIn("very-secret-key", json.dumps(report))

    def test_top_level_entrypoint_and_frozen_resume_run_all_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            commands = FixtureCommands(config)
            with patch.dict(os.environ, {"LLM_MODEL": "fixture"}), patch(
                "agentic_rag_evolve.orchestration.repairs._command", side_effect=commands
            ):
                report = run_repairs(config, progress=lambda _: None)
                count = len(commands.calls)
                resumed = run_repairs(config, resume=True, progress=lambda _: None)
            self.assertEqual(report, resumed)
            self.assertEqual(count, len(commands.calls))

    def test_resume_can_reduce_workers_without_changing_scientific_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            commands = FixtureCommands(config)
            with patch.dict(os.environ, {"LLM_MODEL": "fixture"}), patch(
                "agentic_rag_evolve.orchestration.repairs._command", side_effect=commands
            ):
                run_repairs(config, progress=lambda _: None)
                count = len(commands.calls)
                resumed = run_repairs(replace(config, workers=1), resume=True, progress=lambda _: None)
            self.assertEqual(resumed["workers"], 1)
            self.assertEqual(count, len(commands.calls))

    def test_changed_model_requires_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            with patch.dict(os.environ, {"LLM_MODEL": "new-model"}):
                with self.assertRaisesRegex(ValueError, "models differ"):
                    run_repairs(config, dry_run=True)
                report = run_repairs(replace(config, allow_model_change=True), dry_run=True)
            self.assertTrue(report["model_changed_from_control"])

    def test_legacy_resume_can_upgrade_framework_and_budgets_without_rerunning_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            commands = FixtureCommands(config)
            with patch.dict(os.environ, {}, clear=True), patch(
                "agentic_rag_evolve.orchestration.repairs._command", side_effect=commands
            ):
                run_repairs(config, progress=lambda _: None)
                manifest_path = config.output / "manifest.json"
                legacy = json.loads(manifest_path.read_text())
                legacy.pop("evolution_model")
                legacy["config"].pop("modification_max_rounds")
                legacy["config"].pop("modification_max_tool_calls")
                legacy["models"]["RERANK_MODEL"] = ""
                write(manifest_path, legacy)
                legacy_bytes = manifest_path.read_bytes()
                count = len(commands.calls)
                config.env_file.write_text("DEEPREAD_LLM_MODEL=fixture\nDEEPREAD_LLM_BASE_URL=https://target.invalid/v1\n"
                    "DEEPREAD_LLM_API_KEY=target-secret\nEVOLUTION_LLM_MODEL=strong\n"
                    "EVOLUTION_LLM_BASE_URL=https://framework.invalid/v1\nEVOLUTION_LLM_API_KEY=framework-secret\n")
                updated = replace(config, modification_max_rounds=100, modification_max_tool_calls=200)
                self.assertEqual(run_repairs(updated, resume=True, dry_run=True)["status"], "dry_run")
                result = run_repairs(updated, resume=True, progress=lambda _: None)
                self.assertFalse(result["model_changed_from_control"])
                self.assertEqual(count, len(commands.calls))
                self.assertEqual(manifest_path.read_bytes(), legacy_bytes)
                session = json.loads((config.output / "execution/attempt-0002.json").read_text())
                self.assertEqual(session["evolution_model"], "strong")
                self.assertEqual(session["config"]["modification_max_rounds"], 100)
                self.assertNotIn("framework-secret", json.dumps(session))
                config.env_file.write_text(config.env_file.read_text().replace("DEEPREAD_LLM_MODEL=fixture", "DEEPREAD_LLM_MODEL=other"))
                with self.assertRaisesRegex(ValueError, "models differ"):
                    run_repairs(updated, resume=True, progress=lambda _: None)
                with self.assertRaisesRegex(ValueError, "models changed since start"):
                    run_repairs(replace(updated, allow_model_change=True), resume=True, dry_run=True)

    def test_all_plans_screened_in_isolated_worktrees_and_resume_skips_api(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            commands = FixtureCommands(config)
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=commands):
                report = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
                count = len(commands.calls)
                resumed = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(count, len(commands.calls))
            self.assertEqual(report, resumed)
            results = report["rounds"][0]["candidates"]
            self.assertEqual([r["status"] for r in results], ["screened", "screened"])
            self.assertEqual(results[0]["cohorts"][0]["candidate_mean"], 1.0)
            self.assertEqual(results[1]["cohorts"][0]["candidate_mean"], 0.75)
            self.assertFalse(report["rounds"][0]["promotion_performed"])
            self.assertEqual((config.repo_root / SOURCE).read_text(), "VALUE = 1\n")

    def test_promotion_selects_one_winner_without_mutating_main_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory), promotion=True)
            commands = FixtureCommands(config)
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=commands):
                report = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            result = report["rounds"][0]
            self.assertEqual(result["selected_plan_id"], "plan-1")
            self.assertTrue(result["promotion_performed"])
            self.assertNotEqual(result["next_commit"], base)
            self.assertEqual((config.repo_root / SOURCE).read_text(), "VALUE = 1\n")
            self.assertEqual(len(list((config.output / "baselines/entries").glob("*.json"))), 2)

    def test_two_rounds_promote_then_reject_and_resume_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory), promotion=True, iterations=2)
            commands = FixtureCommands(config)
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=commands):
                report = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
                count = len(commands.calls)
                resumed = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(report, resumed)
            self.assertEqual(len(commands.calls), count)
            self.assertTrue(report["rounds"][0]["promotion_performed"])
            self.assertFalse(report["rounds"][1]["promotion_performed"])
            self.assertEqual([c["status"] for c in report["rounds"][1]["candidates"]], ["rejected", "rejected"])
            self.assertEqual(len(list((config.output / "memory/rejected").glob("*.json"))), 2)
            self.assertEqual(len(list((config.output / "memory/accepted").glob("*.json"))), 1)
            self.assertEqual(commands.calls.count("runner.run_experiment"), 1)

    def test_interrupted_modifier_keeps_dirty_attempt_and_retries_only_failed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            commands = FixtureCommands(config)
            interrupted = []
            def fail_once(*args, **kwargs):
                result = commands(*args, **kwargs)
                if args[2] == "runner.run_candidate_modification" and not interrupted:
                    interrupted.append(True)
                    raise RuntimeError("interrupted after editor writes")
                return result
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=fail_once):
                first = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
                self.assertEqual(first["status"], "failed")
                second = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(commands.calls.count("runner.run_candidate_modification"), 3)
            dirs = list((config.output / "round-0001/candidates").glob("*/attempt-*/worktree"))
            self.assertEqual(len(dirs), 3)
            self.assertTrue(all("VALUE = 1" not in (d / SOURCE).read_text() for d in dirs))

    def test_failed_batch_resumes_with_new_framework_model_and_larger_edit_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            self.freeze_experiment(config, initial, base)
            commands = FixtureCommands(config)
            failed = []
            def fail_once(*args, **kwargs):
                result = commands(*args, **kwargs)
                if args[2] == "runner.run_candidate_modification" and not failed:
                    failed.append(True)
                    raise RuntimeError("budget exhausted")
                return result
            with patch.dict(os.environ, {}, clear=True), patch(
                "agentic_rag_evolve.orchestration.repairs._command", side_effect=fail_once
            ):
                self.assertEqual(run_repairs(config, progress=lambda _: None)["status"], "failed")
                config.env_file.write_text(config.env_file.read_text() +
                    "EVOLUTION_LLM_MODEL=strong\nEVOLUTION_LLM_BASE_URL=https://strong.invalid/v1\n"
                    "EVOLUTION_LLM_API_KEY=strong-key\n")
                upgraded = replace(config, modification_max_rounds=100, modification_max_tool_calls=200)
                self.assertEqual(run_repairs(upgraded, resume=True, progress=lambda _: None)["status"], "completed")
            self.assertEqual(commands.calls.count("runner.run_candidate_modification"), 3)
            self.assertEqual(len(list((config.output / "round-0001/candidates").glob("*/attempt-*/worktree"))), 3)

    def test_outer_round_limit_can_be_extended_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory), promotion=True)
            self.freeze_experiment(config, initial, base)
            commands = FixtureCommands(config)
            with patch.dict(os.environ, {}, clear=True), patch(
                "agentic_rag_evolve.orchestration.repairs._command", side_effect=commands
            ):
                first = run_repairs(config, progress=lambda _: None)
                report = run_repairs(replace(config, max_iterations=2), resume=True, progress=lambda _: None)
                self.assertEqual(report["rounds"][0], first["rounds"][0])
                self.assertEqual(len(report["rounds"]), 2)
                with self.assertRaisesRegex(ValueError, "cannot be reduced"):
                    run_repairs(config, resume=True, progress=lambda _: None)

    def test_rejected_round_replans_with_memory_without_rerunning_baseline_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory), promotion=True, iterations=2)
            commands = FixtureCommands(config)
            commands.holdout_regress = True
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=commands):
                report = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(any(r["promotion_performed"] for r in report["rounds"]))
            self.assertEqual(commands.calls.count("runner.run_experiment"), 0)
            self.assertEqual(commands.calls.count("runner.run_modification_planning"), 1)
            self.assertEqual(len(list((config.output / "memory/rejected").glob("*.json"))), 4)
            self.assertEqual(len(list((config.output / "baselines/entries").glob("*.json"))), 1)

    def test_resume_after_promotion_memory_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory), promotion=True)
            commands = FixtureCommands(config)
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=commands):
                first = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
                state_path = config.output / "round-0001/control/state.json"
                state = json.loads(state_path.read_text())
                state["stages"]["advance"]["status"] = "failed"
                state["stages"]["terminal"]["status"] = "failed"
                write(state_path, state)
                count = len(commands.calls)
                second = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(first, second)
            self.assertEqual(count, len(commands.calls))
            self.assertEqual(len(list((config.output / "memory/accepted").glob("*.json"))), 1)

    def test_cohort_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory), promotion=True)
            spec = json.loads(config.validation_config.read_text())
            spec["cohorts"][0]["task_ids"] = ["dev1"]
            write(config.validation_config, spec)
            with self.assertRaisesRegex(ValueError, "cannot become holdout"):
                _cohorts(config, config.experiment)

    def test_checkpoint_preserves_partial_attempt_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = RepairCheckpoint(Path(directory))
            def fail(directory):
                (directory / "partial.txt").write_text("keep")
                raise RuntimeError("network")
            with self.assertRaises(RuntimeError):
                checkpoint.run("stage", fail)
            def succeed(directory):
                path = directory / "value.json"
                write(path, {})
                return {"value": path}, {"ok": True}
            checkpoint.run("stage", succeed)
            self.assertTrue((Path(directory) / "stage/attempt-0001/partial.txt").exists())
            (Path(directory) / "stage/attempt-0002/value.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                RepairCheckpoint(Path(directory))

    def test_plan_selection_has_no_manual_selector(self):
        document = {"schema_version": "deepread-modification-plan-v1", "plans": [
            {"plan_id": "a", "decision": "proceed"}, {"plan_id": "b", "decision": "defer"},
            {"plan_id": "c", "decision": "proceed"}]}
        self.assertEqual([p["plan_id"] for p in proceeding_plans(document)], ["a", "c"])
        document["plans"][0]["plan_id"] = "../outside"
        with self.assertRaises(ValueError):
            proceeding_plans(document)
        self.assertIsNone(select_winner([{"status": "screened"}]))

    def test_parallel_subprocess_loads_candidate_code_not_editable_main(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory))
            directory = Path(directory) / "command"
            directory.mkdir()
            candidate = Path(directory) / "source"
            candidate.mkdir()
            seen = {}
            def process(command, **kwargs):
                seen.update(kwargs)
                kwargs["stdout"].write('{}\n')
                return subprocess.CompletedProcess(command, 0)
            with patch("subprocess.run", side_effect=process):
                _command(config, directory, "runner.run_deepread", [], cwd=candidate)
            self.assertEqual(seen["cwd"], candidate)
            self.assertEqual(seen["env"]["PYTHONPATH"].split(os.pathsep),
                             [str(config.repo_root / "src"), str(candidate), str(config.repo_root)])

    def test_real_subprocess_uses_current_framework_with_old_candidate_target(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory))
            command_dir = Path(directory) / "command"
            command_dir.mkdir()
            candidate = Path(directory) / "old-candidate"
            for path, contents in (
                (config.repo_root / "src/agentic_rag_evolve/__init__.py", 'VERSION="current-framework"\n'),
                (candidate / "src/agentic_rag_evolve/__init__.py", 'VERSION="old-framework"\n'),
                (config.repo_root / "systems/__init__.py", 'VERSION="main-target"\n'),
                (candidate / "systems/__init__.py", 'VERSION="candidate-target"\n'),
                (candidate / "runner/__init__.py", ""),
                (candidate / "runner/probe.py", 'import json, systems, agentic_rag_evolve\n'
                 'print(json.dumps({"framework":agentic_rag_evolve.VERSION,"target":systems.VERSION}))\n'),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            result = _command(config, command_dir, "runner.probe", [], cwd=candidate)
            self.assertEqual(result, {"framework": "current-framework", "target": "candidate-target"})

    def test_modification_budgets_reach_command_and_reject_nonpositive_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config, initial, base = self.fixture(Path(directory))
            config = replace(config, modification_max_rounds=90, modification_max_tool_calls=180)
            commands = FixtureCommands(config)
            def command(config, directory, module, arguments, **kwargs):
                if module == "runner.run_candidate_modification":
                    arguments = list(map(str, arguments))
                    self.assertEqual(arguments[arguments.index("--max-rounds") + 1], "90")
                    self.assertEqual(arguments[arguments.index("--max-tool-calls") + 1], "180")
                return commands(config, directory, module, arguments, **kwargs)
            with patch("agentic_rag_evolve.orchestration.repairs._command", side_effect=command):
                result = _run_rounds(config, {"base_commit": base}, initial, _cohorts(config, config.experiment), lambda _: None)
            self.assertEqual(result["status"], "completed")
            for key in ("modification_max_rounds", "modification_max_tool_calls"):
                with self.assertRaisesRegex(ValueError, "budgets must be positive"):
                    replace(config, **{key: 0})

    def test_multi_round_requires_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "promotion validation"):
                RepairConfig(experiment=root / "e", output=root / "o", repo_root=root / "r",
                             env_file=root / ".env", max_iterations=2)


if __name__ == "__main__":
    unittest.main()
