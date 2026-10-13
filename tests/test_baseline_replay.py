import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentic_rag_evolve.orchestration.baseline_reuse import validate_baseline_reuse
from agentic_rag_evolve.orchestration.experiment import (
    ExperimentConfig, _baseline, _path_digest, _sha256, _prepare_planning_retry,
)
from runner.run_replay import main


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    repo = root / "repo"
    target = repo / "systems/deepread/example.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "base"], check=True)
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    doc = root / "report.md"
    doc.write_text("# Report\nEvidence")
    documents = root / "documents.jsonl"
    write(documents, {"document_id": "report", "path": "report.md"})
    dataset = root / "dataset.jsonl"
    write(dataset, {"financebench_id": "q1", "doc_name": "report", "question": "Q", "answer": "A"})
    env = root / ".env"
    env.write_text("LLM_MODEL=target\nLLM_API_KEY=test\nLLM_BASE_URL=https://example.invalid/v1\n"
                   "EMBEDDING_MODEL_NAME=embedding\nEMBEDDING_API_KEY=test\n"
                   "EMBEDDING_BASE_URL=https://example.invalid/v1\n")
    store = root / "store"
    write(store / "report_corpus.json", {"data": "fixture"})
    write(store / "STORE_MANIFEST.json", {
        "schema_version": "deepread-store-build-v1",
        "documents": [{"document_id": "report", "sha256": _sha256(doc)}],
        "artifacts": [{"path": "report_corpus.json", "sha256": _sha256(store / "report_corpus.json")}],
    })
    original = root / "original"
    write(original / "baseline/manifest.json", {
        "task_ids": ["q1"], "dataset_sha256": _sha256(dataset),
        "config": {"max_rounds": 50, "retrieval_topk": 5},
        "providers": {"chat": "target", "embedding": "embedding", "reranker": None},
        "store_build": {"sha256": _sha256(store / "STORE_MANIFEST.json")},
    })
    write(original / "baseline/predictions.jsonl", {"task_id": "q1", "answer": "A"})
    write(original / "baseline/summary.json", {"query_count": 1})
    (original / "baseline/deepread_trace.jsonl").write_text('{"full":"old trace"}\n')
    oldconfig = ExperimentConfig(experiment_id="old", source_root=repo, env_file=env,
                                 documents=documents, dataset=dataset, store=store, output=original)
    write(original / "experiment_manifest.json", {"config": oldconfig.manifest_config(), "inputs": {
        "source_revision": revision, "documents_sha256": _sha256(documents), "dataset_sha256": _sha256(dataset)}})
    write(original / "experiment_state.json", {"stages": {"baseline": {"status": "completed", "artifacts": {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in ((p.name, p) for p in (original / "baseline").iterdir())}}}})
    return replace(oldconfig, experiment_id="new", output=root / "new",
                   reuse_baseline_experiment=original, mode="diagnose")


class BaselineReplayTest(unittest.TestCase):
    def test_reuses_exact_bytes_without_model_calls_and_keeps_original_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            old = _path_digest(config.reuse_baseline_experiment)
            with patch("agentic_rag_evolve.orchestration.experiment.run_financebench") as inference, \
                 patch("agentic_rag_evolve.orchestration.experiment.load_provider_bundle") as providers:
                refs, details = _baseline(config)
                again, _ = _baseline(config)
                inference.assert_not_called()
                providers.assert_not_called()
            self.assertEqual(refs, again)
            self.assertTrue(details["baseline_reused"])
            self.assertEqual(old, _path_digest(config.reuse_baseline_experiment))
            self.assertEqual(_path_digest(config.output / "baseline"),
                             _path_digest(config.reuse_baseline_experiment / "baseline"))
            self.assertFalse((config.output / "baseline").is_symlink())

    def test_rejects_changed_original_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            (config.reuse_baseline_experiment / "baseline/predictions.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_baseline_reuse(config)
            self.assertFalse(config.output.exists())

    def test_rejects_changed_target_source_documents_runtime_tasks_and_model(self):
        for kind in ("source", "document", "runtime", "tasks", "model"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                config = fixture(Path(directory))
                if kind == "source":
                    (config.source_root / "systems/deepread/example.py").write_text("VALUE = 2")
                elif kind == "document":
                    (Path(directory) / "report.md").write_text("Different content")
                elif kind == "runtime":
                    config = replace(config, max_rounds=51)
                elif kind == "tasks":
                    config = replace(config, task_ids=("q1", "q1"))
                else:
                    config.env_file.write_text(config.env_file.read_text().replace("LLM_MODEL=target", "LLM_MODEL=other"))
                with self.assertRaises(ValueError):
                    validate_baseline_reuse(config)
                self.assertFalse(config.output.exists())

    def test_partial_copy_preserved_then_new_attempt_used(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            with patch("agentic_rag_evolve.orchestration.baseline_reuse.shutil.copytree", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    _baseline(config)
            self.assertTrue((config.output / "baseline_imports/attempt-0001").is_dir())
            _baseline(config)
            self.assertTrue((config.output / "baseline_imports/attempt-0002/receipt.json").is_file())

    def test_framework_role_model_can_change_without_invalidating_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            config.env_file.write_text(config.env_file.read_text() +
                "EVOLUTION_LLM_MODEL=stronger\nEVOLUTION_LLM_API_KEY=test2\nEVOLUTION_LLM_BASE_URL=https://other.invalid/v1\n")
            validate_baseline_reuse(config)

    def test_dry_run_writes_nothing_and_full_command_runs_both_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            output = Path(directory) / "replay"
            command = ["replay", "--baseline-experiment", str(config.reuse_baseline_experiment),
                       "--output", str(output), "--repo-root", str(config.source_root),
                       "--env-file", str(config.env_file)]
            with patch("sys.argv", command + ["--dry-run"]), patch("builtins.print"), \
                 patch("runner.run_replay.run_experiment") as experiment, patch("runner.run_replay.run_repairs") as repairs:
                self.assertEqual(main(), 0)
                experiment.assert_not_called()
                repairs.assert_not_called()
            self.assertFalse(output.exists())
            phases = []
            with patch("sys.argv", command), patch("builtins.print"), \
                 patch("runner.run_replay.run_experiment", side_effect=lambda *a, **k: phases.append("experiment")), \
                 patch("runner.run_replay.run_repairs", side_effect=lambda *a, **k: phases.append("repairs") or {"status": "completed"}):
                self.assertEqual(main(), 0)
            self.assertEqual(phases, ["experiment", "repairs"])

    def test_failed_planning_archive_includes_full_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hypotheses.json"
            audit = output.with_suffix(".audit.json")
            write(audit, {"schema_version": "audit", "status": "error"})
            trace = output.with_suffix(".trace.json")
            write(trace, {"events": [{"response": "full model content"}]})
            old = trace.read_bytes()
            _prepare_planning_retry(output, audit_schema="audit", successful_statuses={"ok"})
            archived = output.parent / "attempts/hypotheses/attempt-0001/hypotheses.trace.json"
            self.assertEqual(archived.read_bytes(), old)
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
