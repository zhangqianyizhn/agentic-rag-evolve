import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.diagnostics.policy import DIAGNOSTIC_SOURCE_POLICY
from agentic_rag_evolve.orchestration.experiment import (
    BASELINE_STAGES,
    DIAGNOSIS_STAGES,
    ExperimentCheckpoint,
    ExperimentConfig,
    _completed_diagnosis,
    _preflight,
)


class ExperimentRunnerTest(unittest.TestCase):
    def _config(self, root: Path, *, mode: str = "baseline", judge: bool = True):
        document = root / "report.md"
        document.write_text("# Report\n\nEvidence", encoding="utf-8")
        documents = root / "documents.jsonl"
        documents.write_text(
            json.dumps({"document_id": "report", "path": "report.md"}) + "\n",
            encoding="utf-8",
        )
        dataset = root / "dataset.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "financebench_id": "q1",
                    "doc_name": "report",
                    "question": "What happened?",
                    "answer": "Evidence",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        env_file = root / ".env"
        env_file.write_text(
            "LLM_MODEL=chat\n"
            "LLM_BASE_URL=https://example.invalid/v1\n"
            "LLM_API_KEY=test\n"
            "EMBEDDING_MODEL_NAME=embedding\n"
            "EMBEDDING_BASE_URL=https://example.invalid/v1\n"
            "EMBEDDING_API_KEY=test\n",
            encoding="utf-8",
        )
        return ExperimentConfig(
            experiment_id="experiment-1",
            source_root=Path(__file__).resolve().parents[1],
            documents=documents,
            dataset=dataset,
            store=root / "store",
            output=root / "output",
            env_file=env_file,
            mode=mode,
            judge=judge,
        )

    def _write_frozen_bundle(self, config: ExperimentConfig) -> None:
        bundle_root = config.output / "diagnostic_bundles" / "q1"
        bundle_root.mkdir(parents=True)
        sources = []
        for spec in DIAGNOSTIC_SOURCE_POLICY:
            path = config.source_root / spec.path
            data = path.read_bytes()
            sources.append({
                "path": spec.path,
                "component": spec.component,
                "purpose": spec.purpose,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "line_count": len(data.decode("utf-8").splitlines()),
            })
        bundle = bundle_root / "bundle.json"
        bundle.write_text(json.dumps({
            "schema_version": "deepread-diagnostic-input-v1",
            "task": {"task_id": "q1"},
            "trajectory": {"turns": []},
            "access": {"sources": sources, "payloads": []},
        }))
        index = config.output / "diagnostic_bundles" / "index.json"
        index.write_text(json.dumps({
            "schema_version": "deepread-diagnostic-bundle-index-v1",
            "items": [{"task_id": "q1", "bundle": str(bundle)}],
        }))

    def test_modes_have_explicit_stage_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = self._config(Path(directory))
            diagnosis = ExperimentConfig(
                **{**baseline.manifest_config(), "mode": "diagnose"}
            )

        self.assertEqual(baseline.stages, BASELINE_STAGES)
        self.assertEqual(diagnosis.stages, DIAGNOSIS_STAGES)

    def test_diagnose_mode_requires_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires the answer judge"):
                self._config(Path(directory), mode="diagnose", judge=False)

    def test_checkpoint_skips_verified_completed_stage_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            checkpoint = ExperimentCheckpoint(config, resume=False)
            calls = []

            def action():
                calls.append("run")
                artifact = root / "artifact.json"
                artifact.write_text("{}", encoding="utf-8")
                return {"primary": artifact}, {"ok": True}

            self.assertTrue(checkpoint.run("preflight", action))
            resumed = ExperimentCheckpoint(config, resume=True)
            self.assertFalse(resumed.run("preflight", action))

        self.assertEqual(calls, ["run"])

    def test_checkpoint_rejects_changed_completed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            checkpoint = ExperimentCheckpoint(config, resume=False)
            artifact = root / "artifact.json"

            def action():
                artifact.write_text("{}", encoding="utf-8")
                return {"primary": artifact}, {}

            checkpoint.run("preflight", action)
            artifact.write_text('{"changed":true}', encoding="utf-8")
            resumed = ExperimentCheckpoint(config, resume=True)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                resumed.run("preflight", action)

    def test_checkpoint_allows_framework_revision_change_after_sources_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, mode="diagnose")
            checkpoint = ExperimentCheckpoint(config, resume=False)
            self._write_frozen_bundle(config)
            checkpoint.state["stages"]["diagnostic_bundles"] = {"status": "completed"}
            checkpoint._save()
            manifest = json.loads(checkpoint.manifest_path.read_text())
            manifest["inputs"]["source_revision"] = "0" * 40
            checkpoint.manifest_path.write_text(json.dumps(manifest))

            resumed = ExperimentCheckpoint(config, resume=True)

        self.assertEqual(resumed.state["stages"]["diagnostic_bundles"]["status"], "completed")

    def test_checkpoint_rejects_revision_change_when_frozen_deepread_source_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, mode="diagnose")
            checkpoint = ExperimentCheckpoint(config, resume=False)
            self._write_frozen_bundle(config)
            checkpoint.state["stages"]["diagnostic_bundles"] = {"status": "completed"}
            checkpoint._save()
            manifest = json.loads(checkpoint.manifest_path.read_text())
            manifest["inputs"]["source_revision"] = "0" * 40
            checkpoint.manifest_path.write_text(json.dumps(manifest))
            bundle_path = config.output / "diagnostic_bundles" / "q1" / "bundle.json"
            bundle = json.loads(bundle_path.read_text())
            bundle["access"]["sources"][0]["sha256"] = "f" * 64
            bundle_path.write_text(json.dumps(bundle))

            with self.assertRaisesRegex(ValueError, "source changed since bundle creation"):
                ExperimentCheckpoint(config, resume=True)

    def test_preflight_validates_document_coverage_without_recording_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            artifacts, details = _preflight(config)
            report = json.loads(artifacts["primary"].read_text(encoding="utf-8"))

        self.assertEqual(details["selected_query_count"], 1)
        self.assertEqual(report["document_count"], 1)
        self.assertFalse(report["secrets_recorded"])
        self.assertNotIn("test", json.dumps(report))

    def test_diagnosis_resume_selects_latest_completed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / "attempt-0001"
            failed.mkdir()
            (failed / "audit.json").write_text(
                json.dumps({"status": "error"}), encoding="utf-8"
            )
            completed = root / "attempt-0002"
            completed.mkdir()
            audit = completed / "audit.json"
            audit.write_text(json.dumps({"status": "skipped"}), encoding="utf-8")

            result = _completed_diagnosis(root)

        self.assertEqual(result, ("skipped", audit, None))


if __name__ == "__main__":
    unittest.main()
