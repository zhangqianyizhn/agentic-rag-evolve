import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic_rag_evolve.diagnostics.policy import DIAGNOSTIC_SOURCE_POLICY
from agentic_rag_evolve.orchestration.experiment import (
    BASELINE_STAGES,
    DIAGNOSIS_STAGES,
    ExperimentCheckpoint,
    ExperimentConfig,
    _completed_diagnosis,
    _diagnoses,
    _hypotheses,
    _modification_plan,
    _prepare_planning_retry,
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

    def test_experiment_reuses_failed_candidate_for_output_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, mode="diagnose")
            self._write_frozen_bundle(config)
            previous = config.output / "diagnoses/q1/attempt-0001"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text(json.dumps({"status": "validation_error"}))
            (previous / "candidate.json").write_text("{}")

            def complete(**kwargs):
                output = kwargs["output_path"]
                output.mkdir()
                (output / "audit.json").write_text(json.dumps({"status": "skipped"}))
                return SimpleNamespace(status="skipped")

            with patch("agentic_rag_evolve.orchestration.experiment.load_chat_model"), patch(
                "agentic_rag_evolve.orchestration.experiment.run_diagnosis", side_effect=complete
            ) as run:
                _, details = _diagnoses(config)
                self.assertEqual(run.call_args.kwargs["recovery_path"], previous)
                self.assertEqual(run.call_args.kwargs["output_path"].name, "attempt-0002")
                self.assertEqual(details["skipped"], 1)
                _diagnoses(config)
                self.assertEqual(run.call_count, 1)

    def test_failed_planning_stages_archive_audits_and_can_resume(self) -> None:
        for name, action, schema in (
            ("hypotheses", _hypotheses, "deepread-hypothesis-audit-v1"),
            ("plan", _modification_plan, "deepread-modification-plan-audit-v1"),
        ):
            with self.subTest(stage=name), tempfile.TemporaryDirectory() as directory:
                config = self._config(Path(directory), mode="diagnose")
                planning = config.output / "planning"
                planning.mkdir(parents=True)
                cohort = {"schema_version": "deepread-hypothesis-cohort-v1",
                          "cohort_id": "cohort-1", "eligible_diagnoses": []}
                (planning / "cohort.json").write_text(json.dumps(cohort))
                if name == "plan":
                    (planning / "hypotheses.json").write_text(json.dumps({
                        "schema_version": "deepread-improvement-hypotheses-v1",
                        "cohort_id": "cohort-1", "status": "no_eligible_diagnoses",
                        "hypotheses": [],
                    }))
                audit = planning / f"{name}.audit.json"
                failed_bytes = json.dumps({"schema_version": schema, "status": "validation_error"}).encode()
                audit.write_bytes(failed_bytes)
                candidate = planning / f"{name}.candidate.json"
                candidate.write_text("{}")
                with patch("agentic_rag_evolve.orchestration.experiment.load_chat_model",
                           return_value=SimpleNamespace(model_name="unused")) as load:
                    artifacts, _ = action(config)
                    action(config)
                    self.assertEqual(load.call_count, 1)
                archive = artifacts["attempt_history"] / "attempt-0001"
                self.assertEqual((archive / audit.name).read_bytes(), failed_bytes)
                self.assertEqual((archive / candidate.name).read_text(), "{}")
                manifest = json.loads((archive / "manifest.json").read_text())
                for item in manifest["artifacts"]:
                    self.assertEqual(hashlib.sha256((archive / item["path"]).read_bytes()).hexdigest(), item["sha256"])
                self.assertTrue(artifacts["primary"].is_file())

    def test_retry_refuses_to_discard_successful_audit_with_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hypotheses.json"
            output.with_suffix(".audit.json").write_text(json.dumps({
                "schema_version": "deepread-hypothesis-audit-v1", "status": "ok",
            }))
            with self.assertRaisesRegex(ValueError, "successful planning audit has no result"):
                _prepare_planning_retry(output, audit_schema="deepread-hypothesis-audit-v1",
                                        successful_statuses={"ok"})
            self.assertTrue(output.with_suffix(".audit.json").is_file())


if __name__ == "__main__":
    unittest.main()
