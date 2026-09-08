import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.diagnostics import (
    DiagnosticArtifactReader,
    build_diagnostic_bundle,
    store_fingerprint,
)


class DiagnosticBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_root = Path(__file__).resolve().parents[1]

    def _build(self, root: Path):
        payload_data = json.dumps(
            {"ok": True, "structure": "section " * 40},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        payload_relative = Path("payloads/q1/structure.json")
        payload_path = root / payload_relative
        payload_path.parent.mkdir(parents=True)
        payload_path.write_bytes(payload_data)
        digest = hashlib.sha256(payload_data).hexdigest()
        trajectory = {
            "schema_version": "deepread-trajectory-v2",
            "run_id": "r1",
            "task_id": "q1",
            "question": "What was revenue?",
            "status": "ok",
            "answer": "12 million",
            "token_usage": {"input_tokens": 10, "output_tokens": 2},
            "transport_debug": "must not leak",
            "turns": [{
                "round": 1,
                "model": {
                    "name": "model",
                    "reasoning": "Find the report.",
                    "transport_debug": "must not leak",
                },
                "tools": [{
                    "call_id": "c1",
                    "name": "get_doc_structure",
                    "arguments": {"doc_id": ["1"]},
                    "ok": True,
                    "result_summary": {"type": "dict"},
                    "result_ref": {
                        "path": payload_relative.as_posix(),
                        "sha256": digest,
                        "bytes": len(payload_data),
                    },
                    "internal_debug": "must not leak",
                }],
                "raw_event_range": {"first": "event_1", "last": "event_4", "count": 4},
            }],
            "summary": {"round_count": 1, "warnings": []},
        }
        evaluation = [{
            "task_id": "q1",
            "sample_id": "report",
            "question": "What was revenue?",
            "category": "numeric",
            "gold_answers": ["12 million"],
            "evidence": ["Revenue was $12 million."],
            "prediction": {
                "status": "ok",
                "answer": "12 million",
                "termination_reason": "final_answer",
                "rounds_completed": 1,
            },
            "retrieval": {"text_count": 1, "evidence_matches": []},
            "metrics": {"f1": 1.0, "recall": 1.0, "internal_metric": "must not leak"},
            "judge": {"status": "ok", "score": 4, "reasoning": "correct"},
        }]
        manifest = {
            "run_id": "r1",
            "dataset_sha256": "dataset-hash",
            "store_name": "store_index",
            "store_fingerprint": None,
            "config": {"max_rounds": 50, "provider_timeout": "must not leak"},
            "providers": {"chat": "model", "endpoint": "must not leak"},
        }
        trajectory_path = root / "q1.trajectory.json"
        evaluation_path = root / "evaluation.json"
        manifest_path = root / "manifest.json"
        store = root / "store"
        store.mkdir()
        (store / "report_corpus.json").write_text(json.dumps({
            "nodes": [{
                "id": "n1",
                "title": "Revenue",
                "paragraphs": ["Revenue was $12 million."],
                "children": [],
            }]
        }))
        manifest["store_fingerprint"] = store_fingerprint(store)
        trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "diagnostic"
        report = build_diagnostic_bundle(
            trajectory_path=trajectory_path,
            evaluation_path=evaluation_path,
            run_manifest_path=manifest_path,
            store_path=store,
            source_root=self.source_root,
            output_path=output,
        )
        return report, output, payload_relative

    def test_bundle_projects_fields_and_excludes_framework_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, output, _ = self._build(Path(directory))
            bundle = json.loads((output / "bundle.json").read_text())

        serialized_trajectory = json.dumps(bundle["trajectory"])
        source_paths = [source["path"] for source in bundle["access"]["sources"]]
        self.assertEqual(report.task_id, "q1")
        self.assertNotIn("run_id", bundle["trajectory"])
        self.assertNotIn("must not leak", serialized_trajectory)
        self.assertNotIn("must not leak", json.dumps(bundle))
        self.assertEqual(bundle["evaluation"]["gold_evidence"], ["Revenue was $12 million."])
        self.assertNotIn("evidence", bundle["evaluation"])
        self.assertEqual(bundle["evidence_coverage"]["layers"]["corpus"]["recall"], 1.0)
        self.assertEqual(bundle["evidence_coverage"]["primary_signal"], "candidate")
        self.assertEqual(bundle["failure_signals"]["triage"], "pass")
        self.assertEqual(
            bundle["failure_signals"]["signals"][0]["code"],
            "retrieval_candidate_miss",
        )
        self.assertTrue(source_paths)
        self.assertFalse(any("telemetry" in path or "providers" in path for path in source_paths))
        self.assertNotIn("systems/deepread/runtime.py", source_paths)
        self.assertEqual(
            [tool["name"] for tool in bundle["access"]["tools"]],
            ["list_sources", "read_source", "read_payload"],
        )

    def test_reader_enforces_source_and_payload_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, output, payload_relative = self._build(Path(directory))
            reader = DiagnosticArtifactReader(
                bundle_path=output / "bundle.json",
                source_root=self.source_root,
            )
            source = reader.read_source(
                "systems/deepread/DeepRead/agent/runner.py",
                start_line=1,
                end_line=5,
            )
            payload = reader.read_payload(payload_relative.as_posix(), limit_chars=20)

            self.assertEqual(source["start_line"], 1)
            self.assertIn("from __future__", source["content"])
            self.assertTrue(payload["has_more"])
            with self.assertRaises(PermissionError):
                reader.read_source("src/agentic_rag_evolve/telemetry/model.py")
            with self.assertRaises(PermissionError):
                reader.read_payload("../q1.trajectory.json")
            with self.assertRaisesRegex(ValueError, "240 lines"):
                reader.read_source(
                    "systems/deepread/DeepRead/agent/runner.py",
                    start_line=1,
                    end_line=241,
                )

    def test_reader_detects_payload_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, output, payload_relative = self._build(Path(directory))
            reader = DiagnosticArtifactReader(
                bundle_path=output / "bundle.json",
                source_root=self.source_root,
            )
            (output / payload_relative).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload byte count mismatch"):
                reader.read_payload(payload_relative.as_posix())

    def test_builder_rejects_prediction_answer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root)
            evaluation_path = root / "evaluation.json"
            evaluation = json.loads(evaluation_path.read_text())
            evaluation[0]["prediction"]["answer"] = "different answer"
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "answer mismatch"):
                build_diagnostic_bundle(
                    trajectory_path=root / "q1.trajectory.json",
                    evaluation_path=evaluation_path,
                    run_manifest_path=root / "manifest.json",
                    store_path=root / "store",
                    source_root=self.source_root,
                    output_path=root / "mismatch",
                )


if __name__ == "__main__":
    unittest.main()
