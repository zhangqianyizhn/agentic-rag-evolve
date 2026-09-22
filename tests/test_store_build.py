import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.store_build import (
    build_markdown_store,
    verify_store_manifest,
)


class StoreBuildTest(unittest.TestCase):
    def test_builds_and_verifies_provenance_bound_store(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "report.md"
            document.write_text("# Report\n\nRevenue was 42.", encoding="utf-8")
            documents = root / "documents.jsonl"
            documents.write_text(
                json.dumps({"document_id": "report", "path": "report.md"}) + "\n",
                encoding="utf-8",
            )
            store = root / "store"

            result = build_markdown_store(
                document_manifest_path=documents,
                output_path=store,
                source_root=source_root,
                embedder=None,
            )
            verified = verify_store_manifest(store, store / "STORE_MANIFEST.json")

            self.assertEqual(result["schema_version"], "deepread-store-build-v1")
            self.assertEqual(verified["document_count"], 1)
            self.assertIsNone(verified["candidate"])
            self.assertTrue((store / "report_corpus.json").is_file())

    def test_verifier_rejects_changed_store_artifact(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "report.md"
            document.write_text("# Report\n\nBody", encoding="utf-8")
            documents = root / "documents.json"
            documents.write_text(
                json.dumps([{"document_id": "report", "path": "report.md"}]),
                encoding="utf-8",
            )
            store = root / "store"
            build_markdown_store(
                document_manifest_path=documents,
                output_path=store,
                source_root=source_root,
                embedder=None,
            )
            (store / "report_corpus.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_store_manifest(store, store / "STORE_MANIFEST.json")


if __name__ == "__main__":
    unittest.main()
