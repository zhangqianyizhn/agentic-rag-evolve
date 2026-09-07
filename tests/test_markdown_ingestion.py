import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentic_rag_evolve.contracts import DocumentInput
from systems.deepread.ingestion import MarkdownIngestor
from systems.deepread.runtime import DeepReadConfig, GlobalDeepReadRuntime


class FakeEmbedder:
    model_name = "fake-embedding-v1"
    base_url = "local://fake"
    normalized = True

    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class MarkdownIngestionTest(unittest.TestCase):
    def test_builds_flat_corpus_and_vector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.md"
            source.write_text("# Revenue\n\nRevenue was 42.\n\n## Note\n\nAudited.", encoding="utf-8")
            store = root / "store"
            ingestor = MarkdownIngestor(store, FakeEmbedder())

            (result,) = ingestor.ingest((DocumentInput("report-id", source),))

            self.assertEqual(result.document_id, "report-id")
            self.assertEqual(result.paragraph_count, 2)
            self.assertEqual(result.corpus_path.parent, store)
            self.assertTrue(result.embedding_path and result.embedding_path.is_file())
            self.assertTrue(result.id_map_path and result.id_map_path.is_file())

            corpus = json.loads(result.corpus_path.read_text(encoding="utf-8"))
            vectors = np.load(result.embedding_path)
            id_map = json.loads(result.id_map_path.read_text(encoding="utf-8"))
            self.assertEqual(vectors.shape, (2, 3))
            self.assertEqual(len(id_map), 2)
            self.assertEqual(corpus["vector_store"]["model_name"], "fake-embedding-v1")

            runtime = GlobalDeepReadRuntime(
                store,
                root / "trace.jsonl",
                DeepReadConfig("test-model", "https://example.invalid/v1", "unused"),
            )
            self.assertEqual(runtime.load_index().doc_id_map, {"1": "report"})

    def test_can_build_lexical_only_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.markdown"
            source.write_text("# Heading\n\nBody", encoding="utf-8")

            (result,) = MarkdownIngestor(root / "store").ingest(
                (DocumentInput("report", source),)
            )

            corpus = json.loads(result.corpus_path.read_text(encoding="utf-8"))
            self.assertNotIn("vector_store", corpus)
            self.assertIsNone(result.embedding_path)

    def test_rejects_non_markdown_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.pdf"
            source.write_bytes(b"not-a-real-pdf")

            with self.assertRaisesRegex(ValueError, "Markdown required"):
                MarkdownIngestor(Path(directory) / "store").ingest(
                    (DocumentInput("report", source),)
                )

    def test_rejects_duplicate_output_basenames_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one" / "report.md"
            second = root / "two" / "report.md"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("# One", encoding="utf-8")
            second.write_text("# Two", encoding="utf-8")
            store = root / "store"

            with self.assertRaisesRegex(ValueError, "duplicate"):
                MarkdownIngestor(store).ingest(
                    (DocumentInput("one", first), DocumentInput("two", second))
                )

            self.assertFalse(store.exists())


if __name__ == "__main__":
    unittest.main()
