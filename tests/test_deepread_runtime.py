import json
import tempfile
import unittest
from pathlib import Path

from systems.deepread.runtime import DeepReadConfig, GlobalDeepReadRuntime


def _write_corpus(path: Path, title: str) -> None:
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "0",
                        "title": title,
                        "paragraphs": [f"Evidence from {title}."],
                        "children": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class DeepReadRuntimeTest(unittest.TestCase):
    def _config(self) -> DeepReadConfig:
        return DeepReadConfig()

    def test_discovers_only_flat_corpora_in_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            _write_corpus(store / "b_corpus.json", "B")
            _write_corpus(store / "a_corpus.json", "A")
            nested = store / "nested"
            nested.mkdir()
            _write_corpus(nested / "ignored_corpus.json", "ignored")
            runtime = GlobalDeepReadRuntime(store, store / "trace.jsonl", self._config())

            paths = runtime.discover_corpus_paths()
            index = runtime.load_index()

            self.assertEqual([path.name for path in paths], ["a_corpus.json", "b_corpus.json"])
            self.assertEqual(index.doc_id_map, {"1": "a", "2": "b"})

    def test_index_is_cached_until_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            _write_corpus(store / "a_corpus.json", "A")
            runtime = GlobalDeepReadRuntime(store, store / "trace.jsonl", self._config())

            first = runtime.load_index()
            second = runtime.load_index()
            runtime.invalidate_index()
            third = runtime.load_index()

            self.assertIs(first, second)
            self.assertIsNot(first, third)

    def test_missing_store_has_actionable_error(self) -> None:
        runtime = GlobalDeepReadRuntime(
            Path("does-not-exist"), Path("trace.jsonl"), self._config()
        )

        with self.assertRaisesRegex(FileNotFoundError, "store directory"):
            runtime.discover_corpus_paths()


if __name__ == "__main__":
    unittest.main()
