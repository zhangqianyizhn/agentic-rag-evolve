import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.deepread_runner import load_financebench_queries, run_financebench
from agentic_rag_evolve.providers import ProviderBundle
from systems.deepread.runtime import DeepReadConfig


class FakeChat:
    model_name = "fake-chat"
    base_url = "local://fake"

    def complete(self, payload, *, logger=None, query_id=""):
        return {
            "choices": [{"message": {"content": "final", "tool_calls": None}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }


class FakeEmbedding:
    model_name = "fake-embedding"
    base_url = "local://fake"
    normalized = True

    def embed(self, text):
        return [1.0]


class DeepReadRunnerTest(unittest.TestCase):
    def test_financebench_adapter_does_not_copy_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.jsonl"
            dataset.write_text(
                json.dumps({
                    "financebench_id": "q1",
                    "doc_name": "report",
                    "question": "What happened?",
                    "answer": "secret gold",
                    "evidence": [{"evidence_text": "secret evidence"}],
                }) + "\n",
                encoding="utf-8",
            )
            (query,) = load_financebench_queries(dataset)
            self.assertEqual(query.task_id, "q1")
            self.assertNotIn("secret", repr(query))

    def test_runner_writes_stable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps({"financebench_id": "q1", "doc_name": "report", "question": "Question?"}) + "\n",
                encoding="utf-8",
            )
            store = root / "store"
            store.mkdir()
            (store / "report_corpus.json").write_text(
                json.dumps({"nodes": [{"id": "0", "title": "Report", "paragraphs": ["Evidence"], "children": []}]}),
                encoding="utf-8",
            )
            output = root / "run"

            summary = run_financebench(
                dataset_path=dataset,
                store_path=store,
                output_path=output,
                providers=ProviderBundle(chat=FakeChat(), embedding=FakeEmbedding()),
                config=DeepReadConfig(max_rounds=1),
            )

            self.assertEqual(summary, {"query_count": 1, "completed": 1, "failed": 0})
            prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prediction["answer"], "final")
            manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), manifest_text)
            self.assertIn("dataset_sha256", manifest_text)
            self.assertTrue((output / "deepread_trace.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
