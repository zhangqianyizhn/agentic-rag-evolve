import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evaluation.comparison import compare_with_historical


class EvaluationComparisonTest(unittest.TestCase):
    def test_pairs_by_question_and_reports_metric_deltas(self) -> None:
        current = [{
            "task_id": "q1",
            "question": "Question?",
            "metrics": {"f1": 0.7, "recall": 1.0, "accuracy_0_4": 4},
        }]
        with tempfile.TemporaryDirectory() as directory:
            historical = Path(directory) / "historical.json"
            historical.write_text(
                json.dumps({"results": [{
                    "question": "Question?",
                    "metrics": {"F1": 0.5, "Recall": 0.0, "Accuracy": 3},
                }]}),
                encoding="utf-8",
            )
            result = compare_with_historical(current, historical)

        self.assertEqual(result["matched"], 1)
        self.assertAlmostEqual(result["pairs"][0]["metrics"]["f1"]["delta"], 0.2)
        self.assertEqual(result["pairs"][0]["metrics"]["accuracy_0_4"]["delta"], 1)


if __name__ == "__main__":
    unittest.main()
