import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evaluation.evaluator import (
    EvaluationSummary,
    evaluate_financebench,
    write_evaluation,
)


class EvaluatorTest(unittest.TestCase):
    def test_evaluates_only_predictions_and_keeps_gold_out_of_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps({
                    "financebench_id": "q1",
                    "doc_name": "report",
                    "question_type": "domain-relevant",
                    "question": "What was revenue?",
                    "answer": "Revenue was 12 million.",
                    "evidence": [{"evidence_text": "Revenue was 12 million."}],
                }) + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({
                    "task_id": "q1",
                    "status": "ok",
                    "answer": "Revenue was 12 million.",
                    "retrieved_texts": ["Revenue was 12 million."],
                }) + "\n",
                encoding="utf-8",
            )

            details, summary = evaluate_financebench(
                dataset_path=dataset,
                prediction_path=predictions,
            )

            self.assertEqual(summary.total, 1)
            self.assertEqual(summary.average_f1, 1.0)
            self.assertEqual(summary.average_recall, 1.0)
            self.assertIsNone(summary.average_accuracy_0_4)
            self.assertEqual(details[0]["judge"]["status"], "skipped")

    def test_rejects_prediction_not_present_in_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps({"financebench_id": "q1", "doc_name": "r", "question": "Q", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"task_id": "q2", "answer": "A"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not found"):
                evaluate_financebench(dataset_path=dataset, prediction_path=predictions)

    def test_refuses_to_overwrite_evaluation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "existing.json").write_text("{}", encoding="utf-8")
            summary = EvaluationSummary(0, 0, 0, 0, 0, 0, 0.0, 0.0, None, None)
            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                write_evaluation(output, (), summary)

    def test_rejects_question_mismatch_for_same_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps({"financebench_id": "q1", "doc_name": "r", "question": "Expected?", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"task_id": "q1", "question": "Different?", "answer": "A"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "question mismatch"):
                evaluate_financebench(dataset_path=dataset, prediction_path=predictions)


if __name__ == "__main__":
    unittest.main()
