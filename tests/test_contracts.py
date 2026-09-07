import unittest
from pathlib import Path

from agentic_rag_evolve.contracts import (
    DocumentInput,
    DocumentQATask,
    EvaluationReference,
)


class ContractsTest(unittest.TestCase):
    def test_runner_task_excludes_evaluation_labels(self) -> None:
        task = DocumentQATask(
            task_id="financebench-1",
            question="What was revenue?",
            documents=(DocumentInput("report", Path("report.md")),),
        )

        serialized = task.to_dict()

        self.assertNotIn("gold_answers", serialized)
        self.assertNotIn("gold_evidence", serialized)
        self.assertEqual(serialized["documents"][0]["path"], "report.md")

    def test_task_rejects_duplicate_document_ids(self) -> None:
        documents = (
            DocumentInput("report", Path("first.md")),
            DocumentInput("report", Path("second.md")),
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            DocumentQATask("task", "question", documents)

    def test_evaluation_reference_is_separate_and_serializable(self) -> None:
        reference = EvaluationReference(
            task_id="financebench-1",
            gold_answers=("42",),
            gold_evidence=("The reported value was 42.", ""),
        )

        self.assertEqual(reference.to_dict()["gold_evidence"], ["The reported value was 42."])


if __name__ == "__main__":
    unittest.main()
