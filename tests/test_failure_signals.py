import unittest

from agentic_rag_evolve.diagnostics import analyze_failure_signals


class FailureSignalsTest(unittest.TestCase):
    def _analyze(
        self,
        *,
        status="ok",
        answer="answer",
        judge_status="skipped",
        judge_score=None,
        baseline_recall=1.0,
        path="fully_covered",
        candidate_recall=1.0,
    ):
        return analyze_failure_signals(
            evaluation={
                "prediction": {"status": status, "answer": answer},
                "metrics": {"recall": baseline_recall},
                "judge": {"status": judge_status, "score": judge_score},
            },
            trajectory={"task_id": "q1", "status": status, "answer": answer},
            evidence_coverage={
                "primary_signal": path,
                "layers": {
                    "corpus": {"recall": 1.0},
                    "candidate": {"recall": candidate_recall},
                    "read": {"recall": 1.0},
                    "answer": {"recall": 1.0},
                },
            },
        )

    def test_execution_and_empty_answer_are_distinct(self) -> None:
        execution = self._analyze(status="error", answer="")
        empty = self._analyze(answer="")

        self.assertEqual(execution["triage"], "execution_failure")
        self.assertTrue(execution["diagnosis_route"]["eligible"])
        self.assertEqual(execution["diagnosis_route"]["target"], "deepread")
        self.assertEqual(empty["triage"], "no_answer")

    def test_judge_outcomes_drive_answer_triage(self) -> None:
        incorrect = self._analyze(judge_status="ok", judge_score=1)
        partial = self._analyze(judge_status="ok", judge_score=3)
        correct = self._analyze(judge_status="ok", judge_score=4)

        self.assertEqual(incorrect["triage"], "incorrect_answer")
        self.assertEqual(partial["triage"], "partial_answer")
        self.assertEqual(correct["triage"], "pass")
        self.assertFalse(correct["is_bad_case"])

    def test_evaluator_disagreement_is_not_called_answer_error(self) -> None:
        result = self._analyze(
            baseline_recall=0.0,
            candidate_recall=1.0,
            path="read",
        )

        self.assertEqual(result["triage"], "evaluation_suspicious")
        self.assertFalse(result["diagnosis_route"]["eligible"])
        self.assertEqual(result["diagnosis_route"]["target"], "evaluation_review")
        self.assertEqual(
            [signal["code"] for signal in result["signals"]],
            [
                "evidence_not_read",
                "baseline_recall_disagreement",
                "answer_correctness_unavailable",
            ],
        )

    def test_correct_judge_does_not_hide_evaluator_disagreement(self) -> None:
        result = self._analyze(
            judge_status="ok",
            judge_score=4,
            baseline_recall=0.0,
            candidate_recall=1.0,
            path="read",
        )

        self.assertEqual(result["triage"], "evaluation_suspicious")
        self.assertTrue(result["is_bad_case"])
        self.assertIn("judged_correct", {item["code"] for item in result["signals"]})

    def test_incorrect_judge_outcome_takes_priority_over_metric_disagreement(self) -> None:
        result = self._analyze(
            judge_status="ok",
            judge_score=0,
            baseline_recall=0.0,
            candidate_recall=1.0,
        )

        self.assertEqual(result["triage"], "incorrect_answer")

    def test_missing_judgment_remains_explicit(self) -> None:
        result = self._analyze()

        self.assertEqual(result["triage"], "needs_judgment")
        self.assertEqual(result["confidence"], "medium")

    def test_gold_evidence_missing_from_corpus_requires_data_review(self) -> None:
        result = self._analyze(path="corpus", candidate_recall=0.0)

        self.assertEqual(result["triage"], "evaluation_suspicious")
        self.assertEqual(result["signals"][0]["code"], "gold_evidence_missing_from_corpus")


if __name__ == "__main__":
    unittest.main()
