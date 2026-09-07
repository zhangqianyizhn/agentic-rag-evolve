import unittest

from agentic_rag_evolve.evaluation.metrics import evidence_recall, normalize_answer, token_f1


class EvaluationMetricsTest(unittest.TestCase):
    def test_normalization_matches_historical_rules(self) -> None:
        self.assertEqual(normalize_answer("The Revenue, and Profit."), "revenue profit")

    def test_token_f1_uses_multiset_overlap(self) -> None:
        self.assertAlmostEqual(token_f1("revenue revenue profit", "revenue profit"), 0.8)

    def test_evidence_recall_reports_exact_soft_and_short_misses(self) -> None:
        score, matches = evidence_recall(
            ["Revenue increased from 10 to 12 million in fiscal 2022."],
            [
                "Revenue increased from 10 to 12 million",
                "Revenue increased from 10 to 12 million during fiscal 2022.",
                "ID 7",
            ],
            soft_threshold=0.7,
        )
        self.assertAlmostEqual(score, 2 / 3)
        self.assertEqual(matches[0].method, "exact_substring")
        self.assertEqual(matches[1].method, "soft_token_coverage")
        self.assertEqual(matches[2].method, "short_evidence_exact_required")


if __name__ == "__main__":
    unittest.main()
