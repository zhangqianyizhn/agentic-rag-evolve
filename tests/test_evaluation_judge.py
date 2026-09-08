import unittest

from agentic_rag_evolve.evaluation.judge import judge_answer


class FakeJudge:
    model_name = "judge-v1"

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error

    def complete(self, payload):
        if self.error:
            raise self.error
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }


class EvaluationJudgeTest(unittest.TestCase):
    def test_parses_fenced_json_and_clamps_score(self) -> None:
        result = judge_answer(
            FakeJudge('```json\n{"score": 9, "reasoning": "Correct."}\n```'),
            task_id="q1",
            question="Q?",
            gold_answers=["A"],
            answer="A",
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.score, 4)
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(result.output_tokens, 3)

    def test_keeps_judge_failure_distinct_from_zero_score(self) -> None:
        result = judge_answer(
            FakeJudge(error=RuntimeError("offline")),
            task_id="q1",
            question="Q?",
            gold_answers=["A"],
            answer="A",
        )
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.score)
        self.assertIn("RuntimeError", result.error or "")


if __name__ == "__main__":
    unittest.main()
