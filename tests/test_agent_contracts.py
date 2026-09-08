import unittest

from agentic_rag_evolve.telemetry import TracingToolExecutor
from systems.deepread.DeepRead.agent import AgentOutcome, run_agent
from systems.deepread.DeepRead.tool import DeepReadToolExecutor
from systems.deepread.DeepRead.tool.retrieval import DocIndex


class RecordingObserver:
    def __init__(self) -> None:
        self.recoveries = []
        self.invalid_arguments = []
        self.empty_responses = []

    def tool_calls_recovered(self, **fields):
        self.recoveries.append(fields)

    def tool_arguments_invalid(self, **fields):
        self.invalid_arguments.append(fields)

    def model_response_empty(self, **fields):
        self.empty_responses.append(fields)


class EmptyReasoningModel:
    model_name = "fake"
    base_url = "local://fake"

    def complete(self, payload):
        return {"choices": [{"message": {"content": "", "reasoning_content": "still thinking"}}]}


class RecoveredToolModel:
    model_name = "fake"
    base_url = "local://fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, payload):
        self.calls += 1
        if self.calls == 1:
            message = {"content": 'bm25_search {"query":"text","scope":"full"}'}
        else:
            message = {"content": "answer"}
        return {"choices": [{"message": message}]}


class InvalidArgumentsModel:
    model_name = "fake"
    base_url = "local://fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, payload):
        self.calls += 1
        if self.calls == 1:
            message = {
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "bm25_search", "arguments": "{bad-json"},
                }],
            }
        else:
            message = {"content": "answer"}
        return {"choices": [{"message": message}]}


class AgentContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DocIndex(
            [{"id": "n1", "doc_id": "1", "title": "Title", "paragraphs": ["text"], "children": []}],
            neighbor_window=None,
        )
        self.executor = TracingToolExecutor(DeepReadToolExecutor(self.index))

    def test_max_rounds_is_an_explicit_outcome(self) -> None:
        observer = RecordingObserver()
        outcome = run_agent(
            chat_model=EmptyReasoningModel(),
            doc_index=self.index,
            tool_executor=self.executor,
            user_question="question",
            observer=observer,
            max_rounds=2,
        )

        self.assertEqual(
            outcome,
            AgentOutcome(
                answer="(Reached maximum rounds, no final answer generated)",
                termination_reason="max_rounds",
                rounds_completed=2,
            ),
        )
        self.assertEqual(len(observer.empty_responses), 2)
        self.assertEqual(observer.empty_responses[0]["reasoning_preview"], "still thinking")

    def test_recovery_is_reported_through_typed_observer(self) -> None:
        observer = RecordingObserver()
        outcome = run_agent(
            chat_model=RecoveredToolModel(),
            doc_index=self.index,
            tool_executor=self.executor,
            user_question="question",
            observer=observer,
            max_rounds=2,
        )

        self.assertEqual(outcome.answer, "answer")
        self.assertEqual(observer.recoveries[0]["recovery_kind"], "inline_json")
        self.assertEqual(
            observer.recoveries[0]["tool_calls"][0]["function"]["name"],
            "bm25_search",
        )

    def test_invalid_arguments_are_reported_without_logger_dependency(self) -> None:
        observer = RecordingObserver()
        outcome = run_agent(
            chat_model=InvalidArgumentsModel(),
            doc_index=self.index,
            tool_executor=self.executor,
            user_question="question",
            observer=observer,
            max_rounds=2,
        )

        self.assertEqual(outcome.termination_reason, "final_answer")
        self.assertEqual(observer.invalid_arguments[0]["call_id"], "c1")
        self.assertEqual(observer.invalid_arguments[0]["tool_name"], "bm25_search")


if __name__ == "__main__":
    unittest.main()
