import unittest
from unittest.mock import Mock, patch

from agentic_rag_evolve.providers import load_chat_model


class ProviderConfigTest(unittest.TestCase):
    def test_judge_can_load_chat_without_embedding_configuration(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL": "judge-v1",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_API_KEY": "test-only",
            },
            clear=True,
        ):
            model = load_chat_model(timeout=300, max_retries=0)
        self.assertEqual(model.model_name, "judge-v1")
        self.assertEqual(model.timeout, 300)
        self.assertEqual(model.max_retries, 0)

    def test_zero_retries_still_makes_one_request(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL": "judge-v1",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_API_KEY": "test-only",
            },
            clear=True,
        ):
            model = load_chat_model(max_retries=0)
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        with patch("agentic_rag_evolve.providers.http.requests.post", return_value=response) as post:
            result = model.complete({"messages": []})

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
