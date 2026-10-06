import unittest
from unittest.mock import Mock, patch

import requests

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
        self.assertEqual(model.default_max_output_tokens, 65536)

    def test_framework_budget_is_explicit_and_caller_limit_takes_precedence(self) -> None:
        with patch.dict("os.environ", {
            "LLM_MODEL": "chat", "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_API_KEY": "test", "LLM_MAX_OUTPUT_TOKENS": "32768",
        }, clear=True):
            model = load_chat_model(max_retries=0)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("agentic_rag_evolve.providers.http._count_tokens", return_value=0), patch(
            "agentic_rag_evolve.providers.http.requests.post", return_value=response
        ) as post:
            original = {"messages": []}
            model.complete(original)
            self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 32768)
            self.assertNotIn("max_tokens", original)
            model.complete({"messages": [], "max_tokens": 8192})
            self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 8192)
            model.complete({"messages": [], "max_completion_tokens": 4096})
            self.assertNotIn("max_tokens", post.call_args.kwargs["json"])

    def test_frozen_baseline_provider_keeps_original_output_settings(self) -> None:
        from agentic_rag_evolve.providers import load_provider_bundle
        with patch.dict("os.environ", {
            "LLM_MODEL": "chat", "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_API_KEY": "test", "LLM_MAX_OUTPUT_TOKENS": "32768",
            "EMBEDDING_MODEL_NAME": "embedding", "EMBEDDING_BASE_URL": "https://example.invalid/v1",
            "EMBEDDING_API_KEY": "test",
        }, clear=True), patch("agentic_rag_evolve.providers.config.VolcengineMultimodalEmbeddingModel"):
            bundle = load_provider_bundle()
        self.assertIsNone(bundle.chat.default_max_output_tokens)

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

    def test_http_error_includes_bounded_provider_diagnostics(self) -> None:
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
        response = Mock(status_code=429)
        response.json.return_value = {
            "error": {"code": "rate_limit", "message": "Request quota exhausted."}
        }
        response.headers = {"x-request-id": "request-123"}

        with patch(
            "agentic_rag_evolve.providers.http.requests.post",
            return_value=response,
        ):
            with self.assertRaisesRegex(
                requests.HTTPError,
                "HTTP 429; code=rate_limit; message=Request quota exhausted",
            ):
                model.complete({"messages": []})

    def test_retryable_response_honors_retry_after_and_records_attempts(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL": "judge-v1",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_API_KEY": "test-only",
            },
            clear=True,
        ):
            model = load_chat_model(
                max_retries=1,
                retry_base_seconds=15,
                retry_max_seconds=120,
            )
        limited = Mock(status_code=429)
        limited.headers = {"Retry-After": "7"}
        success = Mock(status_code=200)
        success.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }

        with patch(
            "agentic_rag_evolve.providers.http.requests.post",
            side_effect=[limited, success],
        ) as post, patch("agentic_rag_evolve.providers.http.time.sleep") as sleep:
            result = model.complete({"messages": []})

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(7.0)
        self.assertEqual(model.last_attempts, 2)
        self.assertEqual(model.last_retry_delays, [7.0])


if __name__ == "__main__":
    unittest.main()
