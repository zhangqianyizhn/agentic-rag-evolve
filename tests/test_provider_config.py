import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agentic_rag_evolve.providers import load_chat_model


class ProviderConfigTest(unittest.TestCase):
    def test_two_roles_use_separate_endpoints_credentials_and_models(self) -> None:
        from agentic_rag_evolve.providers import load_provider_bundle
        with patch.dict(os.environ, {
            "DEEPREAD_LLM_MODEL": "target-small",
            "DEEPREAD_LLM_BASE_URL": "https://target.invalid/v1",
            "DEEPREAD_LLM_API_KEY": "target-key",
            "EVOLUTION_LLM_MODEL": "framework-strong",
            "EVOLUTION_LLM_BASE_URL": "https://framework.invalid/v1",
            "EVOLUTION_LLM_API_KEY": "framework-key",
            "EVOLUTION_LLM_MAX_OUTPUT_TOKENS": "98304",
            "EMBEDDING_MODEL_NAME": "embedding", "EMBEDDING_BASE_URL": "https://embed.invalid/v1",
            "EMBEDDING_API_KEY": "embed-key",
            "LLM_MODEL": "ignored", "LLM_BASE_URL": "https://ignored.invalid/v1",
            "LLM_API_KEY": "ignored-key", "LLM_MAX_OUTPUT_TOKENS": "1234",
        }, clear=True), patch("agentic_rag_evolve.providers.config.VolcengineMultimodalEmbeddingModel"):
            target = load_provider_bundle().chat
            framework = load_chat_model()
        self.assertEqual((target.model_name, target.base_url, target.api_key),
                         ("target-small", "https://target.invalid/v1", "target-key"))
        self.assertEqual((framework.model_name, framework.base_url, framework.api_key),
                         ("framework-strong", "https://framework.invalid/v1", "framework-key"))
        self.assertIsNone(target.default_max_output_tokens)
        self.assertEqual(framework.default_max_output_tokens, 98304)

    def test_partial_role_configuration_cannot_borrow_legacy_credentials(self) -> None:
        from agentic_rag_evolve.providers import ProviderSettings
        for role, loader in (("DEEPREAD", ProviderSettings.from_env), ("EVOLUTION", load_chat_model)):
            with self.subTest(role=role), patch.dict(os.environ, {
                f"{role}_LLM_MODEL": "new-model", "LLM_MODEL": "old-model",
                "LLM_BASE_URL": "https://old.invalid/v1", "LLM_API_KEY": "old-key",
            }, clear=True):
                with self.assertRaisesRegex(ValueError, f"{role}_LLM_BASE_URL"):
                    loader()

    def test_framework_override_keeps_legacy_target_and_independent_output_default(self) -> None:
        from agentic_rag_evolve.providers.config import llm_environment, provider_environment
        with patch.dict(os.environ, {
            "LLM_MODEL": "baseline", "LLM_BASE_URL": "https://old.invalid/v1", "LLM_API_KEY": "old-key",
            "LLM_MAX_OUTPUT_TOKENS": "1234", "EVOLUTION_LLM_MODEL": "strong",
            "EVOLUTION_LLM_BASE_URL": "https://new.invalid/v1", "EVOLUTION_LLM_API_KEY": "new-key",
        }, clear=True):
            self.assertEqual(llm_environment(provider_environment(), "deepread")["MODEL"], "baseline")
            self.assertEqual(load_chat_model().default_max_output_tokens, 65536)

    def test_loading_multiple_env_files_does_not_reuse_first_files_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            for name in ("first", "second"):
                path = Path(directory) / f"{name}.env"
                path.write_text(f"EVOLUTION_LLM_MODEL={name}\nEVOLUTION_LLM_BASE_URL=https://{name}.invalid/v1\n"
                                f"EVOLUTION_LLM_API_KEY={name}-key\n")
                self.assertEqual(load_chat_model(path).model_name, name)
            self.assertNotIn("EVOLUTION_LLM_MODEL", os.environ)
            with patch.dict(os.environ, {"EVOLUTION_LLM_MODEL": "shell-override"}):
                self.assertEqual(load_chat_model(path).model_name, "shell-override")

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
