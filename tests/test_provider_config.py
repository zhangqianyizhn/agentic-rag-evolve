import unittest
from unittest.mock import patch

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
            model = load_chat_model()
        self.assertEqual(model.model_name, "judge-v1")


if __name__ == "__main__":
    unittest.main()
