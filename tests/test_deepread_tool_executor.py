import unittest
from unittest.mock import MagicMock, sentinel

from systems.deepread.DeepRead.tool import DeepReadToolExecutor


class DeepReadToolExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = MagicMock()
        self.index.neighbor_window = (9, 9)
        for method in (
            "get_doc_structure",
            "read_section",
            "bm25_search",
            "regex_search",
            "vector_search",
            "hybrid_search",
            "semantic_retrieval",
        ):
            getattr(self.index, method).return_value = {"ok": True, "method": method}

    def test_structure_and_read_arguments_match_baseline(self) -> None:
        executor = DeepReadToolExecutor(self.index, enable_multimodal=True)

        executor.execute("get_doc_structure", {"doc_id": [5, "7"]})
        executor.execute("read_section", {"doc_id": "5", "node_id": "2"})

        self.index.get_doc_structure.assert_called_once_with(doc_ids=["5", "7"])
        self.index.read_section.assert_called_once_with(
            doc_id="5",
            node_id="2",
            start_paragraph=0,
            end_paragraph=-1,
            include_images=True,
        )

    def test_retrieval_configuration_and_providers_are_forwarded(self) -> None:
        executor = DeepReadToolExecutor(
            self.index,
            embedding_model=sentinel.embedding,
            reranker=sentinel.reranker,
            neighbor_window=(1, -1),
            vector_topk=5,
            hybrid_topk=6,
            hybrid_topk_bm25=31,
            hybrid_topk_vec=32,
            hybrid_bm25_weight=0.4,
            hybrid_vector_weight=0.6,
            semantic_stage1_method="hybrid",
            semantic_topk1=33,
            semantic_topk2=2,
            semantic_stage1_hybrid_topk_bm25=34,
            semantic_stage1_hybrid_topk_vec=35,
        )

        executor.execute("vector_search", {"query": "q", "doc_id": "1"})
        executor.execute("hybrid_search", {"query": "q", "scope": "full"})
        executor.execute("semantic_retrieval", {"query": "q", "scope": "full"})

        self.index.vector_search.assert_called_once_with(
            query="q",
            scope="full",
            doc_id="1",
            top_k=5,
            include_images=False,
            embedding_model=sentinel.embedding,
            neighbor_window=(1, -1),
        )
        self.index.hybrid_search.assert_called_once_with(
            query="q",
            scope="full",
            doc_id=None,
            top_k=6,
            bm25_weight=0.4,
            vector_weight=0.6,
            top_k_bm25=31,
            top_k_vec=32,
            include_images=False,
            embedding_model=sentinel.embedding,
            neighbor_window=(1, -1),
        )
        self.index.semantic_retrieval.assert_called_once_with(
            query="q",
            scope="full",
            doc_id=None,
            stage1_method="hybrid",
            top_k1=33,
            top_k2=2,
            stage1_hybrid_topk_bm25=34,
            stage1_hybrid_topk_vec=35,
            include_images=False,
            embedding_model=sentinel.embedding,
            reranker=sentinel.reranker,
            neighbor_window=(1, -1),
            hybrid_bm25_weight=0.4,
            hybrid_vector_weight=0.6,
        )

    def test_unknown_tool_is_a_model_visible_error(self) -> None:
        result = DeepReadToolExecutor(self.index).execute("missing", {})
        self.assertEqual(result, {"ok": False, "error": "Tool 'missing' not implemented"})


if __name__ == "__main__":
    unittest.main()
