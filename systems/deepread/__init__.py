"""DeepRead v0 target system and its standalone runtime adapter."""

from .ingestion import EmbeddingProvider, IngestedDocument, MarkdownIngestor
from .providers import VolcengineMultimodalEmbeddingProvider
from .runtime import DeepReadConfig, DeepReadQueryResult, GlobalDeepReadRuntime

__all__ = [
    "DeepReadConfig",
    "DeepReadQueryResult",
    "EmbeddingProvider",
    "GlobalDeepReadRuntime",
    "IngestedDocument",
    "MarkdownIngestor",
    "VolcengineMultimodalEmbeddingProvider",
]
