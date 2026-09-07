"""DeepRead v0 target system and its standalone runtime adapter."""

from .ingestion import IngestedDocument, MarkdownIngestor
from .DeepRead.ports import EmbeddingModel
from .runtime import DeepReadConfig, DeepReadQueryResult, GlobalDeepReadRuntime

__all__ = [
    "DeepReadConfig",
    "DeepReadQueryResult",
    "EmbeddingModel",
    "GlobalDeepReadRuntime",
    "IngestedDocument",
    "MarkdownIngestor",
]
