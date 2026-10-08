"""Environment-backed provider configuration for stable runners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values
from systems.deepread.DeepRead.ports import ChatModel, EmbeddingModel, Reranker

from .http import OpenAICompatibleChatModel, OpenAICompatibleReranker
from .volcengine import VolcengineMultimodalEmbeddingModel


def provider_environment(env_file: Path | None = None) -> dict[str, str]:
    """Read settings without leaking one env file into subsequent provider loads."""
    values = dotenv_values(env_file) if env_file is not None else {}
    return {key: (str(value).strip() if key.startswith(("LLM_", "DEEPREAD_LLM_", "EVOLUTION_LLM_",
                                                       "EMBEDDING_", "RERANK_")) else str(value))
            for key, value in {**values, **os.environ}.items()
            if value is not None}


def _required(name: str, values: Mapping[str, str]) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required provider setting: {name}")
    return value


def llm_environment(values: Mapping[str, str], role: str) -> dict[str, str]:
    """A role-specific endpoint is atomic; never borrow a different endpoint's key."""
    if role not in {"deepread", "evolution"}:
        raise ValueError(f"unknown LLM role: {role}")
    prefix = f"{role.upper()}_LLM_"
    fields = ("MODEL", "BASE_URL", "API_KEY")
    explicit = any(values.get(prefix + field) for field in fields)
    if explicit:
        for field in fields:
            _required(prefix + field, values)
    selected = prefix if explicit else "LLM_"
    result = {field: values.get(selected + field, "") for field in fields}
    # The target keeps its frozen transport defaults. This budget is framework-only.
    result["MAX_OUTPUT_TOKENS"] = values.get(prefix + "MAX_OUTPUT_TOKENS") or (
        values.get("LLM_MAX_OUTPUT_TOKENS") if not explicit else None
    ) or "65536"
    return result


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_dimension: int = 2048
    rerank_model: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "ProviderSettings":
        values = provider_environment(env_file)
        llm = llm_environment(values, "deepread")
        return cls(
            llm_model=_required("MODEL", llm),
            llm_base_url=_required("BASE_URL", llm),
            llm_api_key=_required("API_KEY", llm),
            embedding_model=_required("EMBEDDING_MODEL_NAME", values),
            embedding_base_url=_required("EMBEDDING_BASE_URL", values),
            embedding_api_key=_required("EMBEDDING_API_KEY", values),
            embedding_dimension=int(values.get("EMBEDDING_DIMENSION", "2048")),
            rerank_model=values.get("RERANK_MODEL") or None,
            rerank_base_url=values.get("RERANK_BASE_URL") or None,
            rerank_api_key=values.get("RERANK_API_KEY") or None,
        )


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    chat: ChatModel
    embedding: EmbeddingModel
    reranker: Reranker | None = None


def load_provider_bundle(env_file: Path | None = None) -> ProviderBundle:
    settings = ProviderSettings.from_env(env_file)
    reranker = None
    if settings.rerank_model and settings.rerank_base_url and settings.rerank_api_key:
        reranker = OpenAICompatibleReranker(
            model_name=settings.rerank_model,
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key,
        )
    return ProviderBundle(
        chat=OpenAICompatibleChatModel(
            model_name=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
        embedding=VolcengineMultimodalEmbeddingModel(
            model_name=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=settings.embedding_dimension,
        ),
        reranker=reranker,
    )


def load_chat_model(
    env_file: Path | None = None,
    *,
    timeout: int = 120,
    max_retries: int = 5,
    retry_base_seconds: float = 1.5,
    retry_max_seconds: float = 90.0,
) -> OpenAICompatibleChatModel:
    settings = llm_environment(provider_environment(env_file), "evolution")
    return OpenAICompatibleChatModel(
        model_name=_required("MODEL", settings),
        base_url=_required("BASE_URL", settings),
        api_key=_required("API_KEY", settings),
        timeout=timeout,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        default_max_output_tokens=int(settings["MAX_OUTPUT_TOKENS"]),
    )


def load_embedding_model(env_file: Path | None = None) -> VolcengineMultimodalEmbeddingModel:
    values = provider_environment(env_file)
    return VolcengineMultimodalEmbeddingModel(
        model_name=_required("EMBEDDING_MODEL_NAME", values),
        base_url=_required("EMBEDDING_BASE_URL", values),
        api_key=_required("EMBEDDING_API_KEY", values),
        dimension=int(values.get("EMBEDDING_DIMENSION", "2048")),
    )
