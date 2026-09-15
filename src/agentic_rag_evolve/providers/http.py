"""HTTP provider transports kept outside the evolvable DeepRead source."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import requests
import tiktoken

from agentic_rag_evolve.telemetry.context import emit_trace


def _response_error_summary(response: requests.Response) -> str:
    """Return bounded provider diagnostics without echoing request data."""

    parts = [f"HTTP {response.status_code}"]
    try:
        body = response.json()
    except (ValueError, requests.JSONDecodeError):
        body = None
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            message = error.get("message")
        else:
            code = body.get("code")
            message = body.get("message")
        if code:
            parts.append(f"code={str(code)[:120]}")
        if message:
            parts.append(f"message={str(message)[:500]}")
    request_id = response.headers.get("x-request-id") or response.headers.get(
        "x-tt-logid"
    )
    if request_id:
        parts.append(f"request_id={request_id[:160]}")
    return "; ".join(parts)


def _count_tokens(value: Any) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(value)))


@dataclass(slots=True)
class OpenAICompatibleChatModel:
    model_name: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120
    max_retries: int = 5
    default_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.model_name.strip():
            raise ValueError("chat model_name must not be empty")
        if not self.api_key:
            raise ValueError("chat api_key must not be empty")
        if self.timeout < 1:
            raise ValueError("chat timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("chat max_retries must be non-negative")

    def complete(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}/chat/completions"
        request_payload = dict(payload)
        request_payload["model"] = self.model_name
        request_payload["stream"] = False
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **dict(self.default_headers),
        }

        max_attempts = 1 + self.max_retries
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(min(90, 1.5 * (2 ** (attempt - 2))))
            emit_trace(
                "llm_http_attempt",
                attempt=attempt,
                model=self.model_name,
            )
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_payload,
                    timeout=self.timeout,
                )
                status = response.status_code
                retryable = status in {429, 500, 502, 503, 504}
                if retryable and attempt < max_attempts:
                    emit_trace(
                        "llm_http_error",
                        status_code=status,
                        error=f"HTTP {status}",
                        attempt=attempt,
                        will_retry=True,
                    )
                    continue
                if status >= 400:
                    raise requests.HTTPError(
                        _response_error_summary(response), response=response
                    )
                result = response.json()
                emit_trace(
                    "llm_http_success",
                    attempt=attempt,
                    status_code=status,
                )
                emit_trace(
                    "llm_token_debug",
                    input_tokens=_count_tokens(request_payload.get("messages", [])),
                    output_tokens=_count_tokens(result.get("choices", [])),
                )
                return result
            except requests.RequestException as exc:
                will_retry = attempt < max_attempts
                emit_trace(
                    "llm_http_error",
                    error=str(exc),
                    attempt=attempt,
                    will_retry=will_retry,
                )
                if not will_retry:
                    raise

        raise RuntimeError("chat completion exhausted retries")


@dataclass(slots=True)
class OpenAICompatibleReranker:
    model_name: str
    api_key: str
    base_url: str
    timeout: int = 120

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.model_name.strip():
            raise ValueError("reranker model_name must not be empty")
        if not self.api_key:
            raise ValueError("reranker api_key must not be empty")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int = -1,
    ) -> Mapping[str, Any]:
        response = requests.post(
            f"{self.base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
                "return_documents": True,
                "max_chunks_per_doc": 1024,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
