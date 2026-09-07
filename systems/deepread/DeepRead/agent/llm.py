from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests
import tiktoken
from src.core.token_tracer_util import token_tracker as _token_tracker

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def should_sanitize_for_vllm(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    u = base_url.lower()
    if "openrouter.ai" in u:
        return False
    if "api.openai.com" in u:
        return False
    if "localhost" in u or "127.0.0.1" in u or "vllm" in u:
        return True
    return False

def sanitize_for_vllm(payload: Dict[str, Any], allow_tools: bool = True) -> Dict[str, Any]:
    p = dict(payload)
    for k in ("include_reasoning", "reasoning", "parallel_tool_calls", "response_format", "modalities", "audio", "vision", "metadata"):
        p.pop(k, None)

    if not allow_tools:
        p.pop("tools", None)
        p.pop("tool_choice", None)

    cleaned_msgs: List[Dict[str, Any]] = []
    for m in p.get("messages", []):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        content = m.get("content", "")
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            content = "\n".join(t for t in texts if t)
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        m2: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and allow_tools and m.get("tool_calls"):
            m2["tool_calls"] = m["tool_calls"]
        if role == "tool" and m.get("tool_call_id"):
            m2["tool_call_id"] = m["tool_call_id"]

        cleaned_msgs.append(m2)

    p["messages"] = cleaned_msgs
    return p


def _preview_messages(messages: List[Dict[str, Any]]) -> str:
    try:
        return json.dumps([m for m in messages], ensure_ascii=False)
    except Exception:
        return "<unserializable messages>"


def _preview_tool_calls(tool_calls: Any) -> Any:
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        out.append({"id": tc.get("id"), "name": (tc.get("function") or {}).get("name")})
    return out


def http_chat_completions(
    api_key: Optional[str],
    base_url: Optional[str],
    payload: Dict[str, Any],
    default_headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    logger: Optional["JsonlLogger"] = None,
    query_id: Optional[str] = "",
) -> Dict[str, Any]:
    if not api_key:
        err = RuntimeError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY")
        if logger:
            logger.log("llm_http_error", query_id=query_id, error=str(err))
        raise err

    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload_copy = dict(payload)
    payload_copy["stream"] = False

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if default_headers:
        headers.update(default_headers)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_s = min(90, 1.5 * (2 ** (attempt - 1)))
                time.sleep(wait_s)

            if logger:
                logger.log("llm_http_attempt", query_id=query_id, attempt=attempt + 1, url=url, model=payload_copy.get("model"))

            resp = requests.post(url, headers=headers, json=payload_copy, timeout=timeout)
            status = resp.status_code

            if status in (429, 500, 502, 503, 504):
                will_retry = attempt < max_retries - 1
                if logger:
                    logger.log("llm_http_error", query_id=query_id, status_code=status, error=f"HTTP {status}", attempt=attempt + 1, will_retry=will_retry)
                if will_retry:
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            out = resp.json()
            if logger:
                logger.log("llm_http_success", query_id=query_id, attempt=attempt + 1, status_code=status)
                
                def msg_preview(msg: Dict[str, Any]) -> str:
                    role = msg.get("role")
                    if role == 'assistant':
                        if msg.get("tool_calls") and isinstance(msg.get("tool_calls"), list) and len(msg.get("tool_calls")) > 0:
                            return f"assistant tool_calls: {msg.get('tool_calls')[0]['function']['name']}"
                        else:
                            return f"assistant content: {msg.get('content', '')[:20]}, reasoning: {msg.get('reasoning', '')[:20]}"
                    content = msg.get("content", "")
                    return f"{role}: {content[:20]}"
                
                in_tokens = [{msg_preview(msg) : count_tokens(str(msg))} for msg in payload.get("messages", [])]

                def out_preview(out: Dict[str, Any]) -> str:
                    choices = out.get("choices", [])
                    if not choices:
                        return "<no choices>"
                    finish_reason = choices[0].get("finish_reason")
                    content = choices[0].get("message", {}).get("content", "")
                    reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")
                    return f"finish_reason: {finish_reason}, content: {content[:20]}, reasoning_content: {reasoning_content[:20]}"

                out_tokens = [{out_preview(out): count_tokens(str(out['choices']))}]
                logger.log("llm_token_debug", 
                           query_id=query_id, 
                           input_tokens=in_tokens, 
                           output_tokens=out_tokens)
            if _token_tracker is not None:
                usage = out.get("usage") or {}
                _token_tracker.add(
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
            return out

        except requests.exceptions.RequestException as exc:
            will_retry = attempt < max_retries - 1
            if logger:
                logger.log("llm_http_error", query_id=query_id, error=str(exc), attempt=attempt + 1, will_retry=will_retry)
            if will_retry:
                continue
            raise

        except Exception as exc:  # pragma: no cover
            will_retry = attempt < max_retries - 1
            if logger:
                logger.log("llm_http_error", query_id=query_id, error=str(exc), attempt=attempt + 1, will_retry=will_retry)
            if will_retry:
                continue
            raise

    return {}

