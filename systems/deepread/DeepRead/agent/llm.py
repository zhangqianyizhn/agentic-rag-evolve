from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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
