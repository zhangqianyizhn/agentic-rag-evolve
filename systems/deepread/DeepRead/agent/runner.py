from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import (
    _preview_tool_calls,
    sanitize_for_vllm,
    should_sanitize_for_vllm,
)
from ..ports import ChatModel, EventLogger, ToolExecutor
from ..prompt.system import build_system_prompt
from ..runtime_state import token_tracker
from ..tool.fallback import fallback_tool_calls_from_text, strip_function_calls_block_any, strip_inline_tool_calls
from ..tool.schema import make_tools_schema


def run_agent(
    chat_model: ChatModel,
    doc_index: DocIndex,
    tool_executor: ToolExecutor,
    user_question: str,
    logger: EventLogger,
    max_rounds: int = 50,
    temperature: float = 0.0,
    enable_multimodal: bool = False,
    enable_vector: bool = False,
    enable_hybrid: bool = False,
    enable_semantic: bool = False,
    disable_bm25: bool = False,
    disable_regex: bool = False,
    disable_read: bool = False,
    tool_fallback: bool = True,
    enable_reasoning: bool = True,
    collected_texts: Optional[List[str]] = None,
) -> str:
    tools = make_tools_schema(doc_index, enable_semantic=enable_semantic)

    if disable_bm25:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "bm25_search"]
    if disable_regex:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "regex_search"]
    if not enable_vector:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "vector_search"]
    if not enable_hybrid:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "hybrid_search"]
    if not enable_semantic:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "semantic_retrieval"]
    if disable_read:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "read_section"]

    tool_names = [(t.get("function") or {}).get("name") for t in tools]
    system_prompt = build_system_prompt(doc_index, tool_names, enable_reasoning=enable_reasoning)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    do_sanitize = should_sanitize_for_vllm(chat_model.base_url)

    for round_id in range(1, max_rounds + 1):
        req_payload: Dict[str, Any] = {
            "model": chat_model.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "stream": False,
            "include_reasoning": bool(enable_reasoning),
        }

        payload_to_send = sanitize_for_vllm(req_payload, allow_tools=True) if do_sanitize else req_payload

        try:
            resp = dict(chat_model.complete(payload_to_send))
            usage = resp.get("usage") or {}
            token_tracker.add(
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
        except Exception:
            resp = {}

        msg = (resp.get("choices") or [{}])[0].get("message", {})  # type: ignore

        reasoning_content = None
        if enable_reasoning:
            for rf in ["reasoning", "reasoning_content", "thinking", "internal_monologue"]:
                if msg.get(rf) is not None:
                    reasoning_content = msg.get(rf)
                    break

        tool_calls = msg.get("tool_calls")
        recovered_from_text = False
        recovered_meta: Dict[str, Any] = {}

        content_str = msg.get("content") or ""
        reasoning_str = reasoning_content or ""

        if not tool_calls and tool_fallback:
            rec = fallback_tool_calls_from_text(content_str) or fallback_tool_calls_from_text(reasoning_str)
            if rec:
                tool_calls, recovered_meta = rec
                recovered_from_text = True
                logger.log(
                    "tool_calls_recovered_from_text",
                    round=round_id,
                    recovered=_preview_tool_calls(tool_calls),
                    recovered_kind=recovered_meta.get("kind"),
                )

        assistant_entry: Dict[str, Any] = {"role": "assistant"}

        content_for_history = msg.get("content")
        if isinstance(content_for_history, str) and recovered_from_text:
            content_for_history = strip_function_calls_block_any(content_for_history)
            if recovered_meta.get("kind") == "inline_json" and recovered_meta.get("spans"):
                content_for_history = strip_inline_tool_calls(content_for_history, recovered_meta["spans"])

        if content_for_history is None:
            content_for_history = ""
        assistant_entry["content"] = content_for_history

        if enable_reasoning and reasoning_content:
            assistant_entry["reasoning"] = reasoning_content

        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls

        messages.append(assistant_entry)

        if not tool_calls:
            final_answer = (msg.get("content") or "").strip()
            if final_answer:
                logger.log("final_answer", answer=final_answer)
                return final_answer

            if enable_reasoning and (reasoning_content is not None) and str(reasoning_content).strip():
                logger.log("llm_thinking_only", round=round_id, reasoning_preview=str(reasoning_content)[:2000])
            else:
                logger.log("llm_empty_message", round=round_id)
            continue

        for tc in tool_calls or []:
            tool_name = (tc.get("function") or {}).get("name")
            try:
                args_raw = (tc.get("function") or {}).get("arguments")
                args = args_raw if isinstance(args_raw, dict) else json.loads(args_raw or "{}")
            except Exception as exc:
                logger.log("tool_args_parse_error", tool=tool_name, raw=str(args_raw), error=str(exc))
                args = {}

            try:
                out = dict(tool_executor.execute(tool_name, args, call_id=tc.get("id")))

                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(out, ensure_ascii=False)})

                if collected_texts is not None and isinstance(out, dict) and out.get("ok", True):
                    if tool_name in ("bm25_search", "regex_search", "vector_search", "hybrid_search", "semantic_retrieval"):
                        for r in out.get("results", []):
                            if r.get("text"):
                                collected_texts.append(r["text"])
                            for nb in r.get("neighbors", []):
                                if isinstance(nb, dict) and nb.get("type") == "text" and nb.get("text"):
                                    collected_texts.append(nb["text"])
                    elif tool_name == "read_section":
                        for p in out.get("paragraphs", []):
                            if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                                collected_texts.append(p["text"])
                
                if (
                    enable_multimodal
                    and tool_name in ("bm25_search", "regex_search", "vector_search", "hybrid_search", "semantic_retrieval")
                    and isinstance(out, dict)
                ):
                    seen_keys = set()
                    mm_items: List[Dict[str, Any]] = []
                    for r in out.get("results", []):
                        neighbors = r.get("neighbors") or []
                        ref = r.get("ref") or {}
                        did = str(ref.get("doc_id"))
                        nid = str(ref.get("node_id"))
                        for item in neighbors:
                            if isinstance(item, dict):
                                item_type = item.get("type") or "text"
                                par_idx = int(item.get("paragraph_index", -1))
                                key = (did, nid, par_idx, item_type)
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                tagged = dict(item)
                                tagged["paragraph_index"] = par_idx
                                mm_items.append(tagged)
                    if mm_items:
                        messages.append({"role": "user", "content": mm_items})

            except Exception as exc:
                err = {"ok": False, "error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(err, ensure_ascii=False)})

    logger.log("max_rounds_reached", max_rounds=max_rounds)
    return "(Reached maximum rounds, no final answer generated)"
