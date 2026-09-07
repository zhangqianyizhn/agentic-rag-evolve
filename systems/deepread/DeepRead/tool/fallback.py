from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------
# Tool-call fallback parsing
# ------------------------------
_DSML_MARK_RE = re.compile(r"[\uFF5C|]DSML[\uFF5C|]")

_ALLOWED_TOOL_NAMES = {
    "read_section",
    "bm25_search",
    "regex_search",
    "vector_search",
    "hybrid_search",
    "semantic_retrieval",
}

_INLINE_TOOL_HEAD_RE = re.compile(
    r"\b(?P<name>"
    + "|".join(re.escape(n) for n in sorted(_ALLOWED_TOOL_NAMES, key=len, reverse=True))
    + r")\b\s*(?P<lp>\()?\s*[:=]?\s*\{",
    re.I,
)

_XML_INVOKE_START_RE = re.compile(
    r'<\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)\s+name\s*=\s*"([^"]+)"[^>]*>',
    re.I,
)

_XML_ANY_END_RE = re.compile(
    r"<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_calls|tool_calls|function_call|tool_call)[^>]*>",
    re.I,
)

_XML_PARAM_RE = re.compile(
    r'<\s*(?:parameter|param)\s+name\s*=\s*"([^"]+)"[^>]*>\s*'
    r"([\s\S]*?)"
    r"(?="
    r"<\s*/\s*(?:parameter|param)[^>]*>"
    r"|<\s*(?:parameter|param)\s+name\s*="
    r"|<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_calls|tool_calls|function_call|tool_call)[^>]*>"
    r"|$)",
    re.I,
)


def normalize_toolcall_markup(text: str) -> str:
    if not text:
        return text
    text = _DSML_MARK_RE.sub("", text)
    text = text.replace("\u200b", "")
    return text


def normalize_toolcall_markup_preserve_len(text: str) -> str:
    if not text:
        return text

    def _pad(m: re.Match) -> str:
        return " " * len(m.group(0))

    text = _DSML_MARK_RE.sub(_pad, text)
    text = text.replace("\u200b", " ")
    return text


def _extract_balanced_braces(s: str, brace_start: int) -> Optional[int]:
    if brace_start < 0 or brace_start >= len(s) or s[brace_start] != "{":
        return None

    depth = 0
    in_str = False
    esc = False
    quote_char = ""

    for i in range(brace_start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == quote_char:
                in_str = False
                quote_char = ""
            continue

        if ch == '"' or ch == "'":
            in_str = True
            quote_char = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _try_parse_json_obj(obj_text: str) -> Optional[Dict[str, Any]]:
    if not obj_text:
        return None
    t = obj_text.strip()
    t2 = t.replace('\\"', '"').replace("\\'", "'")

    try:
        v = json.loads(t2)
        return v if isinstance(v, dict) else None
    except Exception:
        pass

    try:
        v = ast.literal_eval(t2)
        return dict(v) if isinstance(v, dict) else None
    except Exception:
        return None


def fallback_tool_calls_from_text_xmlish(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    t = normalize_toolcall_markup(text)
    if ("<invoke" not in t.lower()) and ("<functioninvoke" not in t.lower()):
        return None
    if "<parameter" not in t.lower() and "<param" not in t.lower():
        return None

    calls: List[Dict[str, Any]] = []

    for m in _XML_INVOKE_START_RE.finditer(t):
        raw_name = (m.group(1) or "").strip()
        tool_name = raw_name.replace(" ", "_")
        if tool_name not in _ALLOWED_TOOL_NAMES:
            continue

        body_start = m.end()
        end_pos = len(t)
        end_m = _XML_ANY_END_RE.search(t, pos=body_start)
        if end_m:
            end_pos = end_m.start()

        body = t[body_start:end_pos]

        args: Dict[str, Any] = {}
        for pk, pv in _XML_PARAM_RE.findall(body):
            k = (pk or "").strip()
            v = (pv or "").strip()
            if k:
                args[k] = v

        calls.append(
            {
                "id": f"fallback_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )

    return calls or None


def fallback_tool_calls_from_text_inline_json(text: str) -> Optional[Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]]:
    if not text:
        return None

    t = normalize_toolcall_markup_preserve_len(text)

    calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []

    for m in _INLINE_TOOL_HEAD_RE.finditer(t):
        name = (m.group("name") or "").strip()
        if name not in _ALLOWED_TOOL_NAMES:
            continue

        brace_start = t.find("{", m.start())
        if brace_start < 0:
            continue
        brace_end = _extract_balanced_braces(t, brace_start)
        if brace_end is None:
            continue

        obj_text = t[brace_start : brace_end + 1]
        args = _try_parse_json_obj(obj_text)
        if not isinstance(args, dict):
            continue

        end_excl = brace_end + 1
        j = end_excl
        while j < len(t) and t[j].isspace():
            j += 1
        if (m.group("lp") is not None) and j < len(t) and t[j] == ")":
            end_excl = j + 1

        calls.append(
            {
                "id": f"fallback_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
        spans.append((m.start(), end_excl))

    if not calls:
        return None
    return calls, spans


def fallback_tool_calls_from_text(text: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not text:
        return None

    calls_xml = fallback_tool_calls_from_text_xmlish(text)
    if calls_xml:
        return calls_xml, {"kind": "xmlish"}

    inline = fallback_tool_calls_from_text_inline_json(text)
    if inline:
        calls, spans = inline
        return calls, {"kind": "inline_json", "spans": spans}

    return None


def strip_function_calls_block_any(text: str) -> str:
    if not text:
        return text
    t = normalize_toolcall_markup(text)

    t = re.sub(
        r"<\s*(?:function_calls|tool_calls)[^>]*>[\s\S]*?<\s*/\s*(?:function_calls|tool_calls)[^>]*>",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"<\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)[^>]*>[\s\S]*?(?:<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)[^>]*>|$)",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"<\s*/\s*(?:function_calls|tool_calls|invoke|functioninvoke|toolinvoke|parameter|param)[^>]*>",
        "",
        t,
        flags=re.I,
    )

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def strip_inline_tool_calls(text: str, spans: List[Tuple[int, int]]) -> str:
    if not text or not spans:
        return text
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    for s, e in spans_sorted:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    out_parts: List[str] = []
    last = 0
    for s, e in merged:
        out_parts.append(text[last:s])
        last = e
    out_parts.append(text[last:])

    cleaned = "".join(out_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
