from .fallback import (
    fallback_tool_calls_from_text,
    fallback_tool_calls_from_text_inline_json,
    fallback_tool_calls_from_text_xmlish,
    normalize_toolcall_markup,
    normalize_toolcall_markup_preserve_len,
    strip_function_calls_block_any,
    strip_inline_tool_calls,
)
from .corpus import load_corpus
from .retrieval import DocIndex
from .schema import make_tools_schema

__all__ = [
    "DocIndex",
    "fallback_tool_calls_from_text",
    "fallback_tool_calls_from_text_inline_json",
    "fallback_tool_calls_from_text_xmlish",
    "load_corpus",
    "make_tools_schema",
    "normalize_toolcall_markup",
    "normalize_toolcall_markup_preserve_len",
    "strip_function_calls_block_any",
    "strip_inline_tool_calls",
]
