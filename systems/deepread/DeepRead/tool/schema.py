from __future__ import annotations

from typing import Any, Dict, List


def _neighbor_hint_sentence(doc_index: DocIndex) -> str:
    nw = doc_index.neighbor_window
    if nw is None:
        return ""
    up, down = nw
    return f" Neighbor expansion is enabled: returned hits may include up to {int(up)} paragraph(s) above and {abs(int(down))} paragraph(s) below the matched paragraph."


def make_tools_schema(doc_index: DocIndex, enable_semantic: bool = False) -> List[Dict[str, Any]]:
    nh = _neighbor_hint_sentence(doc_index)

    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_doc_structure",
                "description": (
                    "Retrieve the Directory Structure of one or more documents. "
                    "Pass a list of doc_id strings to get their Directory Structure. "
                    "If doc_ids is omitted or null, returns only the total document count and id range."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of doc_id values to inspect (e.g. [\"1\", \"3\"]). Omit to get the id range only."
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_section",
                "description": "Read a specific paragraph range from the specified document node. Returns paragraphs from start_paragraph (inclusive) to end_paragraph (exclusive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "node_id": {"type": "string"},
                        "start_paragraph": {"type": "integer", "minimum": 0},
                        "end_paragraph": {
                            "type": "integer",
                            "minimum": -1,
                            "description": "Exclusive end paragraph; use -1 to read to the end of the node.",
                        },
                    },
                    "required": ["doc_id", "node_id", "start_paragraph", "end_paragraph"],
                },
            },
        }
    ]

    bm25_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    regex_props: Dict[str, Any] = {
        "pattern": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    vector_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    hybrid_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }

    tools.extend(
        [
            {
                "type": "function",
                "function": {
                    "name": "bm25_search",
                    "description": "Perform BM25-based text retrieval." + nh,
                    "parameters": {"type": "object", "properties": bm25_props, "required": ["query", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "regex_search",
                    "description": "Search for text patterns using regex." + nh,
                    "parameters": {"type": "object", "properties": regex_props, "required": ["pattern", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Perform embedding-based retrieval using cosine similarity." + nh,
                    "parameters": {"type": "object", "properties": vector_props, "required": ["query", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hybrid_search",
                    "description": "Fuse BM25 and embedding retrieval results with adjustable weights (internal)." + nh,
                    "parameters": {"type": "object", "properties": hybrid_props, "required": ["query", "scope"]},
                },
            },
        ]
    )

    if enable_semantic:
        semantic_props: Dict[str, Any] = {
            "query": {"type": "string"},
            "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
            "doc_id": {"type": "string", "default": None},
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "semantic_retrieval",
                    "description": "Semantic Retrieval: stage-1 recall + rerank internally. Neighbor expansion (if enabled) applies ONLY after reranking on final results." + nh,
                    "parameters": {"type": "object", "properties": semantic_props, "required": ["query", "scope"]},
                },
            }
        )

    return tools
