"""Explicit DeepRead source visibility policy for diagnosis agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiagnosticSourceSpec:
    path: str
    component: str
    purpose: str


DIAGNOSTIC_SOURCE_POLICY = (
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/agent/runner.py",
        "agent_loop",
        "Model-turn loop, context assembly, tool selection handling, and stopping behavior.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/agent/contracts.py",
        "agent_loop",
        "Agent outcome and semantic observer contracts.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/agent/llm.py",
        "context_protocol",
        "Provider compatibility sanitization and assistant/tool message normalization.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/prompt/system.py",
        "prompt",
        "DeepRead system instructions and tool-use policy.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/schema.py",
        "tool_affordance",
        "Tool names, descriptions, and argument schemas visible to the model.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/fallback.py",
        "tool_protocol",
        "Recovery of tool calls emitted as text.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/executor.py",
        "tool_execution",
        "Mapping from model tool decisions to retrieval operations and parameters.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/retrieval.py",
        "retrieval",
        "Document index state and retrieval API composition.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/bm25_search.py",
        "retrieval",
        "Lexical BM25 retrieval behavior.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/regex_search.py",
        "retrieval",
        "Keyword and regular-expression retrieval behavior.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/vector_search.py",
        "retrieval",
        "Vector retrieval and candidate construction.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/hybrid_search.py",
        "retrieval",
        "BM25/vector score fusion.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/semantic_retrieval.py",
        "retrieval",
        "Two-stage retrieval and reranking.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/read_section.py",
        "document_read",
        "Coordinate-based section reading.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/utils.py",
        "retrieval",
        "Shared tokenization, neighbor-window, and result helpers.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/tool/corpus.py",
        "index_loading",
        "Corpus and vector-artifact loading into the document index.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/DeepRead/index/markdown_parser.py",
        "indexing",
        "Markdown heading/paragraph parsing and tree construction.",
    ),
    DiagnosticSourceSpec(
        "systems/deepread/ingestion.py",
        "indexing",
        "Markdown-only corpus and embedding artifact construction.",
    ),
)


DIAGNOSTIC_TOOL_CONTRACTS = (
    {
        "name": "list_sources",
        "description": "List source files visible to the diagnosis agent, optionally by component.",
        "arguments": {"component": "optional string"},
    },
    {
        "name": "read_source",
        "description": "Read a bounded line range from an allowlisted source file.",
        "arguments": {
            "path": "exact path returned by list_sources",
            "start_line": "1-based integer",
            "end_line": "inclusive integer; at most 240 lines",
        },
    },
    {
        "name": "read_payload",
        "description": "Read a bounded character slice from a hash-verified external trajectory payload.",
        "arguments": {
            "path": "exact path present in access.payloads",
            "offset_chars": "non-negative integer",
            "limit_chars": "positive integer; at most 12000",
        },
    },
)
