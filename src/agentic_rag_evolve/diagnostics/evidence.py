"""Deterministic gold-evidence coverage across the DeepRead retrieval path."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

RETRIEVAL_TOOLS = {
    "bm25_search",
    "regex_search",
    "vector_search",
    "hybrid_search",
    "semantic_retrieval",
}


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    text: str
    reference: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_fingerprint(store_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(store_path).glob("*_corpus.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _paragraph_text(paragraph: Any) -> str:
    if isinstance(paragraph, str):
        return paragraph
    if isinstance(paragraph, dict):
        return str(paragraph.get("text") or paragraph.get("content") or "")
    return str(paragraph)


def _canonical_tokens(text: Any) -> tuple[str, ...]:
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return tuple(re.findall(r"[a-z0-9]+", value.lower()))


def load_corpus_passages(store_path: Path) -> tuple[EvidencePassage, ...]:
    passages: list[EvidencePassage] = []
    corpus_paths = sorted(Path(store_path).glob("*_corpus.json"))
    if not corpus_paths:
        raise FileNotFoundError(f"no *_corpus.json files found in {store_path}")
    for index, path in enumerate(corpus_paths, start=1):
        doc_name = path.stem.removesuffix("_corpus")
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("nodes") or []:
            paragraphs = [
                (paragraph_index, _paragraph_text(paragraph))
                for paragraph_index, paragraph in enumerate(node.get("paragraphs") or [])
                if _paragraph_text(paragraph)
            ]
            for window_size in (1, 2):
                for start in range(len(paragraphs) - window_size + 1):
                    window = paragraphs[start : start + window_size]
                    passages.append(
                        EvidencePassage(
                            text="\n".join(text for _, text in window),
                            reference={
                                "corpus_file": path.name,
                                "doc_id": str(index),
                                "doc_name": doc_name,
                                "node_id": str(node.get("id")),
                                "paragraph_indexes": [item[0] for item in window],
                            },
                        )
                    )
    return tuple(passages)


def _load_tool_result(tool: Mapping[str, Any], payload_root: Path) -> Mapping[str, Any]:
    result = tool.get("result")
    if isinstance(result, dict):
        return result
    reference = tool.get("result_ref")
    if not isinstance(reference, dict) or not reference.get("path"):
        return {}
    relative = Path(str(reference["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe trajectory payload path: {relative}")
    data = (payload_root / relative).read_bytes()
    if len(data) != int(reference.get("bytes", -1)):
        raise ValueError(f"trajectory payload byte count mismatch: {relative}")
    if hashlib.sha256(data).hexdigest() != reference.get("sha256"):
        raise ValueError(f"trajectory payload sha256 mismatch: {relative}")
    value = json.loads(data)
    return value if isinstance(value, dict) else {}


def _candidate_passages(
    trajectory: Mapping[str, Any], payload_root: Path
) -> tuple[EvidencePassage, ...]:
    passages: list[EvidencePassage] = []
    for turn in trajectory.get("turns") or []:
        for tool in turn.get("tools") or []:
            if tool.get("name") not in RETRIEVAL_TOOLS or not tool.get("ok"):
                continue
            result = _load_tool_result(tool, payload_root)
            for rank, hit in enumerate(result.get("results") or [], start=1):
                if not isinstance(hit, dict):
                    continue
                hit_ref = hit.get("ref") or {}
                parts = [str(hit.get("text") or "")]
                parts.extend(
                    _paragraph_text(neighbor) for neighbor in hit.get("neighbors") or []
                )
                text = "\n".join(part for part in parts if part)
                if not text:
                    continue
                passages.append(
                    EvidencePassage(
                        text,
                        {
                            "turn": turn.get("round"),
                            "tool_call_id": tool.get("call_id"),
                            "tool_name": tool.get("name"),
                            "rank": rank,
                            "doc_id": str(hit_ref.get("doc_id")),
                            "node_id": str(hit_ref.get("node_id")),
                            "paragraph_indexes": hit_ref.get("paragraph_indexes"),
                            "kind": "retrieval_window",
                        },
                    )
                )
    return tuple(passages)


def _read_passages(
    trajectory: Mapping[str, Any], payload_root: Path
) -> tuple[EvidencePassage, ...]:
    passages: list[EvidencePassage] = []
    for turn in trajectory.get("turns") or []:
        for tool in turn.get("tools") or []:
            if tool.get("name") != "read_section" or not tool.get("ok"):
                continue
            result = _load_tool_result(tool, payload_root)
            result_ref = result.get("ref") or {}
            paragraphs = result.get("paragraphs") or []
            text = "\n".join(
                value for paragraph in paragraphs if (value := _paragraph_text(paragraph))
            )
            if text:
                passages.append(
                    EvidencePassage(
                        text,
                        {
                            "turn": turn.get("round"),
                            "tool_call_id": tool.get("call_id"),
                            "tool_name": "read_section",
                            "doc_id": str(result_ref.get("doc_id")),
                            "node_id": str(result_ref.get("node_id")),
                            "paragraph_indexes": [
                                paragraph.get("paragraph_index")
                                for paragraph in paragraphs
                                if isinstance(paragraph, dict)
                            ],
                        },
                    )
                )
    return tuple(passages)


def _match_evidence(
    evidence: str,
    passages: Sequence[EvidencePassage],
    *,
    soft_threshold: float,
    min_soft_tokens: int,
) -> dict[str, Any]:
    evidence_tokens = set(_canonical_tokens(evidence))
    scored: list[tuple[float, EvidencePassage]] = []
    exact: list[EvidencePassage] = []
    canonical_exact: list[EvidencePassage] = []
    compact_evidence = "".join(_canonical_tokens(evidence))
    for passage in passages:
        canonical_passage_tokens = _canonical_tokens(passage.text)
        passage_tokens = set(canonical_passage_tokens)
        coverage = (
            len(evidence_tokens & passage_tokens) / len(evidence_tokens)
            if evidence_tokens
            else 0.0
        )
        if coverage:
            scored.append((coverage, passage))
        if evidence and evidence in passage.text:
            exact.append(passage)
        elif compact_evidence and compact_evidence in "".join(canonical_passage_tokens):
            canonical_exact.append(passage)
    best_coverage = max((item[0] for item in scored), default=0.0)
    if exact:
        matched, method, selected = True, "exact_substring", exact
    elif canonical_exact:
        matched, method, selected = True, "canonical_substring", canonical_exact
    elif len(evidence_tokens) < min_soft_tokens:
        matched, method, selected = False, "short_evidence_exact_required", []
    elif best_coverage >= soft_threshold:
        matched, method, selected = (
            True,
            "canonical_token_coverage",
            [item[1] for item in sorted(scored, key=lambda item: item[0], reverse=True)],
        )
    else:
        matched, method, selected = (
            False,
            "not_matched",
            [item[1] for item in sorted(scored, key=lambda item: item[0], reverse=True)],
        )
    references = []
    for passage in selected[:3]:
        passage_tokens = set(_canonical_tokens(passage.text))
        references.append(
            {
                **dict(passage.reference),
                "token_coverage": round(
                    len(evidence_tokens & passage_tokens) / len(evidence_tokens), 4
                )
                if evidence_tokens
                else 0.0,
                "excerpt": passage.text[:360],
            }
        )
    return {
        "matched": matched,
        "method": method,
        "token_coverage": round(best_coverage, 4),
        "references": references,
    }


def analyze_evidence_ladder(
    *,
    trajectory: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    store_path: Path,
    payload_root: Path,
    expected_store_fingerprint: str | None = None,
    soft_threshold: float = 0.8,
    min_soft_tokens: int = 4,
) -> dict[str, Any]:
    actual_fingerprint = store_fingerprint(store_path)
    if expected_store_fingerprint and actual_fingerprint != expected_store_fingerprint:
        raise ValueError("store fingerprint does not match the run manifest")
    layer_passages = {
        "corpus": load_corpus_passages(store_path),
        "candidate": _candidate_passages(trajectory, payload_root),
        "read": _read_passages(trajectory, payload_root),
        "answer": (
            EvidencePassage(str(trajectory.get("answer") or ""), {"kind": "final_answer"}),
        ),
    }
    evidence_results = []
    missing_counts = {layer: 0 for layer in layer_passages}
    for evidence_index, evidence in enumerate(evaluation.get("gold_evidence") or []):
        layers = {
            layer: _match_evidence(
                str(evidence),
                passages,
                soft_threshold=soft_threshold,
                min_soft_tokens=min_soft_tokens,
            )
            for layer, passages in layer_passages.items()
        }
        earliest_missing = next(
            (
                layer
                for layer in ("corpus", "candidate", "read", "answer")
                if not layers[layer]["matched"]
            ),
            None,
        )
        if earliest_missing:
            missing_counts[earliest_missing] += 1
        evidence_results.append(
            {
                "evidence_index": evidence_index,
                "token_count": len(_canonical_tokens(evidence)),
                "earliest_missing_layer": earliest_missing,
                "layers": layers,
            }
        )
    layer_summary = {
        layer: {
            "matched": sum(item["layers"][layer]["matched"] for item in evidence_results),
            "total": len(evidence_results),
            "recall": (
                sum(item["layers"][layer]["matched"] for item in evidence_results)
                / len(evidence_results)
                if evidence_results
                else 0.0
            ),
            "passage_count": len(layer_passages[layer]),
        }
        for layer in layer_passages
    }
    if not evidence_results:
        primary_signal = "no_gold_evidence"
    else:
        primary_signal = next(
            (
                layer
                for layer in ("corpus", "candidate", "read", "answer")
                if missing_counts[layer]
            ),
            "fully_covered",
        )
    return {
        "schema_version": "deepread-evidence-ladder-v1",
        "task_id": trajectory.get("task_id"),
        "store_fingerprint": actual_fingerprint,
        "matching": {
            "soft_threshold": soft_threshold,
            "min_soft_tokens": min_soft_tokens,
            "method": (
                "best local-window canonical token-set coverage with HTML boundaries; "
                "lexical heuristic, not entailment"
            ),
        },
        "primary_signal": primary_signal,
        "earliest_missing_counts": missing_counts,
        "layers": layer_summary,
        "evidence": evidence_results,
        "limitations": [
            "Answer-layer coverage measures textual support, not arithmetic or logical correctness.",
            "Token-set matching can overestimate coverage inside a long retrieval/read window.",
            "A gold evidence annotation may be sufficient but not the only valid supporting passage.",
        ],
    }
