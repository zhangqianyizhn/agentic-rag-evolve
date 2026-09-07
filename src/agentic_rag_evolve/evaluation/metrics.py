"""Deterministic metrics compatible with the historical ruc-ov baseline."""

from __future__ import annotations

import collections
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Sequence


def normalize_answer(value: Any) -> str:
    text = str(value).replace(",", "").lower()
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = collections.Counter(predicted) & collections.Counter(expected)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def is_refusal(text: str) -> bool:
    phrases = (
        "not mentioned",
        "no information",
        "cannot be answered",
        "none",
        "unknown",
        "don't know",
    )
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    evidence_index: int
    matched: bool
    method: str
    token_count: int
    token_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence_recall(
    retrieved_texts: Sequence[str],
    evidence: Sequence[str],
    *,
    soft_threshold: float = 0.8,
    min_soft_match_tokens: int = 4,
) -> tuple[float, tuple[EvidenceMatch, ...]]:
    """Return baseline-compatible recall plus per-evidence diagnostics."""

    if not evidence:
        return 0.0, ()
    combined = " ".join(str(text) for text in retrieved_texts)
    retrieved_tokens = set(normalize_answer(combined).split())
    matches: list[EvidenceMatch] = []

    for index, expected_text in enumerate(evidence):
        normalized = normalize_answer(expected_text)
        tokens = set(normalized.split())
        coverage = len(tokens & retrieved_tokens) / len(tokens) if tokens else 0.0
        if expected_text in combined:
            method = "exact_substring"
            matched = True
        elif len(tokens) < min_soft_match_tokens:
            method = "short_evidence_exact_required"
            matched = False
        elif coverage >= soft_threshold:
            method = "soft_token_coverage"
            matched = True
        else:
            method = "not_matched"
            matched = False
        matches.append(
            EvidenceMatch(
                evidence_index=index,
                matched=matched,
                method=method,
                token_count=len(tokens),
                token_coverage=coverage,
            )
        )

    return sum(match.matched for match in matches) / len(matches), tuple(matches)
