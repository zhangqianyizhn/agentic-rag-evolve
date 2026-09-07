"""Evaluation primitives that do not import DeepRead internals."""

from .evaluator import EvaluationSummary, evaluate_financebench
from .metrics import EvidenceMatch, evidence_recall, token_f1

__all__ = [
    "EvaluationSummary",
    "EvidenceMatch",
    "evaluate_financebench",
    "evidence_recall",
    "token_f1",
]
