"""Deterministic failure observations and conservative bad-case triage."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _signal(
    code: str,
    *,
    severity: str,
    summary: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "summary": summary,
        "evidence": dict(evidence or {}),
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def analyze_failure_signals(
    *,
    evaluation: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    evidence_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify observable failures without inferring a root cause or repair."""

    prediction = evaluation.get("prediction") or {}
    metrics = evaluation.get("metrics") or {}
    judge = evaluation.get("judge") or {}
    status = str(prediction.get("status") or trajectory.get("status") or "ok")
    answer = str(prediction.get("answer") or trajectory.get("answer") or "").strip()
    judge_status = str(judge.get("status") or "unknown")
    judge_score = _as_float(judge.get("score"))
    baseline_recall = _as_float(metrics.get("recall"))
    layers = evidence_coverage.get("layers") or {}
    candidate_recall = _as_float((layers.get("candidate") or {}).get("recall"))
    path_signal = str(evidence_coverage.get("primary_signal") or "unknown")
    signals: list[dict[str, Any]] = []

    if status != "ok":
        signals.append(
            _signal(
                "execution_error",
                severity="error",
                summary="The DeepRead run did not produce a successful prediction.",
                evidence={
                    "status": status,
                    "error_type": prediction.get("error_type"),
                    "termination_reason": prediction.get("termination_reason"),
                },
            )
        )
    if not answer:
        signals.append(
            _signal(
                "empty_answer",
                severity="error",
                summary="The run produced no non-empty final answer.",
                evidence={"status": status},
            )
        )

    path_codes = {
        "corpus": (
            "gold_evidence_missing_from_corpus",
            "Gold evidence was not found in a local corpus window.",
        ),
        "candidate": (
            "retrieval_candidate_miss",
            "Gold evidence exists in the corpus but did not enter a retrieval candidate window.",
        ),
        "read": (
            "evidence_not_read",
            "Gold evidence entered retrieval candidates but was not present in a read_section result.",
        ),
        "answer": (
            "gold_evidence_not_lexicalized_in_answer",
            "Read evidence was not lexically reproduced in the final answer.",
        ),
        "no_gold_evidence": (
            "no_gold_evidence",
            "The evaluation record has no gold evidence for deterministic path analysis.",
        ),
    }
    if path_signal in path_codes:
        code, summary = path_codes[path_signal]
        signals.append(
            _signal(
                code,
                severity="warning" if path_signal != "answer" else "info",
                summary=summary,
                evidence={
                    "earliest_missing_layer": path_signal,
                    "layer_recall": {
                        layer: value.get("recall")
                        for layer, value in layers.items()
                        if isinstance(value, Mapping)
                    },
                },
            )
        )

    if (
        baseline_recall is not None
        and candidate_recall is not None
        and candidate_recall > baseline_recall
    ):
        signals.append(
            _signal(
                "baseline_recall_disagreement",
                severity="warning",
                summary=(
                    "The canonical candidate coverage is higher than the baseline retrieval "
                    "recall; representation or evaluator alignment requires review."
                ),
                evidence={
                    "baseline_recall": baseline_recall,
                    "candidate_recall": candidate_recall,
                },
            )
        )

    if judge_status == "error":
        signals.append(
            _signal(
                "judge_error",
                severity="error",
                summary="The answer judge failed, so answer correctness is unavailable.",
                evidence={"error": judge.get("error")},
            )
        )
    elif judge_score is not None:
        if judge_score <= 1:
            signals.append(
                _signal(
                    "judged_incorrect",
                    severity="error",
                    summary="The configured answer judge scored the prediction as incorrect.",
                    evidence={"score": judge_score, "status": judge_status},
                )
            )
        elif judge_score < 4:
            signals.append(
                _signal(
                    "judged_partial",
                    severity="warning",
                    summary="The configured answer judge found the prediction incomplete or imperfect.",
                    evidence={"score": judge_score, "status": judge_status},
                )
            )
        else:
            signals.append(
                _signal(
                    "judged_correct",
                    severity="info",
                    summary="The configured answer judge found the prediction fully correct.",
                    evidence={"score": judge_score, "status": judge_status},
                )
            )
    elif judge_status in {"skipped", "unknown", "skipped_prediction_error"}:
        signals.append(
            _signal(
                "answer_correctness_unavailable",
                severity="info",
                summary="No successful answer-judge result is available.",
                evidence={"judge_status": judge_status},
            )
        )

    codes = {item["code"] for item in signals}
    if "execution_error" in codes:
        triage, confidence = "execution_failure", "high"
    elif "empty_answer" in codes:
        triage, confidence = "no_answer", "high"
    elif "judge_error" in codes or "gold_evidence_missing_from_corpus" in codes:
        triage, confidence = "evaluation_suspicious", "high"
    elif "judged_incorrect" in codes:
        triage, confidence = "incorrect_answer", "high"
    elif "judged_partial" in codes:
        triage, confidence = "partial_answer", "high"
    elif "baseline_recall_disagreement" in codes:
        triage, confidence = "evaluation_suspicious", "high"
    elif "judged_correct" in codes:
        triage, confidence = "pass", "high"
    else:
        triage, confidence = "needs_judgment", "medium"

    diagnosis_routes = {
        "execution_failure": (True, "deepread", "Inspect whether the failure is caused by evolvable DeepRead behavior."),
        "no_answer": (True, "deepread", "Diagnose why DeepRead did not produce an answer."),
        "incorrect_answer": (True, "deepread", "Diagnose the answer failure against its evidence path."),
        "partial_answer": (True, "deepread", "Diagnose the missing or inaccurate answer content."),
        "evaluation_suspicious": (
            False,
            "evaluation_review",
            "Review data/evaluator alignment before attributing a DeepRead defect.",
        ),
        "needs_judgment": (
            False,
            "answer_judge",
            "Obtain a reliable answer judgment before DeepRead diagnosis.",
        ),
        "pass": (False, "none", "No accuracy failure is available for diagnosis."),
    }
    eligible, target, reason = diagnosis_routes[triage]

    return {
        "schema_version": "deepread-failure-signals-v1",
        "task_id": trajectory.get("task_id"),
        "triage": triage,
        "confidence": confidence,
        "is_bad_case": triage != "pass",
        "diagnosis_route": {
            "eligible": eligible,
            "target": target,
            "reason": reason,
        },
        "outcome": {
            "prediction_status": status,
            "has_answer": bool(answer),
            "termination_reason": prediction.get("termination_reason"),
            "judge_status": judge_status,
            "judge_score": judge_score,
        },
        "evidence_path": {
            "primary_signal": path_signal,
            "layer_recall": {
                layer: value.get("recall")
                for layer, value in layers.items()
                if isinstance(value, Mapping)
            },
        },
        "signals": signals,
        "signal_counts": dict(Counter(item["severity"] for item in signals)),
        "interpretation": (
            "Deterministic triage only: signals describe observable manifestations, "
            "not root causes or authorized repairs."
        ),
    }
