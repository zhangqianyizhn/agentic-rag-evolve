"""FinanceBench adapter and evaluation orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agentic_rag_evolve.contracts import EvaluationReference

from .judge import JudgeModel, JudgeResult, judge_answer
from .metrics import evidence_recall, is_refusal, token_f1


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    evaluated: int
    prediction_errors: int
    judge_succeeded: int
    judge_errors: int
    judge_skipped: int
    average_f1: float
    average_recall: float
    average_accuracy_0_4: float | None
    average_accuracy_normalized: float | None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_text_tuple(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value]
    return tuple(str(item).strip() for item in values if str(item).strip())


def load_financebench_references(dataset_path: Path) -> dict[str, EvaluationReference]:
    references: dict[str, EvaluationReference] = {}
    with Path(dataset_path).open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("financebench_id") or index)
            evidence_items = record.get("evidence") or []
            evidence = tuple(
                str(item.get("evidence_text", "")).strip()
                if isinstance(item, dict)
                else str(item).strip()
                for item in evidence_items
            )
            if task_id in references:
                raise ValueError(f"duplicate dataset task_id: {task_id}")
            references[task_id] = EvaluationReference(
                task_id=task_id,
                gold_answers=_as_text_tuple(record.get("answer") or record.get("gold_answers") or []),
                gold_evidence=tuple(item for item in evidence if item),
                metadata={
                    "sample_id": str(record.get("doc_name") or ""),
                    "question": str(record.get("question") or "").strip(),
                    "category": str(record.get("question_type") or record.get("category") or ""),
                },
            )
    return references


def load_predictions(prediction_path: Path) -> tuple[dict[str, Any], ...]:
    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(prediction_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            prediction = json.loads(line)
            task_id = str(prediction.get("task_id") or "")
            if not task_id:
                raise ValueError(f"prediction line {line_number} has no task_id")
            if task_id in seen:
                raise ValueError(f"duplicate prediction task_id: {task_id}")
            seen.add(task_id)
            predictions.append(prediction)
    if not predictions:
        raise ValueError(f"prediction file is empty: {prediction_path}")
    return tuple(predictions)


def _average(values: Iterable[float]) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else 0.0


def evaluate_financebench(
    *,
    dataset_path: Path,
    prediction_path: Path,
    judge_model: JudgeModel | None = None,
) -> tuple[tuple[dict[str, Any], ...], EvaluationSummary]:
    references = load_financebench_references(dataset_path)
    predictions = load_predictions(prediction_path)
    details: list[dict[str, Any]] = []

    for prediction in predictions:
        task_id = str(prediction["task_id"])
        if task_id not in references:
            raise ValueError(f"prediction task_id not found in dataset: {task_id}")
        reference = references[task_id]
        expected_question = str(reference.metadata.get("question") or "")
        predicted_question = prediction.get("question")
        if predicted_question is not None and str(predicted_question).strip() != expected_question:
            raise ValueError(f"prediction question mismatch for task_id: {task_id}")
        expected_sample = str(reference.metadata.get("sample_id") or "")
        predicted_sample = prediction.get("sample_id")
        if predicted_sample is not None and str(predicted_sample) != expected_sample:
            raise ValueError(f"prediction sample_id mismatch for task_id: {task_id}")
        status = str(prediction.get("status") or "ok")
        answer = str(prediction.get("answer") or "")
        retrieved = tuple(str(text) for text in prediction.get("retrieved_texts") or [])
        f1 = max((token_f1(answer, gold) for gold in reference.gold_answers), default=0.0)
        recall, evidence_matches = evidence_recall(retrieved, reference.gold_evidence)
        question = expected_question

        if status != "ok":
            judge = JudgeResult(
                status="skipped_prediction_error",
                score=None,
                reasoning="Prediction failed before evaluation.",
            )
        elif is_refusal(answer) and any(is_refusal(gold) for gold in reference.gold_answers):
            f1 = 1.0
            judge = JudgeResult(
                status="heuristic",
                score=4,
                reasoning="Both generated and gold answers identify an unanswerable condition.",
                prompt_type="Heuristic_Refusal_Check",
            )
        elif judge_model is None:
            judge = JudgeResult(
                status="skipped",
                score=None,
                reasoning="No judge model was configured.",
            )
        else:
            judge = judge_answer(
                judge_model,
                task_id=task_id,
                question=question,
                gold_answers=reference.gold_answers,
                answer=answer,
            )

        details.append({
            "task_id": task_id,
            "source_index": prediction.get("source_index"),
            "sample_id": reference.metadata.get("sample_id"),
            "question": question,
            "gold_answers": list(reference.gold_answers),
            "evidence": list(reference.gold_evidence),
            "category": reference.metadata.get("category"),
            "prediction": {
                "status": status,
                "answer": answer,
                "error_type": prediction.get("error_type"),
                "error": prediction.get("error"),
            },
            "retrieval": {
                "text_count": len(retrieved),
                "latency_seconds": prediction.get("latency_seconds"),
                "evidence_matches": [match.to_dict() for match in evidence_matches],
            },
            "token_usage": dict(prediction.get("token_usage") or {}),
            "metrics": {
                "f1": f1,
                "recall": recall,
                "accuracy_0_4": judge.score,
                "accuracy_normalized": judge.score / 4 if judge.score is not None else None,
            },
            "judge": judge.to_dict(),
        })

    successful_predictions = [item for item in details if item["prediction"]["status"] == "ok"]
    judged_scores = [item["metrics"]["accuracy_0_4"] for item in details if item["judge"]["status"] in {"ok", "heuristic"}]
    judge_errors = sum(item["judge"]["status"] == "error" for item in details)
    summary = EvaluationSummary(
        total=len(details),
        evaluated=len(successful_predictions),
        prediction_errors=len(details) - len(successful_predictions),
        judge_succeeded=len(judged_scores),
        judge_errors=judge_errors,
        judge_skipped=len(details) - len(judged_scores) - judge_errors,
        average_f1=_average(item["metrics"]["f1"] for item in details),
        average_recall=_average(item["metrics"]["recall"] for item in details),
        average_accuracy_0_4=_average(judged_scores) if judged_scores else None,
        average_accuracy_normalized=_average(score / 4 for score in judged_scores) if judged_scores else None,
        judge_input_tokens=sum(int(item["judge"]["input_tokens"]) for item in details),
        judge_output_tokens=sum(int(item["judge"]["output_tokens"]) for item in details),
    )
    return tuple(details), summary


def write_evaluation(
    output_path: Path,
    details: Sequence[Mapping[str, Any]],
    summary: EvaluationSummary,
) -> None:
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"evaluation output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "evaluation.json").write_text(
        json.dumps(list(details), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "evaluation_summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
