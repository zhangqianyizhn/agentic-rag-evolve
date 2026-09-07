"""Paired comparison against historical ruc-ov evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def compare_with_historical(
    current: Sequence[Mapping[str, Any]], historical_path: Path
) -> dict[str, Any]:
    historical_data = json.loads(Path(historical_path).read_text(encoding="utf-8"))
    historical_results = historical_data.get("results", historical_data)
    if not isinstance(historical_results, list):
        raise ValueError("historical evaluation must be a list or contain results")
    by_question: dict[str, Mapping[str, Any]] = {}
    for item in historical_results:
        question = str(item.get("question"))
        if question in by_question:
            raise ValueError(f"historical comparison has duplicate question: {question}")
        by_question[question] = item
    pairs: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for item in current:
        old = by_question.get(str(item.get("question")))
        if old is None:
            unmatched.append(str(item.get("task_id")))
            continue
        old_metrics = old.get("metrics") or {}
        new_metrics = item.get("metrics") or {}
        metric_pairs = {
            "f1": (old_metrics.get("F1"), new_metrics.get("f1")),
            "recall": (old_metrics.get("Recall"), new_metrics.get("recall")),
            "accuracy_0_4": (old_metrics.get("Accuracy"), new_metrics.get("accuracy_0_4")),
        }
        pairs.append({
            "task_id": item.get("task_id"),
            "question": item.get("question"),
            "metrics": {
                name: {
                    "historical": old_value,
                    "current": new_value,
                    "delta": new_value - old_value if old_value is not None and new_value is not None else None,
                }
                for name, (old_value, new_value) in metric_pairs.items()
            },
        })
    return {"matched": len(pairs), "unmatched_task_ids": unmatched, "pairs": pairs}
