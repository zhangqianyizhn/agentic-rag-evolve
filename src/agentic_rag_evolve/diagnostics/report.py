"""Compact, human-reviewable reports over diagnostic bundles."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class BadCaseReport:
    total: int
    bad_cases: int
    output_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compact(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _load_bundle(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    bundle_path = path / "bundle.json" if path.is_dir() else path
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "deepread-diagnostic-input-v1":
        raise ValueError(f"unsupported diagnostic bundle: {bundle_path}")
    signals = value.get("failure_signals") or {}
    if signals.get("schema_version") != "deepread-failure-signals-v1":
        raise ValueError(f"diagnostic bundle has no v1 failure signals: {bundle_path}")
    return {**value, "_bundle_path": str(bundle_path.resolve())}


def _record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = bundle.get("evaluation") or {}
    prediction = evaluation.get("prediction") or {}
    coverage = bundle.get("evidence_coverage") or {}
    failure = bundle.get("failure_signals") or {}
    return {
        "task_id": (bundle.get("task") or {}).get("task_id"),
        "sample_id": (bundle.get("task") or {}).get("sample_id"),
        "question": (bundle.get("task") or {}).get("question"),
        "triage": failure.get("triage"),
        "confidence": failure.get("confidence"),
        "is_bad_case": failure.get("is_bad_case"),
        "metrics": evaluation.get("metrics") or {},
        "judge": {
            key: (evaluation.get("judge") or {}).get(key)
            for key in ("status", "score", "reasoning")
        },
        "gold_answers": evaluation.get("gold_answers") or [],
        "generated_answer": prediction.get("answer"),
        "evidence_path": failure.get("evidence_path") or {},
        "diagnosis_route": failure.get("diagnosis_route") or {},
        "signals": failure.get("signals") or [],
        "evidence_references": [
            {
                "evidence_index": evidence.get("evidence_index"),
                "earliest_missing_layer": evidence.get("earliest_missing_layer"),
                "layers": {
                    layer: ((evidence.get("layers") or {}).get(layer) or {}).get(
                        "references", []
                    )[:1]
                    for layer in ("corpus", "candidate", "read")
                },
            }
            for evidence in coverage.get("evidence") or []
        ],
        "bundle_path": bundle.get("_bundle_path"),
    }


def _markdown(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    lines = [
        "# DeepRead bad-case report",
        "",
        f"Total bundles: {summary['total']}; bad cases: {summary['bad_cases']}.",
        "",
        "## Triage summary",
        "",
        "| Triage | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in sorted((summary.get("triage_counts") or {}).items())
    )
    for record in records:
        if not record.get("is_bad_case"):
            continue
        lines.extend(
            [
                "",
                f"## {record.get('task_id')} — `{record.get('triage')}`",
                "",
                f"Question: {_compact(record.get('question'))}",
                "",
                f"Generated: {_compact(record.get('generated_answer'))}",
                "",
                f"Gold: {_compact(' | '.join(record.get('gold_answers') or []))}",
                "",
                "Signals:",
                "",
            ]
        )
        lines.extend(
            f"- `{signal.get('code')}` ({signal.get('severity')}): "
            f"{signal.get('summary')}"
            for signal in record.get("signals") or []
        )
        layer_recall = (record.get("evidence_path") or {}).get("layer_recall") or {}
        lines.extend(
            [
                "",
                "Evidence path: "
                + ", ".join(
                    f"{layer}={value}" for layer, value in layer_recall.items()
                ),
                "",
                "Next route: "
                + str((record.get("diagnosis_route") or {}).get("target") or "unknown"),
                "",
                f"Bundle: `{record.get('bundle_path')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def build_bad_case_report(
    *, bundle_paths: Sequence[Path], output_path: Path
) -> BadCaseReport:
    if not bundle_paths:
        raise ValueError("at least one diagnostic bundle is required")
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"bad-case report directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    records = [_record(_load_bundle(path)) for path in bundle_paths]
    task_ids = [str(record.get("task_id") or "") for record in records]
    if any(not task_id for task_id in task_ids):
        raise ValueError("diagnostic bundle has no task_id")
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate task IDs in bad-case report: {duplicates}")
    records.sort(key=lambda item: str(item["task_id"]))
    bad_cases = [record for record in records if record["is_bad_case"]]
    summary = {
        "schema_version": "deepread-bad-case-summary-v1",
        "total": len(records),
        "bad_cases": len(bad_cases),
        "triage_counts": dict(Counter(str(record["triage"]) for record in records)),
        "signal_counts": dict(
            Counter(
                signal["code"]
                for record in records
                for signal in record.get("signals") or []
            )
        ),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "bad_cases.json").write_text(
        json.dumps(bad_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "bad_cases.md").write_text(
        _markdown(records, summary), encoding="utf-8"
    )
    return BadCaseReport(
        total=len(records),
        bad_cases=len(bad_cases),
        output_files=("summary.json", "bad_cases.json", "bad_cases.md"),
    )
