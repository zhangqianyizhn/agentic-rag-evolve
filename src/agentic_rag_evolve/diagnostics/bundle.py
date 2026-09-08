"""Build a minimal, self-contained task input for a diagnosis agent."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .policy import DIAGNOSTIC_SOURCE_POLICY, DIAGNOSTIC_TOOL_CONTRACTS


TRAJECTORY_TOP_FIELDS = (
    "schema_version",
    "task_id",
    "question",
    "status",
    "answer",
    "token_usage",
    "turns",
    "summary",
)
TURN_FIELDS = ("round", "model", "tools", "annotations", "raw_event_range")
MODEL_FIELDS = (
    "name",
    "reasoning",
    "content",
    "content_ref",
    "token_estimate",
    "provider_attempts",
    "provider_errors",
    "requested_tools",
)
TOOL_FIELDS = (
    "call_id",
    "name",
    "arguments",
    "ok",
    "result",
    "result_summary",
    "result_ref",
    "error",
    "call_event_id",
    "result_event_id",
)
RUN_CONFIG_FIELDS = (
    "temperature",
    "enable_vector",
    "enable_hybrid",
    "enable_semantic",
    "neighbor_window",
    "max_rounds",
    "retrieval_topk",
)
METRIC_FIELDS = ("f1", "recall", "accuracy_0_4", "accuracy_normalized")


@dataclass(frozen=True, slots=True)
class DiagnosticBundleReport:
    task_id: str
    source_count: int
    payload_count: int
    output_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_fields(value: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: value[field] for field in fields if field in value}


def _project_trajectory(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    projected = _project_fields(trajectory, TRAJECTORY_TOP_FIELDS)
    projected_turns: list[dict[str, Any]] = []
    for raw_turn in trajectory.get("turns") or []:
        turn = _project_fields(raw_turn, TURN_FIELDS)
        turn["model"] = _project_fields(raw_turn.get("model") or {}, MODEL_FIELDS)
        if "tools" in raw_turn:
            turn["tools"] = [
                _project_fields(tool, TOOL_FIELDS) for tool in raw_turn.get("tools") or []
            ]
        projected_turns.append(turn)
    projected["turns"] = projected_turns
    return projected


def _select_evaluation(evaluation: Any, task_id: str) -> Mapping[str, Any]:
    records = evaluation if isinstance(evaluation, list) else [evaluation]
    matches = [item for item in records if str(item.get("task_id") or "") == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one evaluation record for task_id {task_id!r}")
    return matches[0]


def _project_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_project_fields(
            evaluation,
            ("task_id", "sample_id", "question", "category", "gold_answers"),
        ),
        "gold_evidence": list(evaluation.get("evidence") or evaluation.get("gold_evidence") or []),
        "prediction": _project_fields(
            evaluation.get("prediction") or {},
            (
                "status",
                "answer",
                "error_type",
                "error",
                "termination_reason",
                "rounds_completed",
            ),
        ),
        "retrieval": _project_fields(
            evaluation.get("retrieval") or {},
            ("text_count", "evidence_matches"),
        ),
        "metrics": _project_fields(evaluation.get("metrics") or {}, METRIC_FIELDS),
        "judge": _project_fields(
            evaluation.get("judge") or {},
            ("status", "score", "reasoning", "prompt_type", "model_name", "error"),
        ),
    }


def _source_manifest(source_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for spec in DIAGNOSTIC_SOURCE_POLICY:
        path = source_root / spec.path
        if not path.is_file():
            raise FileNotFoundError(f"diagnostic source file not found: {path}")
        data = path.read_bytes()
        manifest.append({
            "path": spec.path,
            "component": spec.component,
            "purpose": spec.purpose,
            "sha256": _sha256_bytes(data),
            "bytes": len(data),
            "line_count": len(data.decode("utf-8").splitlines()),
        })
    return manifest


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative artifact path: {value}")
    return path


def _copy_payloads(
    trajectory: Mapping[str, Any],
    *,
    trajectory_root: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for turn in trajectory.get("turns") or []:
        for tool in turn.get("tools") or []:
            reference = tool.get("result_ref")
            if not isinstance(reference, dict) or not reference.get("path"):
                continue
            relative_path = _safe_relative_path(str(reference["path"]))
            source = trajectory_root / relative_path
            data = source.read_bytes()
            digest = _sha256_bytes(data)
            expected_digest = str(reference.get("sha256") or "")
            if digest != expected_digest:
                raise ValueError(f"payload sha256 mismatch: {relative_path}")
            if int(reference.get("bytes", -1)) != len(data):
                raise ValueError(f"payload byte count mismatch: {relative_path}")
            destination = output_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
            path_key = relative_path.as_posix()
            payload = payloads.setdefault(path_key, {
                "path": path_key,
                "sha256": digest,
                "bytes": len(data),
                "uses": [],
            })
            payload["uses"].append({
                "turn": turn.get("round"),
                "tool_call_id": tool.get("call_id"),
                "tool_name": tool.get("name"),
            })
    return [payloads[path] for path in sorted(payloads)]


def build_diagnostic_bundle(
    *,
    trajectory_path: Path,
    evaluation_path: Path,
    run_manifest_path: Path,
    source_root: Path,
    output_path: Path,
) -> DiagnosticBundleReport:
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"diagnostic bundle directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    raw_trajectory = _load_json(trajectory_path)
    if raw_trajectory.get("schema_version") != "deepread-trajectory-v2":
        raise ValueError("diagnostic bundles require deepread-trajectory-v2")
    trajectory = _project_trajectory(raw_trajectory)
    task_id = str(trajectory.get("task_id") or "")
    if not task_id:
        raise ValueError("trajectory has no task_id")
    evaluation = _project_evaluation(_select_evaluation(_load_json(evaluation_path), task_id))
    if trajectory.get("question") != evaluation.get("question"):
        raise ValueError(f"trajectory/evaluation question mismatch for task_id {task_id}")
    run_manifest = _load_json(run_manifest_path)
    if raw_trajectory.get("run_id") and run_manifest.get("run_id") != raw_trajectory.get("run_id"):
        raise ValueError("trajectory/run manifest run_id mismatch")

    payloads = _copy_payloads(
        trajectory,
        trajectory_root=Path(trajectory_path).parent,
        output_path=output_path,
    )
    sources = _source_manifest(Path(source_root))
    bundle = {
        "schema_version": "deepread-diagnostic-input-v1",
        "usage": {
            "purpose": "diagnose a development-set DeepRead result",
            "gold_labels_allowed": True,
            "must_not_be_used_for_final_test_adaptation": True,
        },
        "task": {
            "task_id": task_id,
            "sample_id": evaluation.get("sample_id"),
            "question": trajectory.get("question"),
        },
        "run_context": {
            "config": _project_fields(run_manifest.get("config") or {}, RUN_CONFIG_FIELDS),
            "models": _project_fields(
                run_manifest.get("providers") or {},
                ("chat", "embedding", "reranker"),
            ),
            "corpus": {
                "dataset_sha256": run_manifest.get("dataset_sha256"),
                "store_name": run_manifest.get("store_name"),
                "store_fingerprint": run_manifest.get("store_fingerprint"),
            },
        },
        "evaluation": evaluation,
        "trajectory": trajectory,
        "access": {
            "sources": sources,
            "payloads": payloads,
            "tools": list(DIAGNOSTIC_TOOL_CONTRACTS),
            "excluded_layers": [
                "src/agentic_rag_evolve/telemetry",
                "src/agentic_rag_evolve/providers",
                "src/agentic_rag_evolve/trajectory",
                "systems/deepread/runtime.py",
                "runner",
            ],
        },
    }
    bundle_path = output_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return DiagnosticBundleReport(
        task_id=task_id,
        source_count=len(sources),
        payload_count=len(payloads),
        output_file=bundle_path.name,
    )
