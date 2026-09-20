"""Append-only, hash-chained ledger for one DeepRead evolution iteration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


BASE_STEPS = (
    "baseline_run",
    "evaluation",
    "trajectories",
    "diagnoses",
    "hypothesis_cohort",
    "hypotheses",
    "modification_plan",
    "candidate_created",
    "candidate_modified",
    "static_audit",
    "fixed_tests",
    "validation_gate",
    "outcome",
)
ACCEPTED_TAIL = ("materialization", "baseline_advanced", "terminal_memory", "iteration_report")
REJECTED_TAIL = ("terminal_memory", "iteration_report")

STEP_SCHEMAS = {
    "hypothesis_cohort": "deepread-hypothesis-cohort-v1",
    "hypotheses": "deepread-improvement-hypotheses-v1",
    "modification_plan": "deepread-modification-plan-v1",
    "candidate_created": "deepread-candidate-manifest-v1",
    "candidate_modified": "deepread-candidate-modification-v1",
    "static_audit": "deepread-candidate-audit-v1",
    "fixed_tests": "deepread-candidate-test-audit-v1",
    "validation_gate": "deepread-validation-gate-v1",
    "outcome": "deepread-candidate-outcome-v1",
    "materialization": "deepread-candidate-materialization-v1",
    "baseline_advanced": "deepread-baseline-entry-v1",
    "iteration_report": "deepread-iteration-report-v1",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def initialize_iteration(
    *, root: Path, iteration_id: str, baseline_id: str, baseline_commit: str
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not iteration_id or not baseline_id or len(baseline_commit) != 40:
        raise ValueError("iteration identity and 40-character baseline commit are required")
    manifest = {
        "schema_version": "deepread-iteration-ledger-v1",
        "iteration_id": iteration_id,
        "baseline_id": baseline_id,
        "baseline_commit": baseline_commit,
        "first_step": BASE_STEPS[0],
    }
    manifest["manifest_id"] = "ledger-" + _sha_bytes(_canonical(manifest))[:20]
    _write_exclusive(root / "manifest.json", manifest)
    (root / "events").mkdir()
    return manifest


def _event_files(root: Path) -> list[Path]:
    return sorted((root / "events").glob("*.json"))


def _expected_sequence(outcome: str | None) -> tuple[str, ...]:
    if outcome == "accepted":
        return BASE_STEPS + ACCEPTED_TAIL
    if outcome == "rejected":
        return BASE_STEPS + REJECTED_TAIL
    return BASE_STEPS


def _read_and_verify(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    manifest, manifest_bytes = _read_object(root / "manifest.json", "iteration manifest")
    if manifest.get("schema_version") != "deepread-iteration-ledger-v1":
        raise ValueError("unsupported iteration ledger schema")
    previous = _sha_bytes(manifest_bytes)
    events: list[dict[str, Any]] = []
    outcome: str | None = None
    completed: list[str] = []
    for index, path in enumerate(_event_files(root), 1):
        if path.name != f"{index:04d}.json":
            raise ValueError("iteration event numbering is not contiguous")
        event, data = _read_object(path, "iteration event")
        basis = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("event_sha256") != _sha_bytes(_canonical(basis)):
            raise ValueError(f"iteration event hash is invalid: {path.name}")
        if event.get("previous_sha256") != previous:
            raise ValueError(f"iteration event chain is broken: {path.name}")
        if event.get("sequence") != index:
            raise ValueError(f"iteration event sequence is invalid: {path.name}")
        for reference in (event.get("artifacts") or {}).values():
            artifact_path = Path(str(reference.get("path") or ""))
            if _sha_bytes(artifact_path.read_bytes()) != reference.get("sha256"):
                raise ValueError(f"iteration artifact changed after recording: {artifact_path}")
        if event.get("status") == "completed":
            expected = _expected_sequence(outcome)[len(completed)]
            if event.get("step") != expected:
                raise ValueError(f"unexpected completed iteration step: {event.get('step')}")
            completed.append(expected)
            if expected == "outcome":
                primary = (event.get("artifacts") or {}).get("primary") or {}
                value, _ = _read_object(Path(str(primary.get("path") or "")), "candidate outcome")
                outcome = str(value.get("outcome") or "")
                if outcome not in {"accepted", "rejected"}:
                    raise ValueError("candidate outcome is not terminal")
        previous = _sha_bytes(data)
        events.append(event)
    return manifest, events, outcome


def read_iteration_status(*, root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest, events, outcome = _read_and_verify(root)
    completed = [event["step"] for event in events if event.get("status") == "completed"]
    sequence = _expected_sequence(outcome)
    next_step = sequence[len(completed)] if len(completed) < len(sequence) else None
    return {
        "schema_version": "deepread-iteration-status-v1",
        "iteration_id": manifest["iteration_id"],
        "baseline_id": manifest["baseline_id"],
        "event_count": len(events),
        "completed_steps": completed,
        "failed_attempt_count": sum(event.get("status") == "failed" for event in events),
        "outcome": outcome,
        "next_step": next_step,
        "terminal": next_step is None,
    }


def append_iteration_event(
    *,
    root: Path,
    step: str,
    status: str,
    artifacts: Mapping[str, Path] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    _, events, _ = _read_and_verify(root)
    current = read_iteration_status(root=root)
    if current["terminal"]:
        raise ValueError("iteration is already terminal")
    if step != current["next_step"]:
        raise ValueError(f"expected iteration step {current['next_step']}, got {step}")
    if status not in {"completed", "failed"}:
        raise ValueError("iteration event status must be completed or failed")
    if status == "completed" and not artifacts:
        raise ValueError("completed iteration step must record artifacts")
    if status == "failed" and not error:
        raise ValueError("failed iteration step must record an error")
    refs: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted((artifacts or {}).items()):
        path = Path(raw_path).resolve()
        refs[name] = {"path": str(path), "sha256": _sha_bytes(path.read_bytes())}
    if status == "completed":
        primary = refs.get("primary")
        if primary is None:
            raise ValueError("completed iteration step requires a primary artifact")
        expected_schema = STEP_SCHEMAS.get(step)
        if expected_schema:
            value, _ = _read_object(Path(primary["path"]), f"{step} primary artifact")
            if value.get("schema_version") != expected_schema:
                raise ValueError(f"unsupported primary artifact schema for {step}")
        if step == "terminal_memory":
            value, _ = _read_object(Path(primary["path"]), "terminal memory")
            expected = (
                "deepread-preservation-memory-v1"
                if current["outcome"] == "accepted"
                else "deepread-repair-memory-v1"
            )
            if value.get("schema_version") != expected:
                raise ValueError("terminal memory does not match iteration outcome")
    manifest_data = (root / "manifest.json").read_bytes()
    previous_data = _event_files(root)[-1].read_bytes() if events else manifest_data
    event = {
        "schema_version": "deepread-iteration-event-v1",
        "iteration_id": current["iteration_id"],
        "sequence": len(events) + 1,
        "previous_sha256": _sha_bytes(previous_data),
        "step": step,
        "status": status,
        "artifacts": refs,
        **({"error": error} if error else {}),
    }
    event["event_sha256"] = _sha_bytes(_canonical(event))
    _write_exclusive(root / "events" / f"{event['sequence']:04d}.json", event)
    return event
