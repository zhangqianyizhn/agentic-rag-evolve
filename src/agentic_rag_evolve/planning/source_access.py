"""Restricted, revision-aware source access for modification planning."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from agentic_rag_evolve.evolution import (
    candidate_snapshot_sha256,
    collect_changed_paths,
)


EVOLVABLE_ROOT = "systems/deepread/DeepRead/"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError(f"source path is outside planning scope: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PermissionError(f"source path is outside planning scope: {relative}")
    return resolved


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return value, hashlib.sha256(data).hexdigest()


class PlanningSourceReader:
    """Expose only hypothesis-cited DeepRead files and detect source drift."""

    MAX_SOURCE_LINES = 240
    MAX_SOURCE_CHARS = 30_000

    def __init__(
        self,
        *,
        source_root: Path,
        cohort: Mapping[str, Any],
        hypotheses: Mapping[str, Any],
        candidate_audit_path: Path | None = None,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        if not self.source_root.is_dir():
            raise ValueError(f"planning source root is not a directory: {self.source_root}")
        self._candidate_audit: dict[str, Any] | None = None
        self._candidate_audit_ref: dict[str, str] | None = None
        if candidate_audit_path is not None:
            audit_path = Path(candidate_audit_path).resolve()
            audit, digest = _read_object(audit_path, "candidate audit")
            if audit.get("schema_version") != "deepread-candidate-audit-v1":
                raise ValueError("unsupported candidate audit schema")
            if not audit.get("passed"):
                raise ValueError("candidate audit must pass before planning source access")
            candidate_path = str(audit.get("candidate_path") or "")
            if not candidate_path or Path(candidate_path).resolve() != self.source_root:
                raise ValueError("candidate audit path does not match planning source root")
            self._candidate_audit = audit
            self._candidate_audit_ref = {
                "path": str(audit_path),
                "sha256": digest,
            }

        diagnoses = {
            str(item.get("task_id") or ""): item
            for item in cohort.get("eligible_diagnoses") or []
            if isinstance(item, Mapping)
        }
        entries: dict[str, dict[str, Any]] = {}
        for hypothesis in hypotheses.get("hypotheses") or []:
            if not isinstance(hypothesis, Mapping):
                raise ValueError("improvement hypothesis must be an object")
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            for reference in hypothesis.get("affected_source_refs") or []:
                if not isinstance(reference, Mapping):
                    raise ValueError("affected source reference must be an object")
                task_id = str(reference.get("task_id") or "")
                try:
                    index = int(reference.get("index"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("affected source index must be an integer") from exc
                diagnosis = diagnoses.get(task_id)
                sources = (diagnosis or {}).get("affected_sources") or []
                if index < 0 or index >= len(sources):
                    raise ValueError("hypothesis references an unknown affected source")
                source = sources[index]
                path = str(source.get("path") or "")
                if not path.startswith(EVOLVABLE_ROOT):
                    raise PermissionError(
                        f"planning source is outside evolvable DeepRead root: {path}"
                    )
                resolved = _resolve_under(self.source_root, path)
                raw_path = self.source_root / path
                if not resolved.is_file() or raw_path.is_symlink():
                    raise ValueError(f"planning source is not a regular file: {path}")
                record = entries.setdefault(
                    path,
                    {
                        "path": path,
                        "symbols": set(),
                        "references": [],
                    },
                )
                symbol = str(source.get("symbol") or "").strip()
                if symbol:
                    record["symbols"].add(symbol)
                ref = {
                    "hypothesis_id": hypothesis_id,
                    "task_id": task_id,
                    "index": index,
                }
                if ref not in record["references"]:
                    record["references"].append(ref)
        if not entries:
            raise ValueError("planning source scope is empty")

        self._sources: dict[str, dict[str, Any]] = {}
        for path in sorted(entries):
            resolved = _resolve_under(self.source_root, path)
            data = resolved.read_bytes()
            text = data.decode("utf-8")
            raw = entries[path]
            self._sources[path] = {
                "path": path,
                "symbols": sorted(raw["symbols"]),
                "references": sorted(
                    raw["references"],
                    key=lambda item: (
                        item["hypothesis_id"], item["task_id"], item["index"]
                    ),
                ),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "line_count": len(text.splitlines()),
            }
        self.manifest_sha256 = _canonical_sha256(list(self._sources.values()))
        self.verify_revision()

    @property
    def revision(self) -> dict[str, Any]:
        if self._candidate_audit is not None:
            return {
                "kind": "candidate_snapshot",
                "candidate_id": self._candidate_audit.get("candidate_id"),
                "snapshot_sha256": self._candidate_audit.get(
                    "candidate_snapshot_sha256"
                ),
                "candidate_audit": dict(self._candidate_audit_ref or {}),
                "source_manifest_sha256": self.manifest_sha256,
            }
        return {
            "kind": "allowlisted_source_manifest",
            "source_manifest_sha256": self.manifest_sha256,
        }

    def catalog(self) -> list[dict[str, Any]]:
        self.verify_revision()
        return [dict(item) for item in self._sources.values()]

    def verify_revision(self) -> None:
        for relative, metadata in self._sources.items():
            source = _resolve_under(self.source_root, relative)
            raw_path = self.source_root / relative
            if not source.is_file() or raw_path.is_symlink():
                raise ValueError(f"planning source changed type: {relative}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"planning source changed after access was granted: {relative}")
        if self._candidate_audit is None:
            return
        head = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed_paths = collect_changed_paths(self.source_root)
        if head != self._candidate_audit.get("head_commit"):
            raise ValueError("candidate HEAD changed after static audit")
        if changed_paths != sorted(self._candidate_audit.get("changed_paths") or []):
            raise ValueError("candidate changed paths differ from static audit")
        snapshot = candidate_snapshot_sha256(
            self.source_root, head_commit=head, changed_paths=changed_paths
        )
        if snapshot != self._candidate_audit.get("candidate_snapshot_sha256"):
            raise ValueError("candidate snapshot changed after static audit")

    def read_source(
        self, path: str, *, start_line: int = 1, end_line: int = 240
    ) -> dict[str, Any]:
        if path not in self._sources:
            raise PermissionError(f"source path is not allowlisted for planning: {path}")
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid planning source line range")
        if end_line - start_line + 1 > self.MAX_SOURCE_LINES:
            raise ValueError(f"source read exceeds {self.MAX_SOURCE_LINES} lines")
        self.verify_revision()
        data = _resolve_under(self.source_root, path).read_bytes()
        lines = data.decode("utf-8").splitlines()
        if start_line > len(lines) + 1:
            raise ValueError(f"source start_line exceeds file length: {path}")
        content = "\n".join(lines[start_line - 1 : end_line])
        if len(content) > self.MAX_SOURCE_CHARS:
            raise ValueError(f"source read exceeds {self.MAX_SOURCE_CHARS} characters")
        return {
            "path": path,
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "total_lines": len(lines),
            "sha256": self._sources[path]["sha256"],
            "content": content,
        }
