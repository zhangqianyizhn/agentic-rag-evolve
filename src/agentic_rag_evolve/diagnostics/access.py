"""Restricted readers backing diagnosis-agent artifact tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .policy import DIAGNOSTIC_SOURCE_POLICY


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError(f"path is outside diagnostic scope: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PermissionError(f"path is outside diagnostic scope: {relative}")
    return resolved


class DiagnosticArtifactReader:
    MAX_SOURCE_LINES = 240
    MAX_SOURCE_CHARS = 30_000
    MAX_PAYLOAD_CHARS = 12_000

    def __init__(self, *, bundle_path: Path, source_root: Path) -> None:
        self.bundle_path = Path(bundle_path)
        self.bundle_root = self.bundle_path.parent
        self.source_root = Path(source_root)
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if self.bundle.get("schema_version") != "deepread-diagnostic-input-v1":
            raise ValueError("unsupported diagnostic bundle schema")
        access = self.bundle.get("access") or {}
        source_items = access.get("sources") or []
        payload_items = access.get("payloads") or []
        if len({item["path"] for item in source_items}) != len(source_items):
            raise ValueError("diagnostic bundle contains duplicate source paths")
        if len({item["path"] for item in payload_items}) != len(payload_items):
            raise ValueError("diagnostic bundle contains duplicate payload paths")
        self._sources = {item["path"]: item for item in source_items}
        self._payloads = {item["path"]: item for item in payload_items}
        allowed_sources = {spec.path for spec in DIAGNOSTIC_SOURCE_POLICY}
        if set(self._sources) != allowed_sources:
            raise ValueError("diagnostic bundle source manifest does not match source policy")
        referenced_payloads = {
            str(reference["path"])
            for turn in (self.bundle.get("trajectory") or {}).get("turns") or []
            for tool in turn.get("tools") or []
            if isinstance((reference := tool.get("result_ref")), dict) and reference.get("path")
        }
        if set(self._payloads) != referenced_payloads:
            raise ValueError("diagnostic bundle payload manifest does not match trajectory refs")

    def list_sources(self, component: str | None = None) -> list[dict[str, Any]]:
        sources = list(self._sources.values())
        if component is not None:
            sources = [item for item in sources if item.get("component") == component]
        return sources

    def read_source(self, path: str, *, start_line: int = 1, end_line: int = 240) -> dict[str, Any]:
        if path not in self._sources:
            raise PermissionError(f"source path is not allowlisted: {path}")
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid source line range")
        if end_line - start_line + 1 > self.MAX_SOURCE_LINES:
            raise ValueError(f"source read exceeds {self.MAX_SOURCE_LINES} lines")
        source = _resolve_under(self.source_root, path)
        data = source.read_bytes()
        metadata = self._sources[path]
        if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError(f"source changed since bundle creation: {path}")
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
            "content": content,
        }

    def read_payload(
        self,
        path: str,
        *,
        offset_chars: int = 0,
        limit_chars: int = 12_000,
    ) -> dict[str, Any]:
        if path not in self._payloads:
            raise PermissionError(f"payload path is not referenced by this bundle: {path}")
        if offset_chars < 0 or limit_chars < 1 or limit_chars > self.MAX_PAYLOAD_CHARS:
            raise ValueError("invalid payload character range")
        payload = _resolve_under(self.bundle_root, path)
        data = payload.read_bytes()
        metadata = self._payloads[path]
        if len(data) != int(metadata["bytes"]):
            raise ValueError(f"payload byte count mismatch: {path}")
        if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError(f"payload sha256 mismatch: {path}")
        text = data.decode("utf-8")
        if offset_chars > len(text):
            raise ValueError(f"payload offset exceeds content length: {path}")
        end = min(len(text), offset_chars + limit_chars)
        return {
            "path": path,
            "offset_chars": offset_chars,
            "end_chars": end,
            "total_chars": len(text),
            "has_more": end < len(text),
            "content": text[offset_chars:end],
        }
