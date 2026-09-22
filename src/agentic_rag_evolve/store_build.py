"""Reproducible Markdown store construction with source provenance."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from agentic_rag_evolve.contracts import DocumentInput
from agentic_rag_evolve.evolution.audit import verify_candidate_snapshot
from systems.deepread.DeepRead.ports import EmbeddingModel
from systems.deepread.ingestion import MarkdownIngestor


STORE_SOURCE_PATHS = (
    "systems/deepread/ingestion.py",
    "systems/deepread/DeepRead/index/markdown_parser.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_documents(path: Path) -> tuple[DocumentInput, ...]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise ValueError("document manifest must contain a non-empty JSON list or JSONL")
    documents = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ValueError(f"document manifest item {index} must be an object")
        raw_path = Path(str(raw.get("path") or ""))
        source = raw_path if raw_path.is_absolute() else (path.parent / raw_path).resolve()
        documents.append(
            DocumentInput(
                document_id=str(raw.get("document_id") or ""),
                path=source,
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    ids = [item.document_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("document manifest contains duplicate document_id values")
    return tuple(documents)


def _load_candidate_provenance(
    candidate_audit_path: Path | None, *, source_root: Path
) -> dict[str, Any] | None:
    if candidate_audit_path is None:
        return None
    path = Path(candidate_audit_path).resolve()
    data = path.read_bytes()
    audit = json.loads(data)
    if not isinstance(audit, Mapping) or audit.get("schema_version") != "deepread-candidate-audit-v1":
        raise ValueError("unsupported candidate audit schema")
    if not audit.get("passed"):
        raise ValueError("candidate audit must pass before building a candidate store")
    if Path(str(audit.get("candidate_path") or "")).resolve() != source_root:
        raise ValueError("candidate audit path does not match store source root")
    verify_candidate_snapshot(source_root, audit)
    return {
        "candidate_id": audit.get("candidate_id"),
        "plan_id": audit.get("plan_id"),
        "candidate_snapshot_sha256": audit.get("candidate_snapshot_sha256"),
        "candidate_audit_path": str(path),
        "candidate_audit_sha256": hashlib.sha256(data).hexdigest(),
    }


def build_markdown_store(
    *,
    document_manifest_path: Path,
    output_path: Path,
    source_root: Path,
    embedder: EmbeddingModel | None,
    candidate_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Build a fresh store and bind every artifact to its documents and source snapshot."""

    document_manifest_path = Path(document_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError(f"store source root is not a directory: {source_root}")
    loaded_ingestion = Path(inspect.getfile(MarkdownIngestor)).resolve()
    expected_ingestion = (source_root / "systems/deepread/ingestion.py").resolve()
    if loaded_ingestion != expected_ingestion:
        raise ValueError(
            "loaded ingestion code does not come from source_root; run the builder "
            "from the baseline/candidate checkout being evaluated"
        )
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"store output directory must be empty: {output_path}")

    source_files = []
    for relative in STORE_SOURCE_PATHS:
        source = (source_root / relative).resolve()
        if not source.is_file() or not source.is_relative_to(source_root):
            raise ValueError(f"store source file is missing: {relative}")
        source_files.append({"path": relative, "sha256": _sha256(source)})
    candidate = _load_candidate_provenance(
        candidate_audit_path, source_root=source_root
    )
    documents = _load_documents(document_manifest_path)
    document_refs = [
        {
            "document_id": item.document_id,
            "path": str(item.path.resolve()),
            "sha256": _sha256(item.path),
        }
        for item in documents
    ]

    output_path.mkdir(parents=True, exist_ok=True)
    results = MarkdownIngestor(output_path, embedder).ingest(documents)
    artifacts = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_path.iterdir())
        if path.is_file()
    ]
    manifest = {
        "schema_version": "deepread-store-build-v1",
        "document_manifest": {
            "path": str(document_manifest_path),
            "sha256": _sha256(document_manifest_path),
        },
        "source_root": str(source_root),
        "source_files": source_files,
        "candidate": candidate,
        "embedding_model": embedder.model_name if embedder is not None else None,
        "documents": document_refs,
        "document_count": len(results),
        "paragraph_count": sum(item.paragraph_count for item in results),
        "artifacts": artifacts,
    }
    manifest_path = output_path / "STORE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_store_manifest(store_path: Path, manifest_path: Path) -> dict[str, Any]:
    store_path = Path(store_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "deepread-store-build-v1":
        raise ValueError("unsupported DeepRead store manifest schema")
    if manifest_path.parent != store_path:
        raise ValueError("store manifest must be located directly inside the store")
    artifacts = manifest.get("artifacts") or []
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("store manifest contains no artifacts")
    names = [str(item.get("path") or "") for item in artifacts]
    if len(names) != len(set(names)):
        raise ValueError("store manifest contains duplicate artifact paths")
    actual_names = sorted(
        path.name
        for path in store_path.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    if sorted(names) != actual_names:
        raise ValueError("store files do not exactly match the store manifest")
    for artifact in artifacts:
        relative = Path(str(artifact.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("store manifest contains an unsafe artifact path")
        path = (store_path / relative).resolve()
        if not path.is_relative_to(store_path) or not path.is_file():
            raise ValueError(f"store artifact is missing: {relative}")
        if _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"store artifact hash mismatch: {relative}")
    return manifest
