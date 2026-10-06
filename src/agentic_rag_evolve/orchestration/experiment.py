"""One-command, recoverable execution of the initial DeepRead experiment flow."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from agentic_rag_evolve.deepread_runner import (
    load_financebench_queries,
    run_financebench,
)
from agentic_rag_evolve.diagnosis import run_diagnosis
from agentic_rag_evolve.diagnostics import (
    DiagnosticArtifactReader,
    build_diagnostic_bundle,
)
from agentic_rag_evolve.evaluation.evaluator import (
    evaluate_financebench,
    write_evaluation,
)
from agentic_rag_evolve.planning import (
    run_hypothesis_aggregation,
    run_modification_planning,
    write_hypothesis_cohort,
)
from agentic_rag_evolve.providers import (
    ProviderSettings,
    load_chat_model,
    load_embedding_model,
    load_provider_bundle,
)
from agentic_rag_evolve.reporting import build_no_candidate_iteration_report
from agentic_rag_evolve.store_build import (
    build_markdown_store,
    verify_store_manifest,
)
from agentic_rag_evolve.trajectory import compile_trajectories
from systems.deepread.runtime import DeepReadConfig


EXPERIMENT_SCHEMA = "deepread-experiment-v1"
STATE_SCHEMA = "deepread-experiment-state-v1"
REPORT_SCHEMA = "deepread-experiment-report-v1"
BASELINE_STAGES = (
    "preflight",
    "store",
    "baseline",
    "evaluation",
    "trajectories",
)
DIAGNOSIS_STAGES = BASELINE_STAGES + (
    "diagnostic_bundles",
    "diagnoses",
    "hypothesis_cohort",
    "hypotheses",
    "modification_plan",
    "terminal_report",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    path = Path(path).resolve()
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"experiment artifact is missing: {path}")
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_sha256(item).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:160]
    return safe or hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _git_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("source_root does not resolve to a Git commit")
    return revision


def _verify_frozen_diagnostic_sources(output: Path, source_root: Path) -> None:
    """Verify the current DeepRead edit surface against an existing bundle snapshot."""

    index_path = Path(output) / "diagnostic_bundles" / "index.json"
    index = _read_json(index_path)
    items = list(index.get("items") or [])
    if not items:
        raise ValueError("completed diagnostic bundle stage has no bundle items")
    reader = DiagnosticArtifactReader(
        bundle_path=Path(str(items[0]["bundle"])), source_root=source_root
    )
    for source in reader.list_sources():
        reader.read_source(str(source["path"]), start_line=1, end_line=1)


def _resume_manifest_compatible(
    *,
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    state: Mapping[str, Any],
    config: "ExperimentConfig",
) -> bool:
    """Allow framework-only updates after diagnostic inputs have been frozen."""

    stored_copy = json.loads(json.dumps(stored))
    current_copy = json.loads(json.dumps(current))
    stored_inputs = stored_copy.get("inputs") or {}
    current_inputs = current_copy.get("inputs") or {}
    stored_revision = stored_inputs.get("source_revision")
    current_revision = current_inputs.get("source_revision")
    if stored_revision == current_revision:
        return stored_copy == current_copy
    stored_inputs["source_revision"] = current_revision
    if stored_copy != current_copy:
        return False
    bundle_stage = (state.get("stages") or {}).get("diagnostic_bundles") or {}
    if config.mode != "diagnose" or bundle_stage.get("status") != "completed":
        return False
    _verify_frozen_diagnostic_sources(config.output, config.source_root)
    return True


def _document_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    values = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix.lower() == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(values, list) or not values:
        raise ValueError("documents manifest must contain a non-empty list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ValueError(f"documents manifest item {index} must be an object")
        document_id = str(raw.get("document_id") or "").strip()
        relative = Path(str(raw.get("path") or ""))
        source = relative if relative.is_absolute() else (path.parent / relative).resolve()
        if not document_id:
            raise ValueError(f"documents manifest item {index} has no document_id")
        if source.suffix.lower() not in {".md", ".markdown"} or not source.is_file():
            raise ValueError(f"Markdown document is missing or unsupported: {source}")
        records.append({"document_id": document_id, "path": str(source)})
    ids = [item["document_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("documents manifest contains duplicate document_id values")
    return tuple(records)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    source_root: Path
    documents: Path
    dataset: Path
    store: Path
    output: Path
    env_file: Path
    mode: str = "baseline"
    judge: bool = True
    limit: int | None = None
    task_ids: tuple[str, ...] = ()
    max_rounds: int = 50
    retrieval_topk: int = 5
    diagnosis_max_rounds: int = 12
    diagnosis_max_tool_calls: int | None = None
    diagnosis_max_output_tokens: int | None = None
    request_timeout: int = 1800
    request_max_retries: int = 3
    request_retry_base_seconds: float = 15.0
    request_retry_max_seconds: float = 120.0

    def __post_init__(self) -> None:
        for field in ("source_root", "documents", "dataset", "store", "output", "env_file"):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())
        object.__setattr__(self, "task_ids", tuple(str(item) for item in self.task_ids))
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if self.mode not in {"baseline", "diagnose"}:
            raise ValueError("mode must be baseline or diagnose")
        if self.mode == "diagnose" and not self.judge:
            raise ValueError("diagnose mode requires the answer judge")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.max_rounds < 1 or self.retrieval_topk < 1:
            raise ValueError("DeepRead round and retrieval limits must be positive")

    @property
    def stages(self) -> tuple[str, ...]:
        return DIAGNOSIS_STAGES if self.mode == "diagnose" else BASELINE_STAGES

    def manifest_config(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("source_root", "documents", "dataset", "store", "output", "env_file"):
            value[key] = str(value[key])
        value["task_ids"] = list(self.task_ids)
        return value


class ExperimentCheckpoint:
    """Persist stage completion and verify artifacts before resuming."""

    def __init__(self, config: ExperimentConfig, *, resume: bool) -> None:
        self.config = config
        self.root = config.output
        self.manifest_path = self.root / "experiment_manifest.json"
        self.state_path = self.root / "experiment_state.json"
        manifest = {
            "schema_version": EXPERIMENT_SCHEMA,
            "experiment_id": config.experiment_id,
            "config": config.manifest_config(),
            "inputs": {
                "documents_sha256": _sha256(config.documents),
                "dataset_sha256": _sha256(config.dataset),
                "source_revision": _git_revision(config.source_root),
            },
            "stages": list(config.stages),
        }
        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise FileExistsError(
                    f"experiment output is non-empty; use --resume: {self.root}"
                )
            if not self.manifest_path.is_file() or not self.state_path.is_file():
                raise ValueError("cannot resume an output without experiment checkpoint files")
            stored_manifest = _read_json(self.manifest_path)
            self.state = _read_json(self.state_path)
            if not _resume_manifest_compatible(
                stored=stored_manifest,
                current=manifest,
                state=self.state,
                config=config,
            ):
                raise ValueError("experiment configuration or frozen inputs changed since start")
            if self.state.get("schema_version") != STATE_SCHEMA:
                raise ValueError("unsupported experiment state schema")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            _write_json(self.manifest_path, manifest)
            self.state = {
                "schema_version": STATE_SCHEMA,
                "experiment_id": config.experiment_id,
                "status": "running",
                "next_stage": config.stages[0],
                "stages": {},
            }
            self._save()

    def _save(self) -> None:
        _write_json(self.state_path, self.state)

    def _verify_completed(self, stage: str) -> None:
        record = (self.state.get("stages") or {}).get(stage) or {}
        for reference in (record.get("artifacts") or {}).values():
            path = Path(str(reference.get("path") or ""))
            if _path_digest(path) != reference.get("sha256"):
                raise ValueError(f"completed experiment artifact changed: {path}")

    def run(
        self,
        stage: str,
        action: Callable[[], tuple[Mapping[str, Path], Mapping[str, Any]]],
    ) -> bool:
        record = (self.state.get("stages") or {}).get(stage) or {}
        if record.get("status") == "completed":
            self._verify_completed(stage)
            return False
        self.state["status"] = "running"
        self.state["next_stage"] = stage
        self.state.setdefault("stages", {})[stage] = {"status": "running"}
        self._save()
        try:
            artifacts, details = action()
            references = {
                name: {"path": str(Path(path).resolve()), "sha256": _path_digest(path)}
                for name, path in sorted(artifacts.items())
            }
            if "primary" not in references:
                raise ValueError(f"stage {stage} did not return a primary artifact")
            self.state["stages"][stage] = {
                "status": "completed",
                "artifacts": references,
                "details": dict(details),
            }
            completed = [
                item
                for item in self.config.stages
                if (self.state["stages"].get(item) or {}).get("status") == "completed"
            ]
            next_index = len(completed)
            self.state["next_stage"] = (
                self.config.stages[next_index]
                if next_index < len(self.config.stages)
                else None
            )
            self.state["status"] = (
                "completed" if self.state["next_stage"] is None else "running"
            )
            self._save()
            return True
        except BaseException as exc:
            self.state["status"] = "failed"
            self.state["next_stage"] = stage
            self.state["stages"][stage] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._save()
            raise


def _preflight(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    if not config.source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {config.source_root}")
    if not config.env_file.is_file():
        raise FileNotFoundError(f"environment file not found: {config.env_file}")
    settings = ProviderSettings.from_env(config.env_file)
    documents = _document_manifest(config.documents)
    queries = load_financebench_queries(config.dataset)
    selected = queries
    if config.task_ids:
        by_id = {item.task_id: item for item in queries}
        missing = [item for item in config.task_ids if item not in by_id]
        if missing:
            raise ValueError(f"task IDs not found in dataset: {missing}")
        selected = tuple(by_id[item] for item in config.task_ids)
    if config.limit is not None:
        selected = selected[: config.limit]
    document_ids = {item["document_id"] for item in documents}
    missing_documents = sorted({item.sample_id for item in selected} - document_ids)
    if missing_documents:
        raise ValueError(f"selected tasks reference missing documents: {missing_documents}")
    report = {
        "schema_version": "deepread-experiment-preflight-v1",
        "experiment_id": config.experiment_id,
        "source_revision": _git_revision(config.source_root),
        "mode": config.mode,
        "document_count": len(documents),
        "dataset_query_count": len(queries),
        "selected_query_count": len(selected),
        "models": {
            "chat": settings.llm_model,
            "embedding": settings.embedding_model,
            "reranker": settings.rerank_model,
        },
        "paths": {
            "store": str(config.store),
            "output": str(config.output),
        },
        "secrets_recorded": False,
    }
    path = config.output / "preflight.json"
    _write_json(path, report)
    return {"primary": path}, {"selected_query_count": len(selected)}


def _store(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    manifest_path = config.store / "STORE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = verify_store_manifest(config.store, manifest_path)
        return {"primary": manifest_path, "store": config.store}, {
            "reused": True,
            "document_count": manifest.get("document_count"),
        }
    if config.store.exists() and any(config.store.iterdir()):
        raise FileExistsError(
            f"store is non-empty but has no valid STORE_MANIFEST.json: {config.store}"
        )
    manifest = build_markdown_store(
        document_manifest_path=config.documents,
        output_path=config.store,
        source_root=config.source_root,
        embedder=load_embedding_model(config.env_file),
    )
    return {"primary": manifest_path, "store": config.store}, {
        "reused": False,
        "document_count": manifest.get("document_count"),
        "paragraph_count": manifest.get("paragraph_count"),
    }


def _baseline(
    config: ExperimentConfig, progress: Callable[[str], None] | None = None
) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    output = config.output / "baseline"
    required = tuple(
        output / name
        for name in (
            "manifest.json",
            "predictions.jsonl",
            "deepread_trace.jsonl",
            "summary.json",
        )
    )
    if output.exists() and any(output.iterdir()):
        if not all(path.is_file() for path in required):
            raise FileExistsError(f"incomplete baseline output cannot be reused: {output}")
        summary = _read_json(output / "summary.json")
    else:
        summary = run_financebench(
            dataset_path=config.dataset,
            store_path=config.store,
            output_path=output,
            providers=load_provider_bundle(config.env_file),
            config=DeepReadConfig(
                max_rounds=config.max_rounds,
                retrieval_topk=config.retrieval_topk,
            ),
            limit=config.limit,
            task_ids=config.task_ids or None,
            store_manifest_path=config.store / "STORE_MANIFEST.json",
            progress_callback=(
                lambda index, total, query, status: progress(
                    f"baseline {index}/{total} task={query.task_id} status={status}"
                )
                if progress is not None
                else None
            ),
        )
    return {
        "primary": output / "manifest.json",
        "predictions": output / "predictions.jsonl",
        "trace": output / "deepread_trace.jsonl",
        "summary": output / "summary.json",
    }, summary


def _evaluation(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    output = config.output / "evaluation"
    summary_path = output / "evaluation_summary.json"
    details_path = output / "evaluation.json"
    if output.exists() and any(output.iterdir()):
        if not summary_path.is_file() or not details_path.is_file():
            raise FileExistsError(f"incomplete evaluation output cannot be reused: {output}")
        summary = _read_json(summary_path)
    else:
        judge_model = (
            load_chat_model(
                config.env_file,
                timeout=config.request_timeout,
                max_retries=config.request_max_retries,
                retry_base_seconds=config.request_retry_base_seconds,
                retry_max_seconds=config.request_retry_max_seconds,
            )
            if config.judge
            else None
        )
        details, value = evaluate_financebench(
            dataset_path=config.dataset,
            prediction_path=config.output / "baseline" / "predictions.jsonl",
            judge_model=judge_model,
        )
        write_evaluation(output, details, value)
        summary = value.to_dict()
    return {"primary": summary_path, "details": details_path}, summary


def _trajectories(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    output = config.output / "trajectories"
    report_path = output / "compilation_report.json"
    if output.exists() and any(output.iterdir()):
        if not report_path.is_file():
            raise FileExistsError(f"incomplete trajectory output cannot be reused: {output}")
        report = _read_json(report_path)
    else:
        value = compile_trajectories(
            trace_path=config.output / "baseline" / "deepread_trace.jsonl",
            prediction_path=config.output / "baseline" / "predictions.jsonl",
            output_path=output,
        )
        report = value.to_dict()
    if report.get("unassigned_event_count"):
        raise ValueError("trajectory compilation left unassigned trace events")
    return {"primary": report_path, "trajectories": output}, report


def _diagnostic_bundles(
    config: ExperimentConfig, progress: Callable[[str], None] | None = None
) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    trajectory_root = config.output / "trajectories"
    report = _read_json(trajectory_root / "compilation_report.json")
    output = config.output / "diagnostic_bundles"
    output.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    filenames = list(report.get("output_files") or [])
    for index, filename in enumerate(filenames, start=1):
        trajectory_path = trajectory_root / str(filename)
        trajectory = _read_json(trajectory_path)
        task_id = str(trajectory.get("task_id") or "")
        task_output = output / _safe_name(task_id)
        bundle_path = task_output / "bundle.json"
        if bundle_path.is_file():
            bundle = _read_json(bundle_path)
            if bundle.get("schema_version") != "deepread-diagnostic-input-v1":
                raise ValueError(f"invalid existing diagnostic bundle: {bundle_path}")
        else:
            if task_output.exists() and any(task_output.iterdir()):
                raise FileExistsError(f"incomplete diagnostic bundle: {task_output}")
            build_diagnostic_bundle(
                trajectory_path=trajectory_path,
                evaluation_path=config.output / "evaluation" / "evaluation.json",
                run_manifest_path=config.output / "baseline" / "manifest.json",
                store_path=config.store,
                source_root=config.source_root,
                output_path=task_output,
            )
            bundle = _read_json(bundle_path)
        signals = bundle.get("failure_signals") or {}
        items.append({
            "task_id": task_id,
            "bundle": str(bundle_path.resolve()),
            "triage": signals.get("triage"),
            "diagnosis_eligible": bool(
                (signals.get("diagnosis_route") or {}).get("eligible")
            ),
        })
        if progress is not None:
            progress(
                f"diagnostic_bundles {index}/{len(filenames)} task={task_id} "
                f"triage={signals.get('triage')}"
            )
    index = {
        "schema_version": "deepread-diagnostic-bundle-index-v1",
        "experiment_id": config.experiment_id,
        "count": len(items),
        "items": items,
    }
    index_path = output / "index.json"
    _write_json(index_path, index)
    return {"primary": index_path, "bundles": output}, {"count": len(items)}


def _completed_diagnosis(task_root: Path) -> tuple[str, Path, Path | None] | None:
    for attempt in sorted(task_root.glob("attempt-*"), reverse=True):
        audit_path = attempt / "audit.json"
        if not audit_path.is_file():
            continue
        audit = _read_json(audit_path)
        status = str(audit.get("status") or "")
        diagnosis_path = attempt / "diagnosis.json"
        if status == "ok" and diagnosis_path.is_file():
            return status, audit_path, diagnosis_path
        if status == "skipped":
            return status, audit_path, None
    return None


def _diagnoses(
    config: ExperimentConfig, progress: Callable[[str], None] | None = None
) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    bundle_index = _read_json(config.output / "diagnostic_bundles" / "index.json")
    output = config.output / "diagnoses"
    output.mkdir(parents=True, exist_ok=True)
    model = load_chat_model(
        config.env_file,
        timeout=config.request_timeout,
        max_retries=config.request_max_retries,
        retry_base_seconds=config.request_retry_base_seconds,
        retry_max_seconds=config.request_retry_max_seconds,
    )
    items: list[dict[str, Any]] = []
    bundle_items = list(bundle_index.get("items") or [])
    for index, item in enumerate(bundle_items, start=1):
        task_id = str(item["task_id"])
        task_root = output / _safe_name(task_id)
        task_root.mkdir(parents=True, exist_ok=True)
        completed = _completed_diagnosis(task_root)
        if completed is None:
            recovery_path = next((
                previous
                for previous in sorted(task_root.glob("attempt-*"), reverse=True)
                if (previous / "candidate.json").is_file()
                and (previous / "audit.json").is_file()
                and _read_json(previous / "audit.json").get("status")
                in {"validation_error", "error", "interrupted", "running"}
            ), None)
            attempt_number = len(tuple(task_root.glob("attempt-*"))) + 1
            attempt = task_root / f"attempt-{attempt_number:04d}"
            result = run_diagnosis(
                bundle_path=Path(str(item["bundle"])),
                source_root=config.source_root,
                output_path=attempt,
                model=model,
                max_rounds=config.diagnosis_max_rounds,
                max_tool_calls=config.diagnosis_max_tool_calls,
                max_output_tokens=config.diagnosis_max_output_tokens,
                recovery_path=recovery_path,
            )
            if result.status not in {"ok", "skipped"}:
                raise RuntimeError(
                    f"diagnosis failed for {task_id}: {result.status}; rerun with --resume"
                )
            completed = _completed_diagnosis(task_root)
        if completed is None:
            raise RuntimeError(f"diagnosis produced no completed artifact for {task_id}")
        status, audit_path, diagnosis_path = completed
        items.append({
            "task_id": task_id,
            "status": status,
            "audit": str(audit_path.resolve()),
            "diagnosis": str(diagnosis_path.resolve()) if diagnosis_path else None,
        })
        if progress is not None:
            progress(
                f"diagnoses {index}/{len(bundle_items)} task={task_id} status={status}"
            )
        _write_json(
            output / "index.json",
            {
                "schema_version": "deepread-diagnosis-index-v1",
                "experiment_id": config.experiment_id,
                "count": len(items),
                "items": items,
            },
        )
    index_path = output / "index.json"
    return {"primary": index_path, "diagnoses": output}, {
        "count": len(items),
        "diagnosed": sum(item["status"] == "ok" for item in items),
        "skipped": sum(item["status"] == "skipped" for item in items),
    }


def _hypothesis_cohort(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    index = _read_json(config.output / "diagnoses" / "index.json")
    diagnosis_paths = [Path(item["diagnosis"]) for item in index["items"] if item.get("diagnosis")]
    audit_paths = [
        Path(item["audit"])
        for item in index["items"]
        if item.get("status") == "skipped"
    ]
    output = config.output / "planning" / "cohort.json"
    if output.is_file():
        cohort = _read_json(output)
    else:
        cohort = write_hypothesis_cohort(
            diagnosis_paths, output, diagnosis_audit_paths=audit_paths
        )
    return {"primary": output}, dict(cohort.get("counts") or {})


def _prepare_planning_retry(
    output: Path, *, audit_schema: str, successful_statuses: set[str]
) -> None:
    """Preserve incomplete attempts before reusing the canonical stage filenames."""

    audit_path = output.with_suffix(".audit.json")
    candidate_path = output.with_suffix(".candidate.json")
    existing = [path for path in (output, audit_path, candidate_path) if path.exists()]
    if not existing:
        return
    if not audit_path.is_file():
        raise ValueError(f"planning output has no audit: {output}")
    audit = _read_json(audit_path)
    if audit.get("schema_version") != audit_schema:
        raise ValueError(f"unsupported planning audit schema: {audit_path}")
    if audit.get("status") in successful_statuses:
        if not output.is_file():
            raise ValueError(f"successful planning audit has no result: {audit_path}")
        return
    if audit.get("status") not in {"error", "validation_error", "max_rounds", "running", "interrupted"}:
        raise ValueError(f"unsupported planning retry status: {audit.get('status')}")
    history = output.parent / "attempts" / output.stem
    numbers = [int(path.name.removeprefix("attempt-")) for path in history.glob("attempt-*")]
    attempt = history / f"attempt-{max(numbers, default=0) + 1:04d}"
    references = [{"path": path.name, "sha256": _sha256(path)} for path in existing]
    attempt.mkdir(parents=True)
    _write_json(attempt / "manifest.json", {
        "schema_version": "deepread-planning-retry-archive-v1",
        "stage": output.stem,
        "status": audit.get("status"),
        "artifacts": references,
    })
    for path in existing:
        path.rename(attempt / path.name)


def _planning_artifacts(output: Path) -> dict[str, Path]:
    artifacts = {"primary": output, "audit": output.with_suffix(".audit.json")}
    history = output.parent / "attempts" / output.stem
    if history.is_dir():
        artifacts["attempt_history"] = history
    return artifacts


def _hypotheses(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    output = config.output / "planning" / "hypotheses.json"
    audit = output.with_suffix(".audit.json")
    _prepare_planning_retry(
        output, audit_schema="deepread-hypothesis-audit-v1",
        successful_statuses={"ok", "no_eligible_diagnoses"},
    )
    if output.is_file() and audit.is_file():
        value = _read_json(output)
        details = {
            "status": value.get("status"),
            "hypothesis_count": len(value.get("hypotheses") or []),
        }
    else:
        report = run_hypothesis_aggregation(
            cohort_path=config.output / "planning" / "cohort.json",
            output_path=output,
            model=load_chat_model(
                config.env_file,
                timeout=config.request_timeout,
                max_retries=config.request_max_retries,
                retry_base_seconds=config.request_retry_base_seconds,
                retry_max_seconds=config.request_retry_max_seconds,
            ),
            max_output_tokens=config.diagnosis_max_output_tokens,
        )
        if report.status not in {"ok", "no_eligible_diagnoses"}:
            raise RuntimeError(f"hypothesis aggregation failed: {report.status}")
        details = report.to_dict()
    return _planning_artifacts(output), details


def _modification_plan(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    output = config.output / "planning" / "plan.json"
    audit = output.with_suffix(".audit.json")
    _prepare_planning_retry(
        output, audit_schema="deepread-modification-plan-audit-v1",
        successful_statuses={"ok", "no_plannable_hypotheses"},
    )
    if output.is_file() and audit.is_file():
        value = _read_json(output)
        details = {"plan_count": len(value.get("plans") or [])}
    else:
        report = run_modification_planning(
            cohort_path=config.output / "planning" / "cohort.json",
            hypotheses_path=config.output / "planning" / "hypotheses.json",
            output_path=output,
            model=load_chat_model(
                config.env_file,
                timeout=config.request_timeout,
                max_retries=config.request_max_retries,
                retry_base_seconds=config.request_retry_base_seconds,
                retry_max_seconds=config.request_retry_max_seconds,
            ),
            max_output_tokens=config.diagnosis_max_output_tokens,
            source_root=config.source_root,
        )
        if report.status not in {"ok", "no_plannable_hypotheses"}:
            raise RuntimeError(f"modification planning failed: {report.status}")
        details = report.to_dict()
    return _planning_artifacts(output), details


def _terminal_report(config: ExperimentConfig) -> tuple[Mapping[str, Path], Mapping[str, Any]]:
    plan_path = config.output / "planning" / "plan.json"
    plan = _read_json(plan_path)
    proceeding = [
        item for item in plan.get("plans") or []
        if isinstance(item, Mapping) and item.get("decision") == "proceed"
    ]
    output = config.output / "experiment_report.json"
    if proceeding:
        report = {
            "schema_version": REPORT_SCHEMA,
            "experiment_id": config.experiment_id,
            "status": "candidate_planned",
            "proceeding_plan_count": len(proceeding),
            "plan_ids": [
                str(item.get("plan_id") or item.get("hypothesis_id") or "")
                for item in proceeding
            ],
            "next_action": (
                "Provide frozen development/holdout/cross-dataset validation cohorts before "
                "candidate execution and promotion."
            ),
            "final_test_accessed": False,
        }
        _write_json(output, report)
    else:
        report = build_no_candidate_iteration_report(
            cohort_path=config.output / "planning" / "cohort.json",
            hypotheses_path=config.output / "planning" / "hypotheses.json",
            plan_path=plan_path,
            output_path=output,
        )
    return {"primary": output}, {
        "status": "candidate_planned" if proceeding else "no_candidate",
        "proceeding_plan_count": len(proceeding),
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    checkpoint = ExperimentCheckpoint(config, resume=resume)
    actions: dict[str, Callable[[], tuple[Mapping[str, Path], Mapping[str, Any]]]] = {
        "preflight": lambda: _preflight(config),
        "store": lambda: _store(config),
        "baseline": lambda: _baseline(config, progress),
        "evaluation": lambda: _evaluation(config),
        "trajectories": lambda: _trajectories(config),
        "diagnostic_bundles": lambda: _diagnostic_bundles(config, progress),
        "diagnoses": lambda: _diagnoses(config, progress),
        "hypothesis_cohort": lambda: _hypothesis_cohort(config),
        "hypotheses": lambda: _hypotheses(config),
        "modification_plan": lambda: _modification_plan(config),
        "terminal_report": lambda: _terminal_report(config),
    }
    for stage in config.stages:
        if progress is not None:
            progress(f"stage={stage} status=starting")
        executed = checkpoint.run(stage, actions[stage])
        if progress is not None:
            progress(f"stage={stage} status={'completed' if executed else 'skipped'}")
    return dict(checkpoint.state)
