"""Lossless framework-agent logs, outside the evolvable target's source surface.

One model call is one event; one tool call contains both arguments and result.
Request history uses a common-prefix/delta encoding, never truncation. The file
is atomically checkpointed before I/O, so an interrupted call remains visible.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

from .context import trace_context

_SESSION: ContextVar["AgentTrace | None"] = ContextVar("evolution_trace", default=None)


class AgentTrace:
    def __init__(self, path: Path, role: str, model: Any, metadata: dict | None = None):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite agent trace: {self.path}")
        self.model = model
        self.previous_messages: list = []
        self.current: dict | None = None
        self.value = {
            "schema_version": "evolution-agent-trace-v1", "role": role,
            "model": getattr(model, "model_name", None),
            "provider_settings": {key: getattr(model, key, None) for key in (
                "timeout", "max_retries", "default_max_output_tokens",
                "retry_base_seconds", "retry_max_seconds")},
            "metadata": metadata or {}, "status": "running", "events": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(self.path)

    def log(self, event: str, **fields: Any) -> None:
        # HTTP attempt/retry events belong to the same logical model call.
        if self.current is not None:
            self.current.setdefault("transport", []).append({"event": event, **fields})
            self.save()

    def complete(self, payload: dict) -> Any:
        request = copy.deepcopy(dict(payload))
        messages = request.pop("messages", [])
        prefix = 0
        for old, new in zip(self.previous_messages, messages):
            if old != new:
                break
            prefix += 1
        event = {
            "type": "model", "status": "pending", "request": request,
            "messages_prefix": prefix, "messages_delta": messages[prefix:],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.previous_messages = messages
        self.value["events"].append(event)
        self.current = event
        self.save()
        start = time.monotonic()
        try:
            response = self.model.complete(payload)
            event.update(status="ok", response=copy.deepcopy(response))
            return response
        except BaseException as exc:
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event["elapsed_seconds"] = time.monotonic() - start
            self.current = None
            self.save()

    def __getattr__(self, key: str) -> Any:
        return getattr(self.model, key)

    def finish(self, outcome: Any) -> None:
        self.value["status"] = "finished"
        self.value["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.value["outcome"] = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
        self.save()


def record_agent_tool(*, name: str, arguments: Any, result: Any,
                      call_id: str | None = None, **fields: Any) -> None:
    session = _SESSION.get()
    if session is not None:
        session.value["events"].append(copy.deepcopy({
            "type": "tool", "name": name, "call_id": call_id,
            "arguments": arguments, "result": result, **fields,
        }))
        session.save()


@contextmanager
def traced_model(path: Path, role: str, model: Any, metadata: dict | None = None
                 ) -> Iterator[AgentTrace]:
    """Also usable for a batch of evaluator calls without an agent tool loop."""
    session = AgentTrace(path, role, model, metadata)
    token = _SESSION.set(session)
    try:
        with trace_context(session):
            yield session
    except BaseException as exc:
        # Do not create logs for rejected preconditions that made no API/tool call.
        if session.path.exists():
            session.value.update(status="error", error=f"{type(exc).__name__}: {exc}")
            session.save()
        raise
    finally:
        _SESSION.reset(token)


def traced_agent(role: str, *, directory_output: bool = False):
    """Lazy recording preserves each agent's existing non-overwrite guards."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            output = Path(bound.arguments["output_path"])
            path = output / "agent_trace.json" if directory_output else output.with_suffix(".trace.json")
            source = Path(inspect.getfile(function))
            metadata = {
                "implementation": str(source),
                "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "inputs": {key: {"path": str(value),
                                  "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()
                                  if Path(value).is_file() else None}
                           for key, value in bound.arguments.items()
                           if key.endswith("_path") and key != "output_path" and value is not None},
                "budgets": {key: value for key, value in bound.arguments.items()
                            if key.startswith("max_")},
            }
            with traced_model(path, role, bound.arguments["model"], metadata) as session:
                bound.arguments["model"] = session
                result = function(*bound.args, **bound.kwargs)
                session.finish(result)
                return result
        return run
    return decorate


def reconstruct_requests(trace: dict) -> Iterator[dict]:
    """Reconstruct exact agent-to-provider payloads in model-call order."""
    previous: list = []
    for event in trace["events"]:
        if event["type"] == "model":
            previous = previous[:event["messages_prefix"]] + event["messages_delta"]
            yield copy.deepcopy({**event["request"], "messages": previous})
