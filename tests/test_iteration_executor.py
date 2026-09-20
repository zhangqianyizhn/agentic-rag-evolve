import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.orchestration import (
    execute_iteration,
    initialize_iteration,
    read_iteration_status,
)


class FakeRunner:
    def __init__(self, root: Path, failures: set[str] | None = None):
        self.root = root
        self.failures = set(failures or set())
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        step = command[-1]
        if step in self.failures:
            self.failures.remove(step)
            return subprocess.CompletedProcess(command, 7, stdout="", stderr="temporary")
        (self.root / f"{step}.json").write_text(
            json.dumps({"schema_version": f"test-{step}-v1"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def _runbook(root: Path) -> Path:
    path = root / "runbook.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepread-iteration-runbook-v1",
                "iteration_id": "iteration-1",
                "workspace": str(root),
                "steps": {
                    "baseline_run": {
                        "command": ["fake", "baseline_run"],
                        "artifacts": {"primary": "baseline_run.json"},
                    },
                    "evaluation": {
                        "command": ["fake", "evaluation"],
                        "artifacts": {"primary": "evaluation.json"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class IterationExecutorTest(unittest.TestCase):
    def _init(self, root: Path) -> Path:
        ledger = root / "ledger"
        initialize_iteration(
            root=ledger,
            iteration_id="iteration-1",
            baseline_id="baseline-1",
            baseline_commit="a" * 40,
        )
        return ledger

    def test_step_limit_then_resume_without_rerunning_completed_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = self._init(root)
            runbook = _runbook(root)
            runner = FakeRunner(root)
            first = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=1,
                command_runner=runner,
            )
            second = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=1,
                command_runner=runner,
            )

        self.assertEqual(first.status, "step_limit_reached")
        self.assertEqual(first.next_step, "evaluation")
        self.assertEqual(second.next_step, "trajectories")
        self.assertEqual([call[0][-1] for call in runner.calls], ["baseline_run", "evaluation"])

    def test_failed_command_is_recorded_and_retry_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = self._init(root)
            runbook = _runbook(root)
            runner = FakeRunner(root, {"baseline_run"})
            failed = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=1,
                command_runner=runner,
            )
            retried = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=1,
                command_runner=runner,
            )
            status = read_iteration_status(root=ledger)

        self.assertEqual(failed.status, "step_failed")
        self.assertEqual(retried.next_step, "evaluation")
        self.assertEqual(status["failed_attempt_count"], 1)
        self.assertEqual(status["event_count"], 2)

    def test_missing_step_configuration_pauses_without_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = self._init(root)
            runbook = _runbook(root)
            runner = FakeRunner(root)
            report = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=3,
                command_runner=runner,
            )

        self.assertEqual(report.status, "awaiting_configuration")
        self.assertEqual(report.next_step, "trajectories")
        self.assertEqual(report.steps_completed, 2)

    def test_missing_declared_artifact_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = self._init(root)
            runbook = _runbook(root)

            def runner(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            report = execute_iteration(
                ledger_root=ledger,
                runbook_path=runbook,
                max_steps=1,
                command_runner=runner,
            )
            status = read_iteration_status(root=ledger)

        self.assertEqual(report.status, "step_failed")
        self.assertEqual(status["next_step"], "baseline_run")
        self.assertEqual(status["failed_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
