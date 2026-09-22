import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.orchestration import (
    append_iteration_event,
    initialize_iteration,
    read_iteration_status,
)
from agentic_rag_evolve.orchestration.ledger import (
    ACCEPTED_TAIL,
    BASE_STEPS,
    REJECTED_TAIL,
    STEP_SCHEMAS,
)


def _artifact(
    root: Path,
    step: str,
    *,
    outcome: str | None = None,
    requires_store_rebuild: bool = False,
) -> Path:
    schema = STEP_SCHEMAS.get(step, f"test-{step}-v1")
    if step == "terminal_memory":
        schema = (
            "deepread-preservation-memory-v1"
            if outcome == "accepted"
            else "deepread-repair-memory-v1"
        )
    value = {"schema_version": schema, "step": step}
    if step == "modification_plan":
        value["plans"] = [
            {
                "decision": "proceed",
                "edit_scope": {"requires_store_rebuild": requires_store_rebuild},
            }
        ]
    if step == "outcome":
        value["outcome"] = outcome
    path = root / f"{step}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class IterationLedgerTest(unittest.TestCase):
    def _init(self, root: Path) -> None:
        initialize_iteration(
            root=root,
            iteration_id="iteration-1",
            baseline_id="baseline-1",
            baseline_commit="a" * 40,
        )

    def test_failed_attempt_does_not_advance_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ledger"
            self._init(root)
            append_iteration_event(
                root=root,
                step="baseline_run",
                status="failed",
                error="temporary rate limit",
            )
            interim = read_iteration_status(root=root)
            artifact = _artifact(Path(directory), "baseline_run")
            append_iteration_event(
                root=root,
                step="baseline_run",
                status="completed",
                artifacts={"primary": artifact},
            )
            final = read_iteration_status(root=root)

        self.assertEqual(interim["next_step"], "baseline_run")
        self.assertEqual(interim["failed_attempt_count"], 1)
        self.assertEqual(final["next_step"], "evaluation")
        self.assertEqual(final["event_count"], 2)

    def test_accepted_outcome_selects_materialization_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            for step in BASE_STEPS:
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={"primary": _artifact(base, step, outcome="accepted")},
                )
            self.assertEqual(read_iteration_status(root=root)["next_step"], "materialization")
            for step in ACCEPTED_TAIL:
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={"primary": _artifact(base, step, outcome="accepted")},
                )
            status = read_iteration_status(root=root)

        self.assertTrue(status["terminal"])
        self.assertEqual(status["outcome"], "accepted")

    def test_rejected_outcome_skips_git_mutation_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            for step in BASE_STEPS:
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={"primary": _artifact(base, step, outcome="rejected")},
                )
            self.assertEqual(read_iteration_status(root=root)["next_step"], "terminal_memory")
            for step in REJECTED_TAIL:
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={"primary": _artifact(base, step, outcome="rejected")},
                )
            status = read_iteration_status(root=root)

        self.assertTrue(status["terminal"])
        self.assertNotIn("materialization", status["completed_steps"])

    def test_empty_plan_short_circuits_to_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            through_plan = BASE_STEPS[: BASE_STEPS.index("modification_plan") + 1]
            for step in through_plan:
                artifact = _artifact(base, step)
                if step == "modification_plan":
                    artifact.write_text(
                        json.dumps(
                            {
                                "schema_version": "deepread-modification-plan-v1",
                                "plans": [],
                            }
                        )
                    )
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={"primary": artifact},
                )
            status = read_iteration_status(root=root)

        self.assertEqual(status["outcome"], "no_candidate")
        self.assertEqual(status["next_step"], "iteration_report")

    def test_ingestion_plan_inserts_candidate_store_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            through_tests = BASE_STEPS[: BASE_STEPS.index("fixed_tests") + 1]
            for step in through_tests:
                append_iteration_event(
                    root=root,
                    step=step,
                    status="completed",
                    artifacts={
                        "primary": _artifact(
                            base, step, requires_store_rebuild=True
                        )
                    },
                )
            self.assertEqual(read_iteration_status(root=root)["next_step"], "candidate_store")
            append_iteration_event(
                root=root,
                step="candidate_store",
                status="completed",
                artifacts={"primary": _artifact(base, "candidate_store")},
            )
            self.assertEqual(read_iteration_status(root=root)["next_step"], "validation_gate")

    def test_artifact_tampering_breaks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            artifact = _artifact(base, "baseline_run")
            append_iteration_event(
                root=root,
                step="baseline_run",
                status="completed",
                artifacts={"primary": artifact},
            )
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                read_iteration_status(root=root)

    def test_out_of_order_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "ledger"
            self._init(root)
            with self.assertRaisesRegex(ValueError, "expected iteration step"):
                append_iteration_event(
                    root=root,
                    step="evaluation",
                    status="completed",
                    artifacts={"primary": _artifact(base, "evaluation")},
                )


if __name__ == "__main__":
    unittest.main()
