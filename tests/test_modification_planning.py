import copy
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.planning import (
    ModificationPlanValidationError,
    build_planning_memory_context,
    run_modification_planning,
    validate_modification_plan,
)


class FakeModel:
    model_name = "fake-plan-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return self.responses.pop(0)


def _cohort() -> dict:
    return {
        "schema_version": "deepread-hypothesis-cohort-v1",
        "cohort_id": "cohort-1",
        "eligible_diagnoses": [
            {
                "task_id": "q1",
                "affected_sources": [
                    {
                        "path": "systems/deepread/DeepRead/agent/runner.py",
                        "symbol": "run_agent",
                    }
                ],
            },
            {
                "task_id": "q2",
                "affected_sources": [
                    {
                        "path": "systems/deepread/DeepRead/agent/runner.py",
                        "symbol": "run_agent",
                    }
                ],
            },
        ],
    }


def _hypotheses(*, maturity: str = "recurring") -> dict:
    task_ids = ["q1", "q2"] if maturity == "recurring" else ["q1"]
    return {
        "schema_version": "deepread-improvement-hypotheses-v1",
        "cohort_id": "cohort-1",
        "status": "ready",
        "hypotheses": [
            {
                "hypothesis_id": "hyp_1",
                "maturity": maturity,
                "task_ids": task_ids,
                "affected_source_refs": [
                    {"task_id": "q1", "index": 0, "rationale": "Agent loop."}
                ],
            }
        ],
    }


def _plan(*, decision: str = "proceed") -> dict:
    development = ["q1", "q2"] if decision == "proceed" else []
    refs = (
        [{"task_id": "q1", "index": 0, "rationale": "Narrow active source."}]
        if decision == "proceed"
        else []
    )
    return {
        "schema_version": "deepread-modification-plan-v1",
        "cohort_id": "cohort-1",
        "hypothesis_set_status": "ready",
        "plans": [
            {
                "hypothesis_id": "hyp_1",
                "decision": decision,
                "rationale": "The recurring mechanism is grounded and narrowly scoped.",
                "allowed_source_refs": refs,
                "change_contract": {
                    "current_behavior": "The final answer is emitted without a value check.",
                    "required_behavior_delta": "Verify selected numeric claims against read evidence.",
                    "must_preserve": ["Existing retrieval behavior."],
                    "non_goals": ["Do not change indexing or provider transport."],
                },
                "validation_plan": {
                    "development_task_ids": development,
                    "holdout_selection_rules": ["Select unseen numeric questions with successful reads."],
                    "expected_observations": ["Target answers preserve evidence values."],
                    "rollback_conditions": ["Rollback if holdout accuracy decreases."],
                },
                "risk": {
                    "level": "medium",
                    "regression_scenarios": ["Extra checking could alter already-correct answers."],
                },
            }
        ],
    }


class ModificationPlanningTest(unittest.TestCase):
    def test_validator_resolves_paths_from_source_refs(self) -> None:
        result = validate_modification_plan(
            _plan(), cohort=_cohort(), hypotheses=_hypotheses()
        )

        item = result["plans"][0]
        self.assertEqual(
            item["edit_scope"]["allowed_paths"],
            ["systems/deepread/DeepRead/agent/runner.py"],
        )
        self.assertEqual(item["edit_scope"]["max_files_to_modify"], 1)
        self.assertTrue(item["edit_scope"]["must_inspect_before_edit"])

    def test_singleton_hypothesis_cannot_proceed(self) -> None:
        with self.assertRaisesRegex(
            ModificationPlanValidationError, "singleton hypothesis cannot proceed"
        ):
            validate_modification_plan(
                _plan(), cohort=_cohort(), hypotheses=_hypotheses(maturity="singleton")
            )

    def test_source_scope_cannot_expand_beyond_hypothesis(self) -> None:
        candidate = _plan()
        candidate["plans"][0]["allowed_source_refs"][0]["task_id"] = "q2"

        with self.assertRaisesRegex(
            ModificationPlanValidationError, "not an affected source ref"
        ):
            validate_modification_plan(
                candidate, cohort=_cohort(), hypotheses=_hypotheses()
            )

    def test_resolved_source_must_remain_under_deepread_root(self) -> None:
        cohort = _cohort()
        cohort["eligible_diagnoses"][0]["affected_sources"][0]["path"] = (
            "runner/run_deepread.py"
        )

        with self.assertRaisesRegex(
            ModificationPlanValidationError, "outside the evolvable DeepRead root"
        ):
            validate_modification_plan(
                _plan(), cohort=cohort, hypotheses=_hypotheses()
            )

    def test_proceeding_plan_must_cover_all_development_tasks(self) -> None:
        candidate = _plan()
        candidate["plans"][0]["validation_plan"]["development_task_ids"] = ["q1"]

        with self.assertRaisesRegex(
            ModificationPlanValidationError, "validate every hypothesis task"
        ):
            validate_modification_plan(
                candidate, cohort=_cohort(), hypotheses=_hypotheses()
            )

    def test_empty_hypothesis_set_skips_model(self) -> None:
        hypotheses = _hypotheses()
        hypotheses["status"] = "no_hypotheses"
        hypotheses["hypotheses"] = []
        model = FakeModel([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            hypotheses_path = root / "hypotheses.json"
            output_path = root / "plan.json"
            cohort_path.write_text(json.dumps(_cohort()))
            hypotheses_path.write_text(json.dumps(hypotheses))

            report = run_modification_planning(
                cohort_path=cohort_path,
                hypotheses_path=hypotheses_path,
                output_path=output_path,
                model=model,
            )
            result = json.loads(output_path.read_text())
            audit = json.loads((root / "plan.audit.json").read_text())

        self.assertEqual(report.status, "no_plannable_hypotheses")
        self.assertEqual(report.model_calls, 0)
        self.assertEqual(result["plans"], [])
        self.assertEqual(audit["status"], "no_plannable_hypotheses")
        self.assertEqual(model.calls, [])

    def test_agent_retries_one_invalid_scope(self) -> None:
        invalid = _plan()
        invalid["plans"][0]["allowed_source_refs"][0]["task_id"] = "q2"
        corrected = _plan()
        model = FakeModel(
            [
                {"choices": [{"message": {"content": json.dumps(invalid)}}]},
                {"choices": [{"message": {"content": json.dumps(corrected)}}]},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            hypotheses_path = root / "hypotheses.json"
            output_path = root / "plan.json"
            cohort_path.write_text(json.dumps(_cohort()))
            hypotheses_path.write_text(json.dumps(_hypotheses()))

            report = run_modification_planning(
                cohort_path=cohort_path,
                hypotheses_path=hypotheses_path,
                output_path=output_path,
                model=model,
            )
            candidate = json.loads((root / "plan.candidate.json").read_text())
            audit = json.loads((root / "plan.audit.json").read_text())

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.model_calls, 2)
        self.assertEqual(candidate, invalid)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(len(audit["events"]), 2)

    def test_agent_receives_audited_memory_context(self) -> None:
        model = FakeModel(
            [{"choices": [{"message": {"content": json.dumps(_plan())}}]}]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            hypotheses_path = root / "hypotheses.json"
            memory_path = root / "memory.json"
            output_path = root / "plan.json"
            cohort_path.write_text(json.dumps(_cohort()))
            hypotheses_path.write_text(json.dumps(_hypotheses()))
            context = build_planning_memory_context(
                memory_root=root / "empty-memory",
                cohort_path=cohort_path,
                hypotheses_path=hypotheses_path,
                output_path=memory_path,
            )

            report = run_modification_planning(
                cohort_path=cohort_path,
                hypotheses_path=hypotheses_path,
                output_path=output_path,
                model=model,
                memory_context_path=memory_path,
            )
            audit = json.loads((root / "plan.audit.json").read_text())

        user_payload = json.loads(model.calls[0]["messages"][1]["content"])
        self.assertEqual(report.status, "ok")
        self.assertEqual(user_payload["planning_memory"], context)
        self.assertEqual(audit["memory_context"]["path"], str(memory_path.resolve()))
        self.assertEqual(len(audit["memory_context"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
