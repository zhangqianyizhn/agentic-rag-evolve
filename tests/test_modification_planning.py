import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.planning import (
    ModificationPlanValidationError,
    PlanningSourceReader,
    build_planning_memory_context,
    run_modification_planning,
    validate_modification_plan,
)
from agentic_rag_evolve.evolution import candidate_snapshot_sha256


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


def _read_source_response(call_id: str = "read-1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read_source",
                                "arguments": json.dumps(
                                    {
                                        "path": "systems/deepread/DeepRead/agent/runner.py",
                                        "start_line": 1,
                                        "end_line": 80,
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }


class ModificationPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_root = Path(__file__).resolve().parents[1]

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

    def test_proceeding_plan_must_inspect_selected_source(self) -> None:
        with self.assertRaisesRegex(
            ModificationPlanValidationError, "were not inspected"
        ):
            validate_modification_plan(
                _plan(),
                cohort=_cohort(),
                hypotheses=_hypotheses(),
                observed_source_reads=[],
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
                _read_source_response(),
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
                source_root=self.source_root,
            )
            candidate = json.loads((root / "plan.candidate.json").read_text())
            audit = json.loads((root / "plan.audit.json").read_text())

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.model_calls, 3)
        self.assertEqual(report.tool_calls, 1)
        self.assertEqual(candidate, invalid)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(len(audit["events"]), 4)

    def test_agent_receives_audited_memory_context(self) -> None:
        model = FakeModel(
            [
                _read_source_response(),
                {"choices": [{"message": {"content": json.dumps(_plan())}}]},
            ]
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
                source_root=self.source_root,
            )
            audit = json.loads((root / "plan.audit.json").read_text())

        user_payload = json.loads(model.calls[0]["messages"][1]["content"])
        self.assertEqual(report.status, "ok")
        self.assertEqual(user_payload["planning_memory"], context)
        self.assertEqual(audit["memory_context"]["path"], str(memory_path.resolve()))
        self.assertEqual(len(audit["memory_context"]["sha256"]), 64)
        self.assertEqual(audit["source_access"]["source_count"], 1)
        self.assertEqual(
            audit["source_access"]["revision"]["kind"],
            "allowlisted_source_manifest",
        )
        self.assertEqual(
            audit["source_access"]["inspected_sources"][0]["path"],
            "systems/deepread/DeepRead/agent/runner.py",
        )
        self.assertEqual(
            len(audit["source_access"]["inspected_sources"][0]["sha256"]), 64
        )
        tool_names = [
            item["function"]["name"] for item in model.calls[0]["tools"]
        ]
        self.assertEqual(tool_names, ["list_sources", "read_source"])
        self.assertNotIn("content", json.dumps(audit))

    def test_source_reader_rejects_unreferenced_path_and_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "systems/deepread/DeepRead/agent/runner.py"
            source.parent.mkdir(parents=True)
            source.write_text("def run_agent():\n    return 1\n", encoding="utf-8")
            reader = PlanningSourceReader(
                source_root=root,
                cohort=_cohort(),
                hypotheses=_hypotheses(),
            )

            with self.assertRaisesRegex(PermissionError, "not allowlisted"):
                reader.read_source("systems/deepread/DeepRead/tool/search.py")
            source.write_text("def run_agent():\n    return 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after access"):
                reader.read_source(
                    "systems/deepread/DeepRead/agent/runner.py",
                    start_line=1,
                    end_line=2,
                )

    def test_candidate_audit_detects_changes_outside_source_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            root = container / "candidate"
            source = root / "systems/deepread/DeepRead/agent/runner.py"
            source.parent.mkdir(parents=True)
            source.write_text("def run_agent():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"], check=True
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "baseline"], check=True
            )
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source.write_text("def run_agent():\n    return 2\n", encoding="utf-8")
            changed = ["systems/deepread/DeepRead/agent/runner.py"]
            audit = {
                "schema_version": "deepread-candidate-audit-v1",
                "candidate_id": "candidate-1",
                "candidate_path": str(root),
                "head_commit": head,
                "changed_paths": changed,
                "candidate_snapshot_sha256": candidate_snapshot_sha256(
                    root, head_commit=head, changed_paths=changed
                ),
                "passed": True,
            }
            audit_path = container / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            reader = PlanningSourceReader(
                source_root=root,
                cohort=_cohort(),
                hypotheses=_hypotheses(),
                candidate_audit_path=audit_path,
            )
            self.assertEqual(reader.revision["kind"], "candidate_snapshot")
            self.assertEqual(
                reader.revision["snapshot_sha256"],
                audit["candidate_snapshot_sha256"],
            )
            unscoped = root / "systems/deepread/DeepRead/tool/unscoped.py"
            unscoped.parent.mkdir(parents=True)
            unscoped.write_text("VALUE = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "changed paths"):
                reader.catalog()


if __name__ == "__main__":
    unittest.main()
