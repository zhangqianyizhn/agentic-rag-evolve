import copy
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.planning import (
    HypothesisValidationError,
    run_hypothesis_aggregation,
    validate_hypotheses,
)


class FakeModel:
    model_name = "fake-hypothesis-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return self.responses.pop(0)


def _diagnosis(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "source": f"{task_id}.json",
        "sha256": task_id * 8,
        "failure_manifestation": "The answer ignored a read value.",
        "earliest_intervention": {"turn": 2, "tool_call_id": "read-1"},
        "root_cause_hypothesis": "The final answer is not checked against read evidence.",
        "supporting_evidence": [
            {"kind": "trajectory", "claim": "The answer changed the value.", "turn": 3}
        ],
        "contradicting_evidence": [
            {"kind": "coverage", "claim": "The value was retrieved.", "evidence_index": 0, "layer": "read"}
        ],
        "counterfactual": {"change": "Verify the value."},
        "affected_sources": [
            {"path": "systems/deepread/DeepRead/agent/runner.py", "symbol": "run_agent"}
        ],
        "uncertainties": ["Only one dataset has been checked."],
    }


def _cohort(*task_ids: str) -> dict:
    return {
        "schema_version": "deepread-hypothesis-cohort-v1",
        "cohort_id": "cohort-1",
        "counts": {"input": len(task_ids), "eligible": len(task_ids), "excluded": 0},
        "eligible_diagnoses": [_diagnosis(task_id) for task_id in task_ids],
        "excluded_diagnoses": [],
    }


def _hypothesis(*task_ids: str) -> dict:
    return {
        "schema_version": "deepread-improvement-hypotheses-v1",
        "cohort_id": "cohort-1",
        "status": "ready",
        "hypotheses": [
            {
                "title": "Verify final numeric claims",
                "task_ids": list(task_ids),
                "common_mechanism": "The answer path does not verify selected values against read evidence.",
                "earliest_intervention_pattern": "After the final evidence read and before answer emission.",
                "target_cohort": {
                    "inclusion_signals": ["Read evidence contains the gold value."],
                    "exclusion_signals": ["Gold evidence is absent from the corpus."],
                },
                "behavior_delta": "Check final numeric claims against the evidence already read.",
                "supporting_evidence_refs": [
                    {
                        "task_id": task_id,
                        "side": "supporting",
                        "index": 0,
                        "rationale": "The answer diverges after the value was available.",
                    }
                    for task_id in task_ids
                ],
                "contradicting_evidence_refs": [
                    {
                        "task_id": task_ids[0],
                        "side": "contradicting",
                        "index": 0,
                        "rationale": "Retrieval itself succeeded.",
                    }
                ],
                "affected_source_refs": [
                    {
                        "task_id": task_ids[0],
                        "index": 0,
                        "rationale": "This source owns final answer emission.",
                    }
                ],
                "validation": {
                    "expected_observation": "Target cases preserve the read value in the answer.",
                    "falsifier": "The same divergence remains after explicit verification.",
                    "regression_guards": ["Answers without numeric claims remain unchanged."],
                },
                "uncertainties": ["The mechanism may not recur outside this cohort."],
            }
        ],
        "unclustered_task_ids": [],
    }


class HypothesisAggregationTest(unittest.TestCase):
    def test_validator_builds_stable_grounded_hypothesis(self) -> None:
        cohort = _cohort("q1", "q2")

        result = validate_hypotheses(_hypothesis("q1", "q2"), cohort=cohort)

        item = result["hypotheses"][0]
        self.assertEqual(item["maturity"], "recurring")
        self.assertTrue(item["hypothesis_id"].startswith("hyp_"))
        self.assertEqual(item["task_ids"], ["q1", "q2"])

    def test_validator_rejects_unknown_evidence_reference(self) -> None:
        cohort = _cohort("q1")
        candidate = _hypothesis("q1")
        candidate["hypotheses"][0]["supporting_evidence_refs"][0]["index"] = 9

        with self.assertRaisesRegex(HypothesisValidationError, "unknown evidence index"):
            validate_hypotheses(candidate, cohort=cohort)

    def test_validator_rejects_evidence_on_wrong_side(self) -> None:
        cohort = _cohort("q1")
        candidate = _hypothesis("q1")
        candidate["hypotheses"][0]["supporting_evidence_refs"][0]["side"] = (
            "contradicting"
        )

        with self.assertRaisesRegex(HypothesisValidationError, "side must be"):
            validate_hypotheses(candidate, cohort=cohort)

    def test_every_eligible_task_must_be_clustered_or_unclustered(self) -> None:
        cohort = _cohort("q1", "q2")
        candidate = _hypothesis("q1")

        with self.assertRaisesRegex(HypothesisValidationError, "not accounted for"):
            validate_hypotheses(candidate, cohort=cohort)

    def test_empty_cohort_skips_model(self) -> None:
        cohort = _cohort()
        model = FakeModel([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            output_path = root / "hypotheses.json"
            cohort_path.write_text(json.dumps(cohort))

            report = run_hypothesis_aggregation(
                cohort_path=cohort_path,
                output_path=output_path,
                model=model,
            )
            result = json.loads(output_path.read_text())
            audit = json.loads((root / "hypotheses.audit.json").read_text())

        self.assertEqual(report.status, "no_eligible_diagnoses")
        self.assertEqual(report.model_calls, 0)
        self.assertEqual(result["status"], "no_eligible_diagnoses")
        self.assertEqual(audit["status"], "no_eligible_diagnoses")
        self.assertEqual(model.calls, [])

    def test_agent_retries_one_invalid_candidate(self) -> None:
        cohort = _cohort("q1")
        invalid = _hypothesis("q1")
        invalid["hypotheses"][0]["task_ids"] = ["missing"]
        corrected = _hypothesis("q1")
        model = FakeModel(
            [
                {"choices": [{"message": {"content": json.dumps(invalid)}}]},
                {"choices": [{"message": {"content": json.dumps(corrected)}}]},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            output_path = root / "hypotheses.json"
            cohort_path.write_text(json.dumps(cohort))

            report = run_hypothesis_aggregation(
                cohort_path=cohort_path,
                output_path=output_path,
                model=model,
            )
            result = json.loads(output_path.read_text())
            candidate = json.loads((root / "hypotheses.candidate.json").read_text())
            audit = json.loads((root / "hypotheses.audit.json").read_text())

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.model_calls, 2)
        self.assertEqual(report.validation_failures, 1)
        self.assertEqual(result["hypotheses"][0]["maturity"], "singleton")
        self.assertEqual(candidate, invalid)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(len(audit["events"]), 2)
        self.assertEqual(
            audit["token_usage"],
            {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        )


if __name__ == "__main__":
    unittest.main()
