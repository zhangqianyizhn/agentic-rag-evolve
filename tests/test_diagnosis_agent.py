import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.diagnosis import (
    DiagnosisValidationError,
    run_diagnosis,
    validate_diagnosis,
)
from agentic_rag_evolve.diagnostics.policy import DIAGNOSTIC_SOURCE_POLICY


class FakeDiagnosisModel:
    model_name = "fake-diagnosis-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, payload):
        self.calls.append(payload)
        return self.responses.pop(0)


class DiagnosisAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_root = Path(__file__).resolve().parents[1]

    def _bundle(self, root: Path, *, triage: str = "incorrect_answer") -> Path:
        sources = []
        for spec in DIAGNOSTIC_SOURCE_POLICY:
            data = (self.source_root / spec.path).read_bytes()
            sources.append(
                {
                    "path": spec.path,
                    "component": spec.component,
                    "purpose": spec.purpose,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "line_count": len(data.decode("utf-8").splitlines()),
                }
            )
        eligible = triage in {"incorrect_answer", "partial_answer", "no_answer"}
        bundle = {
            "schema_version": "deepread-diagnostic-input-v1",
            "task": {
                "task_id": "q1",
                "sample_id": "report",
                "question": "What was revenue?",
            },
            "run_context": {},
            "evaluation": {
                "gold_answers": ["12 million"],
                "gold_evidence": ["Revenue was 12 million."],
                "prediction": {"status": "ok", "answer": "10 million"},
                "metrics": {"f1": 0.5, "recall": 1.0},
                "judge": {"status": "ok", "score": 1, "reasoning": "wrong value"},
            },
            "evidence_coverage": {
                "primary_signal": "answer",
                "layers": {
                    "corpus": {"recall": 1.0},
                    "candidate": {"recall": 1.0},
                    "read": {"recall": 1.0},
                    "answer": {"recall": 0.0},
                },
                "evidence": [
                    {
                        "evidence_index": 0,
                        "earliest_missing_layer": "answer",
                        "layers": {
                            "corpus": {"matched": True, "references": []},
                            "candidate": {"matched": True, "references": []},
                            "read": {"matched": True, "references": []},
                            "answer": {"matched": False, "references": []},
                        },
                    }
                ],
            },
            "failure_signals": {
                "triage": triage,
                "diagnosis_route": {
                    "eligible": eligible,
                    "target": "deepread" if eligible else "evaluation_review",
                    "reason": "test route",
                },
            },
            "trajectory": {
                "task_id": "q1",
                "question": "What was revenue?",
                "status": "ok",
                "answer": "10 million",
                "turns": [
                    {
                        "round": 1,
                        "model": {"reasoning": "Use the first number."},
                        "tools": [
                            {
                                "call_id": "search-1",
                                "name": "bm25_search",
                                "arguments": {"query": "revenue"},
                                "ok": True,
                                "result": {"results": []},
                            }
                        ],
                    }
                ],
            },
            "access": {"sources": sources, "payloads": [], "tools": []},
        }
        path = root / "bundle.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def _diagnosis(self):
        return {
            "schema_version": "deepread-diagnosis-v1",
            "task_id": "q1",
            "status": "diagnosed",
            "failure_manifestation": "The answer reports 10 million instead of 12 million.",
            "earliest_intervention": {
                "turn": 1,
                "tool_call_id": "search-1",
                "rationale": "The first retrieval decision did not verify the selected value.",
            },
            "root_cause_hypothesis": (
                "The loop accepts a final value without requiring evidence verification."
            ),
            "supporting_evidence": [
                {
                    "kind": "trajectory",
                    "claim": "Turn 1 selected the unsupported value.",
                    "turn": 1,
                    "tool_call_id": "search-1",
                },
                {
                    "kind": "source",
                    "claim": "The agent loop controls final-answer acceptance.",
                    "path": "systems/deepread/DeepRead/agent/runner.py",
                    "start_line": 1,
                    "end_line": 30,
                },
            ],
            "contradicting_evidence": [
                {
                    "kind": "coverage",
                    "claim": "The gold evidence was available before answer generation.",
                    "evidence_index": 0,
                    "layer": "read",
                }
            ],
            "counterfactual": {
                "change": "Require the chosen numeric value to be checked against read evidence.",
                "expected_observation": "The answer uses 12 million.",
                "falsifier": "The same wrong value remains after explicit evidence verification.",
            },
            "affected_sources": [
                {
                    "path": "systems/deepread/DeepRead/agent/runner.py",
                    "start_line": 1,
                    "end_line": 30,
                    "symbol": "run_agent",
                    "rationale": "This function owns answer acceptance and loop termination.",
                }
            ],
            "uncertainties": ["A single synthetic case does not establish prevalence."],
        }

    def test_agent_reads_source_then_returns_validated_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = self._bundle(root)
            model = FakeDiagnosisModel(
                [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "read-1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_source",
                                                "arguments": json.dumps(
                                                    {
                                                        "path": "systems/deepread/DeepRead/agent/runner.py",
                                                        "start_line": 1,
                                                        "end_line": 30,
                                                    }
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    },
                    {
                        "choices": [
                            {"message": {"content": json.dumps(self._diagnosis())}}
                        ],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 80},
                    },
                ]
            )
            output = root / "output"
            report = run_diagnosis(
                bundle_path=bundle_path,
                source_root=self.source_root,
                output_path=output,
                model=model,
            )
            diagnosis = json.loads((output / "diagnosis.json").read_text())
            audit = json.loads((output / "audit.json").read_text())

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.rounds, 2)
        self.assertEqual(report.tool_calls, 1)
        self.assertEqual(diagnosis["status"], "diagnosed")
        self.assertEqual(audit["token_usage"], {"input_tokens": 220, "output_tokens": 90})
        self.assertEqual(
            [(event["kind"], event["round"]) for event in audit["events"]],
            [("model", 1), ("tool", 1), ("model", 2)],
        )
        self.assertNotIn("content", json.dumps(audit))

    def test_validator_rejects_source_lines_the_agent_did_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = json.loads(self._bundle(root).read_text())

        with self.assertRaisesRegex(DiagnosisValidationError, "not read"):
            validate_diagnosis(
                self._diagnosis(),
                bundle=bundle,
                observed_source_reads=[],
                observed_payload_reads=[],
            )

    def test_non_deepread_route_is_skipped_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = self._bundle(root, triage="evaluation_suspicious")
            model = FakeDiagnosisModel([])
            output = root / "output"
            report = run_diagnosis(
                bundle_path=bundle_path,
                source_root=self.source_root,
                output_path=output,
                model=model,
            )
            audit = json.loads((output / "audit.json").read_text())

        self.assertEqual(report.status, "skipped")
        self.assertEqual(model.calls, [])
        self.assertEqual(audit["route"]["target"], "evaluation_review")


if __name__ == "__main__":
    unittest.main()
