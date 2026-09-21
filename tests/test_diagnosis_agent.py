import copy
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
        self.calls.append(copy.deepcopy(payload))
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
                    "quote": "def run_agent(",
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
        self.assertEqual(
            audit["token_usage"],
            {"input_tokens": 220, "output_tokens": 90, "reasoning_tokens": 0},
        )
        self.assertEqual(
            [(event["kind"], event["round"]) for event in audit["events"]],
            [("model", 1), ("tool", 1), ("model", 2)],
        )
        self.assertTrue(
            all(
                event["status"] == "ok"
                for event in audit["events"]
                if event["kind"] == "model"
            )
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

    def test_validator_rejects_quote_outside_cited_source_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = json.loads(self._bundle(root).read_text())
        diagnosis = self._diagnosis()
        diagnosis["supporting_evidence"][1]["end_line"] = 5
        source_path = self.source_root / "systems/deepread/DeepRead/agent/runner.py"
        observed = {
            "path": "systems/deepread/DeepRead/agent/runner.py",
            "start_line": 1,
            "end_line": 30,
            "content": "\n".join(source_path.read_text().splitlines()[:30]),
        }

        with self.assertRaisesRegex(DiagnosisValidationError, "not present"):
            validate_diagnosis(
                diagnosis,
                bundle=bundle,
                observed_source_reads=[observed],
                observed_payload_reads=[],
            )

    def test_validation_retry_explains_flat_anchor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = self._bundle(root)
            corrected = self._diagnosis()
            corrected["status"] = "not_agent_failure"
            corrected["earliest_intervention"] = None
            corrected["supporting_evidence"] = [
                {
                    "kind": "evaluation",
                    "claim": "The judge marked the answer incorrect.",
                    "field": "judge.score",
                }
            ]
            corrected["contradicting_evidence"] = []
            corrected["affected_sources"] = []
            invalid = dict(corrected)
            invalid["supporting_evidence"] = [
                {"kind": "unsupported", "claim": "The judge marked the answer incorrect."}
            ]
            model = FakeDiagnosisModel(
                [
                    {
                        "choices": [{"message": {"content": json.dumps(invalid)}}],
                        "usage": {},
                    },
                    {
                        "choices": [{"message": {"content": json.dumps(corrected)}}],
                        "usage": {},
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
            audit = json.loads((output / "audit.json").read_text())
            retained_candidate = json.loads((output / "candidate.json").read_text())

        self.assertEqual(report.status, "ok")
        correction = model.calls[1]["messages"][-1]["content"]
        self.assertIn('Judge facts use kind="evaluation"', correction)
        self.assertEqual(
            audit["events"][0]["candidate_shape"]["supporting_evidence"],
            [{"kind": "unsupported", "fields": ["claim", "kind"]}],
        )
        self.assertEqual(retained_candidate, invalid)
        self.assertNotIn("content", json.dumps(audit))

    def test_validator_normalizes_nested_and_judge_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = json.loads(self._bundle(root).read_text())
        diagnosis = self._diagnosis()
        diagnosis["status"] = "not_agent_failure"
        diagnosis["earliest_intervention"] = None
        diagnosis["affected_sources"] = []
        diagnosis["supporting_evidence"] = [
            {
                "kind": "evaluation",
                "claim": "The judge score is low.",
                "evaluation": {"field": "judge.score"},
            }
        ]
        diagnosis["contradicting_evidence"] = [
            {"kind": "judge", "claim": "The judge explains the mismatch."}
        ]

        normalized = validate_diagnosis(diagnosis, bundle=bundle)

        self.assertEqual(
            normalized["supporting_evidence"][0],
            {
                "kind": "evaluation",
                "claim": "The judge score is low.",
                "field": "judge.score",
            },
        )
        self.assertEqual(
            normalized["contradicting_evidence"][0]["field"], "judge.reasoning"
        )

    def test_validator_normalizes_single_uncertainty_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = json.loads(self._bundle(Path(directory)).read_text())
        diagnosis = self._diagnosis()
        diagnosis["uncertainties"] = "The annotation convention remains ambiguous."

        normalized = validate_diagnosis(diagnosis, bundle=bundle)

        self.assertEqual(
            normalized["uncertainties"],
            ["The annotation convention remains ambiguous."],
        )

    def test_validator_allows_detailed_failure_manifestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = json.loads(self._bundle(Path(directory)).read_text())
        diagnosis = self._diagnosis()
        diagnosis["failure_manifestation"] = "x" * 1_000

        normalized = validate_diagnosis(diagnosis, bundle=bundle)

        self.assertEqual(len(normalized["failure_manifestation"]), 1_000)

    def test_source_read_is_safely_clamped_without_a_retry_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = self._bundle(root)
            diagnosis = self._diagnosis()
            diagnosis["supporting_evidence"][1]["path"] = (
                "systems/deepread/DeepRead/index/markdown_parser.py"
            )
            diagnosis["supporting_evidence"][1]["quote"] = (
                "def _read_file(path: str) -> str:"
            )
            diagnosis["supporting_evidence"][1]["end_line"] = 240
            diagnosis["affected_sources"][0]["path"] = (
                "systems/deepread/DeepRead/index/markdown_parser.py"
            )
            diagnosis["affected_sources"][0]["end_line"] = 240
            model = FakeDiagnosisModel(
                [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "read-wide",
                                            "type": "function",
                                            "function": {
                                                "name": "read_source",
                                                "arguments": json.dumps(
                                                    {
                                                        "path": "systems/deepread/DeepRead/index/markdown_parser.py",
                                                        "start_line": 1,
                                                        "end_line": 241,
                                                    }
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {},
                    },
                    {
                        "choices": [
                            {"message": {"content": json.dumps(diagnosis)}}
                        ],
                        "usage": {},
                    },
                ]
            )

            report = run_diagnosis(
                bundle_path=bundle_path,
                source_root=self.source_root,
                output_path=root / "output",
                model=model,
            )

        tool_result = json.loads(model.calls[1]["messages"][-1]["content"])
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.rounds, 2)
        self.assertEqual(tool_result["end_line"], 240)
        self.assertEqual(tool_result["requested_end_line"], 241)
        self.assertTrue(tool_result["has_more"])

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

    def test_tool_round_budget_keeps_one_tool_free_finalization_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = self._bundle(root)
            insufficient = self._diagnosis()
            insufficient.update(
                status="insufficient_evidence",
                earliest_intervention=None,
                root_cause_hypothesis="The available evidence does not isolate a DeepRead defect.",
                supporting_evidence=[],
                contradicting_evidence=[],
                counterfactual={
                    "change": "Collect another comparable failure.",
                    "expected_observation": "The same mechanism recurs.",
                    "falsifier": "The error does not recur.",
                },
                affected_sources=[],
                uncertainties=["The single trace is insufficient."],
            )
            model = FakeDiagnosisModel(
                [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "list-1",
                                            "type": "function",
                                            "function": {
                                                "name": "list_sources",
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {},
                    },
                    {
                        "choices": [{"message": {"content": json.dumps(insufficient)}}],
                        "usage": {},
                    },
                ]
            )
            report = run_diagnosis(
                bundle_path=bundle_path,
                source_root=self.source_root,
                output_path=root / "output",
                model=model,
                max_rounds=1,
            )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.rounds, 2)
        self.assertNotIn("tools", model.calls[1])
        self.assertIn("budget is exhausted", model.calls[1]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
