import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import audit_candidate, run_candidate_modification


ALLOWED = "systems/deepread/DeepRead/agent/runner.py"


class FakeModel:
    model_name = "fake-modifier"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return self.responses.pop(0)


def _call(name: str, arguments: dict, call_id: str) -> dict:
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
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    candidate = root / "candidate"
    source = candidate / ALLOWED
    source.parent.mkdir(parents=True)
    source.write_text("def answer():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(candidate), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(candidate), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(candidate), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = {
        "schema_version": "deepread-modification-plan-v1",
        "cohort_id": "cohort-1",
        "plans": [
            {
                "plan_id": "plan-1",
                "decision": "proceed",
                "change_contract": {
                    "current_behavior": "returns one",
                    "required_behavior_delta": "return two",
                    "must_preserve": ["function signature"],
                    "non_goals": ["other files"],
                },
                "validation_plan": {},
                "risk": {"level": "low", "regression_scenarios": []},
                "edit_scope": {
                    "allowed_paths": [ALLOWED],
                    "max_files_to_modify": 1,
                    "forbidden_roots": ["tests/"],
                    "must_inspect_before_edit": True,
                },
            }
        ],
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manifest = {
        "schema_version": "deepread-candidate-manifest-v1",
        "candidate_id": "candidate-1",
        "plan_id": "plan-1",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "test_policy_id": "unit-v1",
        "test_policy_sha256": "f" * 64,
        "base_commit": head,
        "candidate_path": str(candidate),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return candidate, plan_path, manifest_path


class CandidateModifierTest(unittest.TestCase):
    def test_agent_applies_scoped_replacement_and_submits(self) -> None:
        responses = [
            _call("read_source", {"path": ALLOWED, "start_line": 1, "end_line": 20}, "r1"),
            _call(
                "replace_text",
                {"path": ALLOWED, "old_text": "    return 1", "new_text": "    return 2"},
                "e1",
            ),
            _call("show_diff", {}, "d1"),
            _call(
                "submit_modification",
                {
                    "summary": "Return the verified value.",
                    "implemented_behavior_delta": "Changed answer from one to two.",
                    "preservation_notes": ["Kept the function signature."],
                    "uncertainties": [],
                },
                "s1",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, plan, manifest = _fixture(root)
            output = root / "modification.json"
            report = run_candidate_modification(
                manifest_path=manifest,
                plan_path=plan,
                output_path=output,
                model=FakeModel(responses),
            )
            artifact = json.loads(output.read_text())
            audit = audit_candidate(
                manifest_path=manifest,
                plan_path=plan,
                modification_path=output,
            )
            source_text = (candidate / ALLOWED).read_text()
            output_sha = hashlib.sha256(output.read_bytes()).hexdigest()

        self.assertEqual(report.status, "modified")
        self.assertEqual(artifact["changed_paths"], [ALLOWED])
        self.assertNotIn("old_text", artifact["events"][3]["arguments"])
        self.assertIn("return 2", source_text)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["modification_sha256"], output_sha)

    def test_edit_before_inspection_is_rejected_without_mutation(self) -> None:
        responses = [
            _call(
                "replace_text",
                {"path": ALLOWED, "old_text": "return 1", "new_text": "return 2"},
                "e1",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, plan, manifest = _fixture(root)
            output = root / "modification.json"
            report = run_candidate_modification(
                manifest_path=manifest,
                plan_path=plan,
                output_path=output,
                model=FakeModel(responses),
                max_rounds=1,
            )
            artifact = json.loads(output.read_text())
            source_text = (candidate / ALLOWED).read_text()

        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(artifact["events"][-1]["ok"], False)
        self.assertIn("inspect source", artifact["events"][-1]["error"])
        self.assertIn("return 1", source_text)

    def test_static_audit_detects_tampered_modification_artifact(self) -> None:
        responses = [
            _call("read_source", {"path": ALLOWED, "start_line": 1, "end_line": 20}, "r1"),
            _call("replace_text", {"path": ALLOWED, "old_text": "return 1", "new_text": "return 2"}, "e1"),
            _call("show_diff", {}, "d1"),
            _call(
                "submit_modification",
                {"summary": "done", "implemented_behavior_delta": "two", "preservation_notes": [], "uncertainties": []},
                "s1",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan, manifest = _fixture(root)
            output = root / "modification.json"
            run_candidate_modification(
                manifest_path=manifest, plan_path=plan, output_path=output, model=FakeModel(responses)
            )
            artifact = json.loads(output.read_text())
            artifact["candidate_snapshot_sha256"] = "0" * 64
            output.write_text(json.dumps(artifact))
            audit = audit_candidate(
                manifest_path=manifest,
                plan_path=plan,
                modification_path=output,
            )

        self.assertFalse(audit["passed"])
        self.assertIn("modification_snapshot_mismatch", audit["violations"])


if __name__ == "__main__":
    unittest.main()
