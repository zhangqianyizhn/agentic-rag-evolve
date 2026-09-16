import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import (
    audit_candidate_tests,
    candidate_snapshot_sha256,
)


HEAD = "a" * 40
CHANGED = "systems/deepread/DeepRead/agent/runner.py"


class CandidateTestAuditTest(unittest.TestCase):
    def _fixture(self, root: Path):
        candidate = root / "candidate"
        source = candidate / CHANGED
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="utf-8")
        (candidate / "tests").mkdir()
        snapshot = candidate_snapshot_sha256(
            candidate, head_commit=HEAD, changed_paths=[CHANGED]
        )
        audit = {
            "schema_version": "deepread-candidate-audit-v1",
            "candidate_id": "candidate-1",
            "plan_id": "plan-1",
            "plan_sha256": "b" * 64,
            "head_commit": HEAD,
            "candidate_path": str(candidate),
            "candidate_snapshot_sha256": snapshot,
            "changed_paths": [CHANGED],
            "passed": True,
        }
        audit_path = root / "candidate-audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        policy = {
            "schema_version": "deepread-candidate-test-policy-v1",
            "policy_id": "unit-v1",
            "checks": [
                {
                    "check_id": "unit",
                    "kind": "unittest_discover",
                    "start_directory": "tests",
                    "pattern": "test_*.py",
                    "timeout_seconds": 30,
                    "minimum_tests": 10,
                }
            ],
        }
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        audit["test_policy_id"] = "unit-v1"
        audit["test_policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return candidate, source, audit_path, policy_path

    def _git_runner(self, command):
        if "rev-parse" in command:
            output = HEAD + "\n"
        elif "--name-only" in command:
            output = CHANGED + "\0"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def test_passes_fixed_unittest_policy_with_sanitized_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, _, audit_path, policy_path = self._fixture(root)

            def process(command, **kwargs):
                self.assertEqual(kwargs["cwd"], candidate.resolve())
                self.assertNotIn("DOUBAO_API_KEY", kwargs["env"])
                self.assertEqual(command[1:4], ["-m", "unittest", "discover"])
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="Ran 12 tests in 0.01s\n\nOK\n"
                )

            result = audit_candidate_tests(
                candidate_audit_path=audit_path,
                policy_path=policy_path,
                python_executable=Path("/usr/bin/python3"),
                command_runner=self._git_runner,
                process_runner=process,
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["candidate_state_preserved"])
        self.assertEqual(result["checks"][0]["tests_run"], 12)

    def test_rejects_snapshot_changed_before_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source, audit_path, policy_path = self._fixture(root)
            source.write_text("value = 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                audit_candidate_tests(
                    candidate_audit_path=audit_path,
                    policy_path=policy_path,
                    python_executable=Path("/usr/bin/python3"),
                    command_runner=self._git_runner,
                )

    def test_fails_when_test_count_shrinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, audit_path, policy_path = self._fixture(root)

            def process(command, **kwargs):
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="Ran 9 tests in 0.01s\n\nOK\n"
                )

            result = audit_candidate_tests(
                candidate_audit_path=audit_path,
                policy_path=policy_path,
                python_executable=Path("/usr/bin/python3"),
                command_runner=self._git_runner,
                process_runner=process,
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["failure_reasons"], ["too_few_tests"])

    def test_detects_candidate_mutation_during_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source, audit_path, policy_path = self._fixture(root)

            def process(command, **kwargs):
                source.write_text("value = 3\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="Ran 12 tests in 0.01s\n\nOK\n"
                )

            result = audit_candidate_tests(
                candidate_audit_path=audit_path,
                policy_path=policy_path,
                python_executable=Path("/usr/bin/python3"),
                command_runner=self._git_runner,
                process_runner=process,
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["candidate_state_preserved"])

    def test_policy_cannot_supply_an_arbitrary_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, audit_path, policy_path = self._fixture(root)
            policy = json.loads(policy_path.read_text())
            policy["checks"][0]["kind"] = "shell"
            policy_path.write_text(json.dumps(policy))
            audit = json.loads(audit_path.read_text())
            audit["test_policy_sha256"] = hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest()
            audit_path.write_text(json.dumps(audit))

            with self.assertRaisesRegex(ValueError, "unsupported candidate test kind"):
                audit_candidate_tests(
                    candidate_audit_path=audit_path,
                    policy_path=policy_path,
                    python_executable=Path("/usr/bin/python3"),
                    command_runner=self._git_runner,
                )


if __name__ == "__main__":
    unittest.main()
