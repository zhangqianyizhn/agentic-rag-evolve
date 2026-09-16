import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import audit_candidate, create_candidate_worktree


BASE_COMMIT = "a" * 40
ALLOWED_PATH = "systems/deepread/DeepRead/agent/runner.py"


def _plan(decision: str = "proceed") -> dict:
    return {
        "schema_version": "deepread-modification-plan-v1",
        "cohort_id": "cohort-1",
        "hypothesis_set_status": "ready",
        "plans": [
            {
                "plan_id": "plan-1",
                "decision": decision,
                "edit_scope": {
                    "allowed_paths": [ALLOWED_PATH],
                    "max_files_to_modify": 1,
                    "forbidden_roots": ["runner/", "src/", "tests/"],
                },
            }
        ],
    }


def _write_plan(root: Path, decision: str = "proceed") -> Path:
    path = root / "plan.json"
    path.write_text(json.dumps(_plan(decision)), encoding="utf-8")
    return path


def _write_policy(root: Path) -> Path:
    path = root / "test-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepread-candidate-test-policy-v1",
                "policy_id": "unit-v1",
                "checks": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class CandidateIsolationTest(unittest.TestCase):
    def test_create_candidate_uses_detached_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            plan = _write_plan(root)
            policy = _write_policy(root)
            candidate = root / "candidate"
            manifest_path = root / "artifacts" / "manifest.json"
            commands = []

            def runner(command):
                commands.append(list(command))
                stdout = BASE_COMMIT + "\n" if "rev-parse" in command else ""
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            manifest = create_candidate_worktree(
                repo_root=repo,
                plan_path=plan,
                test_policy_path=policy,
                plan_id="plan-1",
                base_revision="main",
                candidate_path=candidate,
                manifest_path=manifest_path,
                command_runner=runner,
            )

        self.assertEqual(manifest["base_commit"], BASE_COMMIT)
        self.assertEqual(manifest["isolation"], "detached_git_worktree")
        self.assertEqual(manifest["test_policy_id"], "unit-v1")
        self.assertEqual(commands[1][-2:], [str(candidate.resolve()), BASE_COMMIT])
        self.assertIn("--detach", commands[1])

    def test_deferred_plan_cannot_create_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            plan = _write_plan(root, "defer")
            policy = _write_policy(root)

            with self.assertRaisesRegex(ValueError, "not approved to proceed"):
                create_candidate_worktree(
                    repo_root=repo,
                    plan_path=plan,
                    test_policy_path=policy,
                    plan_id="plan-1",
                    base_revision="main",
                    candidate_path=root / "candidate",
                    manifest_path=root / "manifest.json",
                )

    def _audit_fixture(self, root: Path):
        candidate = root / "candidate"
        (candidate / Path(ALLOWED_PATH).parent).mkdir(parents=True)
        (candidate / ALLOWED_PATH).write_text("value = 1\n", encoding="utf-8")
        plan_path = _write_plan(root)
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        policy_path = _write_policy(root)
        manifest = {
            "schema_version": "deepread-candidate-manifest-v1",
            "candidate_id": "candidate-1",
            "plan_id": "plan-1",
            "plan_sha256": plan_sha,
            "test_policy_id": "unit-v1",
            "test_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "base_commit": BASE_COMMIT,
            "candidate_path": str(candidate),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return candidate, plan_path, manifest_path

    def test_audit_accepts_in_scope_syntax_valid_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan_path, manifest_path = self._audit_fixture(root)

            def runner(command):
                if "rev-parse" in command:
                    stdout = BASE_COMMIT + "\n"
                elif "--name-only" in command:
                    stdout = ALLOWED_PATH + "\0"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            result = audit_candidate(
                manifest_path=manifest_path,
                plan_path=plan_path,
                command_runner=runner,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["changed_paths"], [ALLOWED_PATH])
        self.assertEqual(result["candidate_path"], str((root / "candidate").resolve()))
        self.assertEqual(len(result["candidate_snapshot_sha256"]), 64)
        self.assertEqual(result["violations"], [])

    def test_audit_reports_scope_budget_and_syntax_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, plan_path, manifest_path = self._audit_fixture(root)
            outside = "runner/bad.py"
            (candidate / "runner").mkdir()
            (candidate / outside).write_text("not valid python !!!", encoding="utf-8")

            def runner(command):
                if "rev-parse" in command:
                    stdout = BASE_COMMIT + "\n"
                elif "--name-only" in command:
                    stdout = ALLOWED_PATH + "\0" + outside + "\0"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            result = audit_candidate(
                manifest_path=manifest_path,
                plan_path=plan_path,
                command_runner=runner,
            )

        self.assertFalse(result["passed"])
        self.assertIn("changed_files_exceed_budget", result["violations"])
        self.assertIn("changed_files_outside_allowed_paths", result["violations"])
        self.assertIn("forbidden_roots_touched", result["violations"])
        self.assertIn("python_syntax_errors", result["violations"])

    def test_audit_rejects_broken_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, plan_path, manifest_path = self._audit_fixture(root)
            target = candidate / ALLOWED_PATH
            target.unlink()
            target.symlink_to(candidate / "missing-target.py")

            def runner(command):
                if "rev-parse" in command:
                    stdout = BASE_COMMIT + "\n"
                elif "--name-only" in command:
                    stdout = ALLOWED_PATH + "\0"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            result = audit_candidate(
                manifest_path=manifest_path,
                plan_path=plan_path,
                command_runner=runner,
            )

        self.assertIn("changed_symlinks", result["violations"])


if __name__ == "__main__":
    unittest.main()
