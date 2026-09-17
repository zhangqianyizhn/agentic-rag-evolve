import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import (
    advance_baseline_registry,
    current_baseline,
    initialize_baseline_registry,
)


def _canonical(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class BaselineRegistryTest(unittest.TestCase):
    def _repo(self, root: Path):
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
        source = repo / "source.py"
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_DATE": "1000000000 +0000",
                "GIT_COMMITTER_DATE": "1000000000 +0000",
            }
        )
        source.write_text("value = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
        subprocess.run(
            [
                "git", "-C", str(repo), "-c", "user.name=Test",
                "-c", "user.email=test@local", "commit", "--no-gpg-sign",
                "-m", "base",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        base = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        environment["GIT_AUTHOR_DATE"] = "1000000001 +0000"
        environment["GIT_COMMITTER_DATE"] = "1000000001 +0000"
        source.write_text("value = 2\n")
        subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
        subprocess.run(
            [
                "git", "-C", str(repo), "-c", "user.name=Test",
                "-c", "user.email=test@local", "commit", "--no-gpg-sign",
                "-m", "candidate",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        child = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "-C", str(repo), "show", "-s", "--format=%T", child],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return repo, base, child, tree

    def _materialization(
        self, root: Path, repo: Path, base: str, child: str, tree: str
    ) -> Path:
        outcome = root / "outcome.json"
        outcome.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-candidate-outcome-v1",
                    "outcome": "accepted",
                    "decision_id": "outcome-1",
                }
            )
        )
        outcome_sha = hashlib.sha256(outcome.read_bytes()).hexdigest()
        basis = {
            "decision_id": "outcome-1",
            "outcome_record_sha256": outcome_sha,
            "materialized_commit": child,
        }
        materialization_id = "materialization-" + _canonical(basis)[:20]
        path = root / "materialization.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "deepread-candidate-materialization-v1",
                    "materialization_id": materialization_id,
                    "decision_id": "outcome-1",
                    "outcome_record": str(outcome),
                    "outcome_record_sha256": outcome_sha,
                    "base_commit": base,
                    "materialized_commit": child,
                    "tree": tree,
                    "candidate_path": str(repo),
                    "status": "materialized_detached",
                    "branch_created": False,
                    "baseline_updated": False,
                }
            )
        )
        return path

    def test_initializes_and_advances_linear_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, base, child, tree = self._repo(root)
            registry = root / "registry"
            genesis, _ = initialize_baseline_registry(
                registry_root=registry, repo_root=repo, revision=base
            )
            materialization = self._materialization(root, repo, base, child, tree)
            advanced, path = advance_baseline_registry(
                registry_root=registry, materialization_path=materialization
            )
            current = current_baseline(registry)
            durable = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", advanced["durable_ref"]],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertEqual(genesis["generation"], 0)
        self.assertEqual(advanced["parent_baseline_id"], genesis["baseline_id"])
        self.assertEqual(advanced["commit"], child)
        self.assertEqual(current, advanced)
        self.assertEqual(durable, child)
        self.assertEqual(path.name[:5], "0001-")

    def test_registry_cannot_be_initialized_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, base, _, _ = self._repo(root)
            registry = root / "registry"
            initialize_baseline_registry(
                registry_root=registry, repo_root=repo, revision=base
            )

            with self.assertRaisesRegex(FileExistsError, "already initialized"):
                initialize_baseline_registry(
                    registry_root=registry, repo_root=repo, revision=base
                )

    def test_same_materialization_cannot_advance_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, base, child, tree = self._repo(root)
            registry = root / "registry"
            initialize_baseline_registry(
                registry_root=registry, repo_root=repo, revision=base
            )
            materialization = self._materialization(root, repo, base, child, tree)
            advance_baseline_registry(
                registry_root=registry, materialization_path=materialization
            )

            with self.assertRaisesRegex(ValueError, "current baseline"):
                advance_baseline_registry(
                    registry_root=registry, materialization_path=materialization
                )

    def test_rejects_materialization_from_another_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, base, child, tree = self._repo(root)
            registry = root / "registry"
            initialize_baseline_registry(
                registry_root=registry, repo_root=repo, revision=base
            )
            materialization = self._materialization(root, repo, "f" * 40, child, tree)

            with self.assertRaisesRegex(ValueError, "current baseline"):
                advance_baseline_registry(
                    registry_root=registry, materialization_path=materialization
                )

    def test_detects_modified_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, base, _, _ = self._repo(root)
            registry = root / "registry"
            _, path = initialize_baseline_registry(
                registry_root=registry, repo_root=repo, revision=base
            )
            entry = json.loads(path.read_text())
            entry["commit"] = "f" * 40
            path.write_text(json.dumps(entry))

            with self.assertRaisesRegex(ValueError, "identity is invalid"):
                current_baseline(registry)


if __name__ == "__main__":
    unittest.main()
