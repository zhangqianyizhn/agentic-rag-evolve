import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.evolution import (
    candidate_snapshot_sha256,
    materialize_accepted_candidate,
)


CHANGED = "systems/deepread/DeepRead/agent/runner.py"


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class CandidateMaterializationTest(unittest.TestCase):
    def _fixture(self, root: Path):
        candidate = root / "candidate"
        candidate.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(candidate)], check=True, capture_output=True)
        source = candidate / CHANGED
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(candidate), "add", CHANGED], check=True)
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_DATE": "1000000000 +0000",
                "GIT_COMMITTER_DATE": "1000000000 +0000",
            }
        )
        subprocess.run(
            [
                "git", "-C", str(candidate), "-c", "user.name=Base",
                "-c", "user.email=base@local", "commit", "--no-gpg-sign",
                "-m", "base",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        base = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source.write_text("value = 2\n", encoding="utf-8")
        snapshot = candidate_snapshot_sha256(
            candidate, head_commit=base, changed_paths=[CHANGED]
        )

        artifacts = root / "artifacts"
        artifacts.mkdir()
        manifest = _write(
            artifacts / "manifest.json",
            {
                "schema_version": "deepread-candidate-manifest-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "base_commit": base,
                "candidate_path": str(candidate.resolve()),
            },
        )
        plan = _write(
            artifacts / "plan.json",
            {"schema_version": "deepread-modification-plan-v1"},
        )
        audit = _write(
            artifacts / "audit.json",
            {
                "schema_version": "deepread-candidate-audit-v1",
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "base_commit": base,
                "head_commit": base,
                "candidate_path": str(candidate.resolve()),
                "candidate_snapshot_sha256": snapshot,
                "changed_paths": [CHANGED],
                "passed": True,
            },
        )
        test_audit = _write(
            artifacts / "test-audit.json",
            {"schema_version": "deepread-candidate-test-audit-v1"},
        )
        suite = _write(
            artifacts / "suite.json",
            {"schema_version": "deepread-validation-suite-v1"},
        )
        gate = _write(
            artifacts / "gate.json",
            {"schema_version": "deepread-validation-gate-v1"},
        )
        references = {
            "candidate_manifest": {"path": str(manifest), "sha256": _sha(manifest)},
            "modification_plan": {"path": str(plan), "sha256": _sha(plan)},
            "candidate_audit": {"path": str(audit), "sha256": _sha(audit)},
            "candidate_test_audit": {
                "path": str(test_audit),
                "sha256": _sha(test_audit),
            },
            "validation_suite": {"path": str(suite), "sha256": _sha(suite)},
            "validation_gate": {"path": str(gate), "sha256": _sha(gate)},
        }
        basis = {
            "candidate_id": "candidate-1",
            "candidate_snapshot_sha256": snapshot,
            "validation_gate_sha256": _sha(gate),
            "outcome": "accepted",
        }
        basis_hash = _canonical(basis)
        outcome = _write(
            artifacts / "outcome.json",
            {
                "schema_version": "deepread-candidate-outcome-v1",
                "decision_id": "outcome-" + basis_hash[:20],
                "candidate_id": "candidate-1",
                "plan_id": "plan-1",
                "outcome": "accepted",
                "candidate_status": "accepted",
                "promotion_status": "eligible_for_materialization",
                "base_commit": base,
                "candidate_path": str(candidate.resolve()),
                "candidate_snapshot_sha256": snapshot,
                "artifacts": references,
                "git_mutation_performed": False,
                "decision_basis_sha256": basis_hash,
            },
        )
        return candidate, source, outcome, references

    def test_materializes_exact_snapshot_as_clean_detached_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, _, outcome, _ = self._fixture(Path(directory))

            result = materialize_accepted_candidate(outcome_path=outcome)
            repeated = materialize_accepted_candidate(outcome_path=outcome)
            head = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(candidate), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(head, result["materialized_commit"])
        self.assertEqual(repeated["materialized_commit"], result["materialized_commit"])
        self.assertEqual(status, "")
        self.assertEqual(result["status"], "materialized_detached")
        self.assertFalse(result["branch_created"])
        self.assertFalse(result["baseline_updated"])

    def test_rejects_nonaccepted_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, outcome_path, _ = self._fixture(Path(directory))
            outcome = json.loads(outcome_path.read_text())
            outcome["outcome"] = "rejected"
            outcome_path.write_text(json.dumps(outcome))

            with self.assertRaisesRegex(ValueError, "only an accepted"):
                materialize_accepted_candidate(outcome_path=outcome_path)

    def test_rejects_changed_source_after_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, source, outcome, _ = self._fixture(Path(directory))
            source.write_text("value = 3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                materialize_accepted_candidate(outcome_path=outcome)

    def test_rejects_changed_lineage_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, outcome, references = self._fixture(Path(directory))
            plan = Path(references["modification_plan"]["path"])
            plan.write_text("{}")

            with self.assertRaisesRegex(ValueError, "artifact changed"):
                materialize_accepted_candidate(outcome_path=outcome)


if __name__ == "__main__":
    unittest.main()
