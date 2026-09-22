"""Candidate isolation and audit primitives."""

from .audit import (
    audit_candidate,
    candidate_snapshot_sha256,
    collect_changed_paths,
    verify_candidate_snapshot,
)
from .candidates import create_candidate_worktree
from .test_audit import audit_candidate_tests
from .registry import record_candidate_outcome
from .materialize import materialize_accepted_candidate
from .modifier import CandidateModificationReport, run_candidate_modification
from .baselines import (
    advance_baseline_registry,
    current_baseline,
    initialize_baseline_registry,
)

__all__ = [
    "audit_candidate",
    "candidate_snapshot_sha256",
    "collect_changed_paths",
    "verify_candidate_snapshot",
    "create_candidate_worktree",
    "audit_candidate_tests",
    "record_candidate_outcome",
    "materialize_accepted_candidate",
    "CandidateModificationReport",
    "run_candidate_modification",
    "advance_baseline_registry",
    "current_baseline",
    "initialize_baseline_registry",
]
