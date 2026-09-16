"""Candidate isolation and audit primitives."""

from .audit import audit_candidate, candidate_snapshot_sha256, collect_changed_paths
from .candidates import create_candidate_worktree
from .test_audit import audit_candidate_tests
from .registry import record_candidate_outcome
from .materialize import materialize_accepted_candidate

__all__ = [
    "audit_candidate",
    "candidate_snapshot_sha256",
    "collect_changed_paths",
    "create_candidate_worktree",
    "audit_candidate_tests",
    "record_candidate_outcome",
    "materialize_accepted_candidate",
]
