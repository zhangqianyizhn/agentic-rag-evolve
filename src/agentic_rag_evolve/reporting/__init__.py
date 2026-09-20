"""Compact, lineage-checked reports for completed evolution iterations."""

from .iteration import ITERATION_REPORT_SCHEMA, build_iteration_report
from .noop import build_no_candidate_iteration_report

__all__ = [
    "ITERATION_REPORT_SCHEMA",
    "build_iteration_report",
    "build_no_candidate_iteration_report",
]
