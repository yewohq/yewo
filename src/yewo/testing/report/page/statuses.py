"""Presentation words and tones for persisted report statuses."""

from yewo.testing.models import ReportStatus


TONES = {
    ReportStatus.PASSED: "pass",
    ReportStatus.FAILED: "fail",
    ReportStatus.UNVERIFIED: "warn",
    ReportStatus.UNSUPPORTED: "warn",
    ReportStatus.ERROR: "warn",
    ReportStatus.ABORTED: "warn",
    ReportStatus.SKIPPED: "muted",
    ReportStatus.PENDING: "muted",
    ReportStatus.RUNNING: "muted",
}
"""How each verdict is coloured without calling infrastructure an app defect."""


VERBS = {
    ReportStatus.FAILED: "failed",
    ReportStatus.UNVERIFIED: "reached no verdict",
    ReportStatus.UNSUPPORTED: "could not be run here",
    ReportStatus.ERROR: "could not run",
    ReportStatus.ABORTED: "was interrupted",
    ReportStatus.SKIPPED: "was skipped",
    ReportStatus.PENDING: "has not started",
    ReportStatus.RUNNING: "is still running",
}
"""What the headline says a non-passing scenario did."""
