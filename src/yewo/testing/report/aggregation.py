"""Aggregate scenario reports into journey counts and status."""

from typing import Sequence

from yewo.testing.models import JourneyStatus, ReportStatus, ScenarioReport


def tally(reports: Sequence[ScenarioReport]) -> tuple[int, int, int]:
    """Passed, failed and skipped, where anything else that ran counts as failed."""
    passed = sum(1 for report in reports if report.status == ReportStatus.PASSED)
    skipped = sum(1 for report in reports if report.status == ReportStatus.SKIPPED)
    return passed, len(reports) - passed - skipped, skipped


def aggregate_journey_status(passed_count: int, failed_count: int) -> JourneyStatus:
    """A suite passes only when at least one scenario has verified evidence."""
    return ReportStatus.PASSED if passed_count > 0 and failed_count == 0 else ReportStatus.FAILED
