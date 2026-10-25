"""Page values for deterministic run-level policy checks."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from yewo.testing.models import PolicyCheckStatus

from .source import RunSource


@dataclass(frozen=True)
class PolicyCheckView:
    """One deterministic run-level check as the page presents it."""

    check_id: str
    tone: str
    status: str
    platform: str
    source: str
    evidence: str
    waiver_reason: str = ""


def policy_check_views(report: RunSource) -> Tuple[PolicyCheckView, ...]:
    """Run-level checks in their recorded order."""
    tones = {
        PolicyCheckStatus.PASSED: "pass",
        PolicyCheckStatus.FAILED: "fail",
        PolicyCheckStatus.WAIVED: "muted",
        PolicyCheckStatus.UNSUPPORTED: "warn",
        PolicyCheckStatus.EVIDENCE_MISSING: "warn",
    }
    return tuple(
        PolicyCheckView(
            check_id=check.check_id,
            tone=tones[check.status],
            status=check.status.value.replace("_", " ").title(),
            platform=check.platform.value,
            source=check.source,
            evidence=check.evidence,
            waiver_reason=check.waiver_reason or "",
        )
        for check in report.policy_checks
    )


def policy_checks_succeeded(checks: Sequence[PolicyCheckView]) -> bool:
    """Whether the page can describe the policy result as accepted."""
    return all(check.tone in ("pass", "muted") for check in checks)
