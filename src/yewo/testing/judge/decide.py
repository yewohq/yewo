"""One scenario's verdict, decided from what the run recorded.

Nothing here touches a device or a model. It takes the evidence a run
produced and returns one word about it. The same decision is then reached
twice: once at the end of a live run, and again over stored evidence.

That second path makes the decision testable as data. A fixture holds the
evidence and the verdict it must produce. A change to the ranking then shows
up as a fixture that stopped agreeing. A unit test would instead be edited to
match the new behaviour, in the same commit that changed it.

The ranking itself lives in `verdicts.py`. This module holds the order the
checks run in, and each check names the cause it reports.
"""

import logging
from typing import List, NamedTuple, Optional, Protocol, Sequence

from yewo.agent.run.outcomes import DeviceRunResult
from yewo.errors import Cause
from yewo.testing.judge import rules
from yewo.testing.judge.capabilities import unconfirmed_capabilities
from yewo.testing.judge.verdicts import _missing_required_ids, _terminal_outcome
from yewo.testing.models import (
    ExpectedOutcome,
    QAIssue,
    ReportStatus,
    RuleVerification,
    SensitiveFinding,
    ScenarioStatus,
)

class Feasibility(Protocol):
    """What the verdict reads off the planner's last answer.

    A protocol rather than the response model, because two things answer it.
    A live run passes what the planner said. A stored run passes what the
    record kept, which is these two fields and nothing else.
    """

    is_scenario_feasible: bool
    infeasibility_reason: Optional[str]


logger = logging.getLogger(__name__)


class Verdict(NamedTuple):
    """One word about a scenario, the reason for it, and what caused it.

    A tuple, because every caller before this module existed unpacked three
    values. The names are for the callers that came after.
    """

    status: ScenarioStatus
    reason: Optional[str]
    cause: Optional[Cause]


def determine_status(
    run_result: DeviceRunResult,
    stop_reason: Optional[str],
    all_verifications: List[RuleVerification],
    all_issues: List[QAIssue],
    qa_outputs: Sequence[Feasibility],
    assessments: Sequence[object],
    required_outcomes: Sequence[ExpectedOutcome | str],
    security_rules: Sequence[str],
    all_findings: Optional[List[SensitiveFinding]] = None,
    required_capabilities: Sequence = (),
) -> Verdict:
    """What this scenario adds up to.

    Args:
        run_result: How the walk ended.
        stop_reason: Why a hook stopped it, where one did.
        all_verifications: Every answer the auditor gave about a rule.
        all_issues: Every defect the auditor reported.
        qa_outputs: What the planner answered, for feasibility alone.
        assessments: The screens that were judged. Counted, never read, so a
            caller that kept only how many there were can say so.
        required_outcomes: What the scenario declared it would show.
        security_rules: What must never happen.
        all_findings: What was seen where a sensitive value could be.
        required_capabilities: The device states the scenario declared.
    """
    # Feasibility stays the planner's call: it is a statement about whether
    # the agent can proceed, not about whether the app is correct.
    terminal = _terminal_outcome(
        run_result,
        stop_reason,
        all_verifications,
        all_issues,
        qa_outputs,
        all_findings,
    )
    if terminal:
        return Verdict(*terminal)

    if run_result.status != "goal_reached" or not assessments:
        return Verdict(
            ReportStatus.UNVERIFIED,
            "The scenario did not produce verified completion evidence.",
            Cause.AGENT,
        )

    # Completion is the planner's call, because it is the only one told what
    # the scenario set out to do. The auditor is deliberately blind to the
    # plan, so it cannot judge whether a plan finished.
    #
    # Independence is not lost. The expected outcomes reach the auditor as
    # rules, and the checks below refuse the scenario unless the auditor
    # evidenced every one of them on a screen it judged without the plan.
    unprovable = rules.unprovable(required_outcomes)
    if unprovable:
        return Verdict(
            ReportStatus.UNVERIFIED,
            "Nothing recorded during a run can settle these outcomes: "
            + ", ".join(unprovable)
            + ". They ask how the app performed, and Yewo measures no part of that.",
            Cause.SCENARIO,
        )

    # A declared capability that never reached the device leaves every
    # outcome below it describing a state the run never entered.
    unconfirmed = (
        unconfirmed_capabilities(required_capabilities, run_result.capability_receipts)
        if required_capabilities
        else []
    )
    if unconfirmed:
        return Verdict(
            ReportStatus.UNVERIFIED,
            "The scenario declared device states it never had confirmed: "
            + ", ".join(unconfirmed),
            Cause.EVIDENCE,
        )

    missing = _missing_required_ids(all_verifications, required_outcomes, security_rules)
    if missing:
        return Verdict(
            ReportStatus.UNVERIFIED,
            "Required verification evidence is missing or untested: " + ", ".join(missing),
            Cause.EVIDENCE,
        )

    # Named last so a specific gap is reported in preference to this.
    if not required_outcomes:
        return Verdict(
            ReportStatus.UNVERIFIED,
            "The scenario declared no expected outcome, so reaching the goal "
            "rests on the planner's own word.",
            Cause.SCENARIO,
        )

    return Verdict(ReportStatus.PASSED, None, None)
