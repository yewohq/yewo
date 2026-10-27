"""Turning a run and its evidence into one word about the scenario.

Nothing here touches a device or a model. It reads what a run recorded and
decides what that adds up to. The ranking between a crash, an infeasible stop,
and missing evidence lives in one place.
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

from yewo.agent.run.outcomes import DeviceRunResult
from yewo.errors import Cause
from yewo.testing.judge import rules
from yewo.testing.models import (
    CrashDiagnostic,
    ExpectedOutcome,
    QAIssue,
    ReportStatus,
    RuleVerification,
    ScenarioStatus,
)
from yewo.agent.run.postconditions import (
    VerificationClassification,
    VerificationStatus,
)

if TYPE_CHECKING:
    from yewo.testing.judge.decide import Feasibility

logger = logging.getLogger(__name__)


def _terminal_outcome(
    run_result: DeviceRunResult,
    stop_reason: Optional[str],
    verifications: List[RuleVerification],
    issues: List[QAIssue],
    outputs: Sequence["Feasibility"],
    findings: Optional[List] = None,
    crash_diagnostic: Optional[CrashDiagnostic] = None,
) -> Optional[tuple[ScenarioStatus, Optional[str], Optional[Cause]]]:
    """Classify outcomes that cannot become a successful verification."""

    # Ranked first: an uncaught crash, fatal exception, or ANR from device logs
    if crash_diagnostic is not None:
        return ReportStatus.FAILED, f"App crashed: {crash_diagnostic.summary}", None

    # A crash screen detected by perception
    if any(issue.issue_type == "crash_screen" for issue in issues):
        return ReportStatus.FAILED, "App crashed: crash screen detected.", None

    latest_postconditions = {}
    for verification in run_result.postcondition_verifications:
        latest_postconditions[verification.value_name] = verification
    failed_postcondition = next(
        (
            verification
            for verification in latest_postconditions.values()
            if verification.status is VerificationStatus.FAILED
            and verification.classification is VerificationClassification.APPLICATION
        ),
        None,
    )
    if failed_postcondition is not None:
        return ReportStatus.FAILED, failed_postcondition.reason, None
    if run_result.status == "hook_stop" and stop_reason and "infeasible" in stop_reason.lower():
        return ReportStatus.SKIPPED, stop_reason, Cause.SCENARIO
    if run_result.status == "exception":
        return (
            ReportStatus.ERROR,
            run_result.failure_reason or "Execution exception",
            run_result.cause or Cause.UNKNOWN,
        )
    if run_result.status in ("loop_detected", "step_limit"):
        return ReportStatus.UNVERIFIED, run_result.failure_reason, Cause.AGENT
    agent_entry_failure = next(
        (
            verification
            for verification in latest_postconditions.values()
            if verification.status is VerificationStatus.FAILED
            and verification.classification is VerificationClassification.AGENT_ACTION
        ),
        None,
    )
    if agent_entry_failure is not None:
        return ReportStatus.UNVERIFIED, agent_entry_failure.reason, Cause.AGENT
    critical = next(
        (
            issue
            for issue in issues
            if issue.severity == "critical"
            and not _unsupported_exposure(issue, findings)
        ),
        None,
    )
    if critical:
        return ReportStatus.FAILED, f"Critical QA issue: {critical.description}", None
    violated = _violated_security_rule(verifications)
    if violated:
        return ReportStatus.FAILED, f"Security rule violated: {violated.rule}", None
    unverified_postcondition = next(
        (
            verification
            for verification in latest_postconditions.values()
            if verification.status is VerificationStatus.UNVERIFIED
        ),
        None,
    )
    if unverified_postcondition is not None:
        return ReportStatus.UNVERIFIED, unverified_postcondition.reason, Cause.EVIDENCE
    # Security rules never reach this. _violated_security_rule above fails on
    # one anywhere, so what is left here is an expected outcome, which is a
    # claim about the screen it names and not about any other.
    passed_rule_ids = _passed_rule_ids(verifications)
    failed = next(
        (
            verification
            for (rule_id, _), verification in _per_screen(verifications).items()
            if verification.status == "failed" and rule_id not in passed_rule_ids
        ),
        None,
    )
    if failed:
        return ReportStatus.FAILED, f"Rule validation failed: {failed.rule}", None
    if outputs and not outputs[-1].is_scenario_feasible:
        return (
            ReportStatus.SKIPPED,
            outputs[-1].infeasibility_reason or "Scenario infeasible",
            Cause.SCENARIO,
        )
    return None


EXPOSURE = "sensitive_data_exposure"


def _unsupported_exposure(issue, findings) -> bool:
    """Whether an exposure claim has no reading of the screen behind it.

    The auditor answers twice about a screen. Its prose can mistake a bare
    field caption for a visible value. Its structured reading must distinguish
    a legible value, a mask, a caption, and an absent field. The structured
    reading is the evidence. An unsupported prose claim remains reported but
    does not fail the scenario.

    With no reading at all there is nothing to weigh, so the claim stands.
    """
    if issue.issue_type != EXPOSURE:
        return False
    screen_findings = [
        finding for finding in (findings or []) if finding.screen_name == issue.current_screen
    ]
    return bool(screen_findings) and not any(finding.is_exposure() for finding in screen_findings)


def _per_screen(verifications: List[RuleVerification]) -> dict:
    """The last answer for each rule, on each screen that could answer it.

    Every rule is put to every screen, so a scenario accumulates many answers
    per rule. Screens the rule cannot apply to answer "untested", and those say
    nothing about it. Reading them as the verdict would let walking away from a
    field decide whether the field worked.

    Kept per screen rather than per rule, because `applicable` is the auditor's
    own claim and nothing can check it. One wrong claim then reaches only the
    screen it was made about.
    """
    latest = {}
    for verification in verifications:
        if verification.rule_id and verification.applicable:
            latest[(verification.rule_id, verification.screen_name or "")] = verification
    return latest


def _passed_rule_ids(verifications: List[RuleVerification]) -> set:
    """The rules some screen ended on a pass.

    A screen that saw an outcome hold is evidence of it, and a later answer
    about a different screen does not take that away. A later answer about the
    same screen does: there, the outcome stopped holding where it could be
    seen.
    """
    return {
        rule_id
        for (rule_id, _), verification in _per_screen(verifications).items()
        if verification.status == "passed"
    }


def per_rule(verifications: List[RuleVerification]) -> dict:
    """How each rule settled, as one answer per rule id.

    The auditor is put to every screen, and answers every rule on each of
    them. So a scenario holds many answers per rule, and most say "not
    here". A reader wants the answer that decided the rule. It is the one
    the verdict was taken from.

    The two kinds settle differently, and this follows the verdict rather
    than averaging them. An expected outcome holds where it is claimed, so
    a pass anywhere settles it. A security rule describes what must never
    happen, so a failure anywhere stands and no later screen takes it back.
    Reading a security rule the other way would show a violation as passed
    on a run the judge failed for it.

    A rule nothing could answer is absent. It was never decided, and
    reporting it either way would invent an answer.
    """
    violated = {
        verification.rule_id: verification
        for verification in verifications
        if verification.status == "failed"
        and rules.is_kind(verification.rule_id, rules.SECURITY)
    }
    latest = _per_screen(verifications)
    passed_rule_ids = _passed_rule_ids(verifications)
    decided = dict(violated)
    for (rule_id, _), verification in latest.items():
        if rule_id in violated:
            continue
        if rule_id in passed_rule_ids:
            if verification.status == "passed":
                decided[rule_id] = verification
        elif rule_id not in decided and verification.status == "failed":
            decided[rule_id] = verification
    return decided


def _violated_security_rule(verifications: List[RuleVerification]):
    """A security rule failing anywhere is a violation, however the run ended.

    Unlike an expected outcome, these describe what must never happen. A
    sensitive value exposed at step 4 stays exposed, whatever step 9 shows.
    """
    return next(
        (
            verification
            for verification in verifications
            if verification.status == "failed"
            and rules.is_kind(verification.rule_id, rules.SECURITY)
        ),
        None,
    )


def _missing_required_ids(
    verifications: List[RuleVerification],
    required_outcomes: Sequence[ExpectedOutcome | str],
    security_rules: Sequence[str],
) -> list[str]:
    """Lists the required assertions a scenario ended without evidencing.

    The two kinds are judged differently because they mean different things.

    An expected outcome is evidenced once a screen ended on a pass for it. Many
    outcomes can only be seen in passing. A field accepts a long value, and
    the next screen has no such field. A later "failed" about the same screen
    takes the evidence away, because the outcome stopped holding where it could
    be seen. A later "failed" about a different screen does not: the outcome
    was never a claim about that screen.

    A security rule describes what must never happen. The auditor answers every
    rule on every screen, so a mention alone proves nothing. A rule counts as
    required only once some screen could have broken it. One pass anywhere
    evidences it. It stays evidenced whatever later screens show, because a
    failure anywhere already failed the scenario.
    """
    expected_ids = rules.ids(required_outcomes, rules.EXPECTED)
    security_ids = rules.ids(security_rules, rules.SECURITY)
    applicable = {
        verification.rule_id
        for verification in verifications
        if verification.rule_id and verification.applicable
    }
    evidenced = {
        verification.rule_id
        for verification in verifications
        if verification.rule_id
        and verification.applicable
        and verification.status == "passed"
    }
    passed_expected_ids = _passed_rule_ids(verifications) & expected_ids
    required_ids = expected_ids | (security_ids & applicable)
    return sorted(required_ids - passed_expected_ids - (evidenced & security_ids))
