"""Connect obligation routing to screen, transition, and metric evidence lanes."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import cast

from yewo.errors import Cause, cause_of
from yewo.llm.failures import reportable
from yewo.llm.llm_client import LLMClient
from yewo.testing.judge import rules, transitions
from yewo.testing.judge.qa_auditor import QAAuditor, ScreenAuditSession
from yewo.testing.judge.screen_audits import LiveScreenAudits
from yewo.testing.models import GeneratedTestCase, RuleVerification
from yewo.testing.obligations import (
    ObligationController,
    ObligationEvent,
    ObligationRunEvidence,
    ScenarioContract,
    select_outcomes,
)
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


def build_audit_lanes(
    llm: LLMClient,
    test_case: GeneratedTestCase,
    security_rules: Sequence[str],
    scrubber: SecretScrubber,
) -> tuple[ObligationController, LiveScreenAudits]:
    """Build independent universal and observation-window audit lanes."""
    controller = ObligationController.from_scenario(
        cast(ScenarioContract, test_case)
    )
    indexes = controller.route(
        ObligationEvent.OBSERVATION,
        evidence_reference="live-observation-window",
    )
    routed_outcomes = select_outcomes(test_case.expected_outcomes, indexes)
    obligation_auditor = (
        QAAuditor(llm, routed_outcomes, [], scrubber=scrubber)
        if any(outcome is not None for outcome in routed_outcomes)
        else None
    )
    universal_auditor = QAAuditor(
        llm, [], list(security_rules), scrubber=scrubber
    )
    return controller, LiveScreenAudits(
        ScreenAuditSession(universal_auditor),
        ScreenAuditSession(obligation_auditor)
        if obligation_auditor is not None
        else None,
    )


def finalize_run_end(
    llm: LLMClient,
    test_case: GeneratedTestCase,
    scrubber: SecretScrubber,
    controller: ObligationController,
    audits: LiveScreenAudits,
) -> None:
    """Route run-end screen outcomes to the final observed state only."""
    indexes = controller.route(
        ObligationEvent.RUN_END,
        evidence_reference="final-observation",
    )
    routed_outcomes = select_outcomes(test_case.expected_outcomes, indexes)
    has_screen_outcomes = any(
        outcome is not None for outcome in rules.on_screen_only(routed_outcomes)
    )
    final_session = (
        ScreenAuditSession(QAAuditor(llm, routed_outcomes, [], scrubber=scrubber))
        if has_screen_outcomes
        else None
    )
    audits.finalize(final_session)


def route_aggregate_evidence(controller: ObligationController) -> None:
    """Open non-screen evidence windows before their deterministic judges run."""
    controller.route(
        ObligationEvent.TRANSITION,
        evidence_reference="screen-order",
    )
    controller.route(
        ObligationEvent.MEASUREMENT,
        evidence_reference="measurement-samples",
    )


def settle_transitions(
    llm: LLMClient,
    test_case: GeneratedTestCase,
    assessments: Sequence,
    scrubber: SecretScrubber,
) -> list[RuleVerification]:
    """Judge screen-order obligations without converting local errors to evidence."""
    try:
        return transitions.settle(
            llm,
            test_case.expected_outcomes,
            assessments,
            scrubber=scrubber,
        )
    except Exception as exc:  # noqa: BLE001 - model boundary
        if cause_of(exc) is Cause.PROVIDER:
            raise
        logger.warning("Could not judge scenario transitions: %s", reportable(exc))
        return []
def settle_obligations(
    controller: ObligationController,
    verifications: Sequence[RuleVerification],
) -> ObligationRunEvidence:
    """Accept only routed expected-outcome evidence and close every window."""
    for verification in verifications:
        identifier = verification.rule_id
        if not rules.is_kind(identifier, rules.EXPECTED) or identifier is None:
            continue
        outcome_index = int(identifier.split(":", 1)[1])
        controller.record_verification(
            outcome_index,
            verification.status,
            verification.applicable,
            verification.screen_name or identifier,
        )
    return ObligationRunEvidence(
        routes=list(controller.route_receipts),
        settlements=list(controller.close_run()),
        readiness=controller.readiness(),
    )


def snapshot_evidence(
    controller: ObligationController | None,
) -> ObligationRunEvidence:
    """Close a partial run without converting unfinished obligations to passes."""
    if controller is None:
        return ObligationRunEvidence()
    return ObligationRunEvidence(
        routes=list(controller.route_receipts),
        settlements=list(controller.close_run()),
        readiness=controller.readiness(),
    )
