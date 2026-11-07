"""Live expectation evaluation for one compiled procedure step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    TextDeliveryReason,
    TextDeliveryStatus,
)
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    ReplayProcedureStep,
)
from yewo.agent.replay.state_matching import (
    state_matches_observation,
    state_metadata_matches_observation,
)
from yewo.observations import Observation, ScreenshotReference


ProcedureExpectationFailure = tuple[
    Literal["runtime", "expectation"],
    str | None,
    str,
]


@dataclass(frozen=True)
class ProcedureExpectationEvaluation:
    """One expectation failure or a narrow keyboard-state equivalence."""

    failure: ProcedureExpectationFailure | None = None
    keyboard_equivalent_state_id: str | None = None


def evaluate_procedure_expectation(
    observation: Observation,
    step: ReplayProcedureStep,
    feedback: PlanningHistoryEntry,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
) -> ProcedureExpectationEvaluation:
    """Evaluate action evidence and its expected procedure destination."""
    expected_state_id = step.expects.destination_state_id
    if feedback.refusal is not None or feedback.receipt is not None:
        return _failure(
            "runtime",
            expected_state_id,
            "The runtime refused the procedure action before settlement.",
        )
    outcome = feedback.outcome
    if outcome is None:
        return _failure(
            "expectation",
            expected_state_id,
            "The procedure action has no settled outcome evidence.",
        )
    if step.expects.required_signals - outcome.signals:
        return _failure(
            "expectation",
            expected_state_id,
            "The procedure action did not produce every required outcome signal.",
        )
    destination_matches = expected_state_id is None or state_matches_observation(
        observation,
        evidence.states[expected_state_id],
        evidence.state_grouping_maximum_phash_distance,
        screenshot_loader,
    )
    keyboard_contract_matches = expected_state_id is not None and (
        _focus_contract_matches(observation, step, outcome, evidence)
        or _text_delivery_contract_matches(observation, step, outcome, evidence)
    )
    if not destination_matches and not keyboard_contract_matches:
        return _failure(
            "expectation",
            expected_state_id,
            "The current observation does not match the procedure expectation.",
        )
    return ProcedureExpectationEvaluation(
        keyboard_equivalent_state_id=(
            expected_state_id if not destination_matches and keyboard_contract_matches else None
        )
    )


def _focus_contract_matches(
    observation: Observation,
    step: ReplayProcedureStep,
    outcome: ActionOutcome,
    evidence: EvidenceDocument,
) -> bool:
    expected_state_id = step.expects.destination_state_id
    return (
        expected_state_id is not None
        and step.proposal.action is ActionKind.FOCUS
        and ActionOutcomeSignal.FOCUS_CHANGED in outcome.signals
        and state_metadata_matches_observation(
            observation,
            evidence.states[expected_state_id],
        )
    )


def _text_delivery_contract_matches(
    observation: Observation,
    step: ReplayProcedureStep,
    outcome: ActionOutcome,
    evidence: EvidenceDocument,
) -> bool:
    expected_state_id = step.expects.destination_state_id
    if (
        expected_state_id is None
        or step.proposal.action not in {ActionKind.TYPE, ActionKind.CLEAR}
        or outcome.disposition is not ActionOutcomeDisposition.OBSERVED
        or outcome.text_delivery is None
        or not state_metadata_matches_observation(
            observation,
            evidence.states[expected_state_id],
        )
    ):
        return False
    delivery = outcome.text_delivery
    if delivery.status is TextDeliveryStatus.MATCHED:
        return True
    return (
        step.proposal.action is ActionKind.TYPE
        and delivery.status is TextDeliveryStatus.UNVERIFIABLE
        and delivery.reason is TextDeliveryReason.SECURE_FIELD
        and delivery.is_secure is True
    )


def _failure(
    guard: Literal["runtime", "expectation"],
    expected_state_id: str | None,
    reason: str,
) -> ProcedureExpectationEvaluation:
    return ProcedureExpectationEvaluation(
        failure=(guard, expected_state_id, reason),
    )
