"""Capture one dispatched action's settlement and recovery evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.action_diagnostics import ActionApplicationFailure
from yewo.agent.actuation import (
    ActionExecutor,
    ActionKind,
    ActionOutcome,
    ActionOutcomeVerifier,
    ActionProposal,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.agent.application_state import ApplicationRecovery
from yewo.agent.narration import narrate
from yewo.agent.planning_evidence import PlannerDecisionMetadata
from yewo.agent.planning_history import (
    PlanningHistoryEntry,
    build_planning_history_entry,
)
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentStep,
    AgentStepStore,
    AgentTerminalReason,
)
from yewo.agent.settlement import SettlementInterrupted, capture_settled_observation
from yewo.grounding import GroundingEvidence
from yewo.devices.focus import FocusedFieldReading
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted


@dataclass(frozen=True)
class SettledAction:
    """Evidence captured after one accepted device dispatch."""

    observation: Observation
    settled_observation: Observation
    settlement_observation_ids: tuple[str, ...]
    settlement_completed: bool
    terminal: AgentTerminalReason | None
    recovery_observation_id: str | None
    outcome: ActionOutcome
    application_failure: ActionApplicationFailure | None

    def planning_history_entry(
        self,
        proposal: ActionProposal,
        target_grounding: GroundingEvidence | None,
        source_observation: Observation,
    ) -> PlanningHistoryEntry:
        """Preserve the action's actual settled edge, not a recovered screen."""
        return build_planning_history_entry(
            proposal,
            self.outcome,
            target_grounding,
            source_observation,
            self.settled_observation,
            self.application_failure,
        )


def require_agent_continuation(interruption: InterruptionSignal) -> None:
    """Raise at an explicit Device Agent interruption boundary."""
    if interruption.is_set():
        raise RunWorkflowInterrupted


def settle_dispatched_action(
    observations: ObservationService,
    executor: ActionExecutor,
    outcome_verifier: ActionOutcomeVerifier,
    application_recovery: ApplicationRecovery,
    step_store: AgentStepStore,
    step_index: int,
    current_observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    target_grounding: GroundingEvidence | None,
    destination_grounding: GroundingEvidence | None,
    authorization: AuthorizedAction,
    receipt: DispatchReceipt,
    mode_policy: AgentModePolicy,
    interruption: InterruptionSignal,
    sleep: Callable[[float], None],
    steps: list[AgentStep],
    before_focus: FocusedFieldReading | None,
) -> SettledAction:
    """Settle one action and preserve partial evidence before propagating failures."""
    try:
        settlement = capture_settled_observation(
            observations,
            current_observation,
            mode_policy,
            interruption,
            sleep,
        )
        after = settlement.observation
        if not settlement.settled:
            executor.discard_text_delivery(authorization)
            verified_before_focus = None
            after_focus = None
            text_delivery = None
        else:
            verified_before_focus = before_focus
            after_focus = read_focused_field(executor, authorization)
            text_delivery = executor.verify_text_delivery(authorization)
        outcome = outcome_verifier.verify(
            authorization,
            receipt,
            current_observation,
            after if settlement.settled else None,
            before_focus=verified_before_focus,
            after_focus=after_focus,
            text_delivery=text_delivery,
        )
        recovered, terminal, recovery_id, outcome, failure = application_recovery.apply_action(
            authorization,
            outcome,
            after,
            interruption,
        )
        return SettledAction(
            observation=recovered,
            settled_observation=after,
            settlement_observation_ids=settlement.observation_ids,
            settlement_completed=settlement.settled,
            terminal=terminal,
            recovery_observation_id=recovery_id,
            outcome=outcome,
            application_failure=failure,
        )
    except RunWorkflowInterrupted as error:
        narrate(f"The run interrupted step {step_index} during settlement.")
        executor.discard_text_delivery(authorization)
        settlement_ids = error.observation_ids if isinstance(error, SettlementInterrupted) else ()
        _record_failure(
            step_store,
            step_index,
            current_observation,
            proposal,
            decision,
            target_grounding,
            destination_grounding,
            authorization,
            receipt,
            AgentTerminalReason.INTERRUPTED,
            steps,
            error,
            settlement_ids,
        )
        raise
    except BaseException as error:
        narrate(f"Step {step_index} outcome capture failed: {type(error).__name__}.")
        executor.discard_text_delivery(authorization)
        _record_failure(
            step_store,
            step_index,
            current_observation,
            proposal,
            decision,
            target_grounding,
            destination_grounding,
            authorization,
            receipt,
            AgentTerminalReason.RUNTIME_FAILURE,
            steps,
            error,
            (),
        )
        raise


def _record_failure(
    step_store: AgentStepStore,
    step_index: int,
    observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    target_grounding: GroundingEvidence | None,
    destination_grounding: GroundingEvidence | None,
    authorization: AuthorizedAction,
    receipt: DispatchReceipt,
    terminal: AgentTerminalReason,
    steps: list[AgentStep],
    error: BaseException,
    settlement_observation_ids: tuple[str, ...],
) -> None:
    step_store.record_partial_or_raise(
        step_index,
        observation,
        proposal,
        decision,
        target_grounding,
        destination_grounding,
        authorization,
        receipt,
        terminal,
        steps,
        error,
        settlement_observation_ids=settlement_observation_ids,
    )


def read_focused_field(
    executor: ActionExecutor,
    action: AuthorizedAction,
) -> FocusedFieldReading | None:
    if action.action not in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}:
        return None
    return executor.focused_field_reading()
