"""Capture one dispatched action's settlement and recovery evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.actuation.outcome_verifier import (
    OUTCOME_VERIFICATION_MODEL_REQUESTS,
    OutcomeVerdict,
    OutcomeVerifier,
    apply_outcome_assessment,
    needs_outcome_verification,
)
from yewo.agent.actuation import (
    ActionExecutor,
    ActionKind,
    ActionOutcome,
    ActionOutcomeVerifier,
    ActionProposal,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.agent.runtime.application_state import (
    ApplicationRecovery,
    application_terminal_reason,
)
from yewo.agent.narration import narrate, trace
from yewo.agent.planning.evidence import PlannerDecisionMetadata
from yewo.agent.planning.history import (
    PlanningHistoryEntry,
    build_planning_history_entry,
)
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentStep,
    AgentStepStore,
    AgentTerminalReason,
)
from yewo.agent.runtime.observation_settlement import (
    SettlementInterrupted,
    capture_settled_observation,
    resolve_action_settlement_policy,
)
from yewo.grounding import GroundingEvidence
from yewo.devices.focus import FocusedFieldReading
from yewo.observations import (
    Observation,
    ObservationCaptureError,
    ObservationService,
    ScreenFingerprint,
)
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
    outcome: ActionOutcome | None
    application_failure: ActionApplicationFailure | None
    outcome_verification_requests: int = 0

    def planning_history_entry(
        self,
        proposal: ActionProposal,
        target_grounding: GroundingEvidence | None,
        source_observation: Observation,
    ) -> PlanningHistoryEntry:
        """Preserve the action's actual settled edge, not a recovered screen."""
        if self.outcome is None:
            raise ValueError("An unsettled application failure cannot enter planning history.")
        return build_planning_history_entry(
            proposal,
            self.outcome,
            target_grounding,
            source_observation,
            self.settled_observation,
            self.application_failure,
        )


def record_settled_action(
    step_store: AgentStepStore,
    step_index: int,
    current_observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    target_grounding: GroundingEvidence | None,
    destination_grounding: GroundingEvidence | None,
    authorization: AuthorizedAction,
    receipt: DispatchReceipt,
    settled_action: SettledAction,
) -> AgentStep:
    """Persist one settled dispatch without duplicating its evidence mapping."""
    return step_store.record(
        step_index,
        current_observation,
        proposal,
        decision,
        settled_action.terminal,
        target_grounding=target_grounding,
        destination_grounding=destination_grounding,
        authorization=authorization,
        receipt=receipt,
        settlement_observation_ids=settled_action.settlement_observation_ids,
        settled_observation_id=(
            settled_action.settled_observation.observation_id
            if settled_action.settlement_completed
            else None
        ),
        recovery_observation_id=settled_action.recovery_observation_id,
        outcome=settled_action.outcome,
        application_failure=settled_action.application_failure,
    )


def unsettled_application_failure_reason(settled_action: SettledAction) -> str:
    """Describe an application exit that prevented outcome capture."""
    if settled_action.application_failure is not None:
        return settled_action.application_failure.summary
    return "The application exited before outcome evidence could be captured."


def application_terminal_conclusion(
    settled_action: SettledAction,
) -> tuple[AgentTerminalReason, str] | None:
    """Return the terminal and reason for a post-dispatch application exit."""
    terminal = settled_action.terminal
    if terminal is None:
        return None
    if terminal not in {
        AgentTerminalReason.APPLICATION_EXITED,
        AgentTerminalReason.APPLICATION_INACTIVE,
    }:
        return None
    reason = (
        settled_action.application_failure.summary
        if settled_action.application_failure is not None
        else application_terminal_reason(terminal, initial=False)
    )
    return terminal, reason


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
    settlement_policy_source: object,
    interruption: InterruptionSignal,
    sleep: Callable[[float], None],
    steps: list[AgentStep],
    before_focus: FocusedFieldReading | None,
    spent_model_requests: int,
    baseline: ScreenFingerprint | None = None,
    predicted_outcome_verifier: OutcomeVerifier | None = None,
) -> SettledAction:
    """Settle one action and preserve partial evidence before propagating failures."""
    try:
        settlement_policy = resolve_action_settlement_policy(
            mode_policy,
            settlement_policy_source,
            decision,
            proposal.action,
        )
        trace(
            f"step={step_index} settlement_initial_ms="
            f"{settlement_policy.initial_delay_seconds * 1_000:.1f} "
            f"settlement_interval_ms={settlement_policy.interval_seconds * 1_000:.1f} "
            f"settlement_maximum_observations={settlement_policy.maximum_observations} "
            f"settlement_minimum_observations={settlement_policy.minimum_observations} "
            f"settlement_required_observations="
            f"{settlement_policy.required_stable_observations}"
        )
        settlement = capture_settled_observation(
            observations,
            current_observation,
            settlement_policy,
            interruption,
            sleep,
        )
        trace(
            f"step={step_index} settlement_observations="
            f"{len(settlement.observation_ids)} "
            f"settlement_completed={str(settlement.is_settled).lower()}"
        )
        after = settlement.observation
        if not settlement.is_settled:
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
            after if settlement.is_settled else None,
            baseline=baseline,
            before_focus=verified_before_focus,
            after_focus=after_focus,
            text_delivery=text_delivery,
            scroll_exhausted=executor.scroll_exhausted(authorization),
        )
        outcome, outcome_verification_requests = _verified_prediction(
            predicted_outcome_verifier,
            authorization,
            current_observation,
            after if settlement.is_settled else None,
            outcome,
            step_index,
            mode_policy,
            spent_model_requests,
            baseline,
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
            settlement_completed=settlement.is_settled,
            terminal=terminal,
            recovery_observation_id=recovery_id,
            outcome=outcome,
            application_failure=failure,
            outcome_verification_requests=outcome_verification_requests,
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
    except ObservationCaptureError as error:
        failure = application_recovery.diagnose_action(authorization)
        if failure is not None:
            narrate(
                f"Step {step_index} observation capture failed after a diagnosed application exit."
            )
            executor.discard_text_delivery(authorization)
            return SettledAction(
                observation=current_observation,
                settled_observation=current_observation,
                settlement_observation_ids=(),
                settlement_completed=False,
                terminal=AgentTerminalReason.APPLICATION_EXITED,
                recovery_observation_id=None,
                outcome=None,
                application_failure=failure,
            )
        _record_capture_failure(
            error,
            executor,
            authorization,
            step_store,
            step_index,
            current_observation,
            proposal,
            decision,
            target_grounding,
            destination_grounding,
            receipt,
            steps,
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


def _record_capture_failure(
    error: ObservationCaptureError,
    executor: ActionExecutor,
    authorization: AuthorizedAction,
    step_store: AgentStepStore,
    step_index: int,
    current_observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    target_grounding: GroundingEvidence | None,
    destination_grounding: GroundingEvidence | None,
    receipt: DispatchReceipt,
    steps: list[AgentStep],
) -> None:
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


def _screen_moved_before_dispatch(
    before: Observation,
    baseline: ScreenFingerprint | None,
) -> bool:
    """Whether the screen changed between the planning observation and the dispatch.

    Absent a baseline this answers false. A missing baseline is not evidence of movement.
    Refusing every verification without one would discard the check on any device the runtime
    could not read at dispatch time.
    """
    if baseline is None:
        return False
    return baseline.masked_pixels_sha256 != before.presentation_key.fingerprint.masked_pixels_sha256


def _verified_prediction(
    predicted_outcome_verifier: OutcomeVerifier | None,
    authorization: AuthorizedAction,
    before: Observation,
    after: Observation | None,
    outcome: ActionOutcome,
    step_index: int,
    mode_policy: AgentModePolicy,
    spent_model_requests: int,
    baseline: ScreenFingerprint | None = None,
) -> tuple[ActionOutcome, int]:
    """Ask a reader of both screens whether the predicted outcome happened.

    Only where device evidence left the question open, because every other disposition already
    rests on evidence a second opinion cannot improve. The call reserves its capacity before it
    asks, and returns its count so the run budget charges it. A request counted only after it
    returns lets the run exceed the ceiling it reports.
    """
    if predicted_outcome_verifier is None or after is None:
        return outcome, 0
    if not needs_outcome_verification(outcome):
        return outcome, 0
    if not mode_policy.has_model_request_capacity(
        spent_model_requests, OUTCOME_VERIFICATION_MODEL_REQUESTS
    ):
        # The dispatched action keeps the outcome its device evidence established. The loop
        # reports the exhausted budget on its next pass.
        trace(f"step={step_index} predicted_outcome=skipped reason=model_request_budget")
        return outcome, 0
    if _screen_moved_before_dispatch(before, baseline):
        # The only screenshot held for `before` is the one planning read. Planning, grounding
        # and authorization each take model calls, and the dispatch baseline says the screen
        # changed across them. Comparing that image against the settled one asks about a
        # transition the action did not make.
        trace(f"step={step_index} predicted_outcome=skipped reason=stale_before_screenshot")
        return outcome, 0
    assessment = predicted_outcome_verifier.verify(
        authorization,
        before.screenshot,
        after,
    )
    trace(f"step={step_index} predicted_outcome={assessment.verdict.value}")
    revised = apply_outcome_assessment(outcome, assessment)
    # An ambiguous outcome is revised for every verdict, so object identity cannot say
    # whether the prediction held.
    if assessment.verdict is OutcomeVerdict.NOT_HAPPENED:
        narrate(f"Step {step_index} did not produce the outcome it predicted. {assessment.reason}")
    elif revised is not outcome:
        narrate(f"Step {step_index} outcome read from both screens. {assessment.reason}")
    return revised, OUTCOME_VERIFICATION_MODEL_REQUESTS
