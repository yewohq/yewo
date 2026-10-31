"""Run one bounded observe-plan-act loop across workflow task briefs."""

from __future__ import annotations

from collections.abc import Callable
import time

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionExecutor,
    ActionKind,
    ActionOutcomeVerifier,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    ActionSource,
    AuthorizedAction,
    DispatchDisposition,
)
from yewo.agent.application_state import (
    ApplicationRecovery,
    application_terminal_reason,
    observation_terminal,
)
from yewo.agent.app_memory import ScreenComparisonPolicy
from yewo.agent.no_progress import (
    ActionNoProgressTracker,
    grounding_correction_for_history,
    proposal_refusal as _proposal_refusal,
    record_outcome_with_facts as _record_outcome,
    record_refusal_with_facts as _record_refusal,
)
from yewo.agent.narration import (
    narrate as _narrate,
    narrate_grounding as _narrate_grounding,
    narrate_no_progress as _narrate_no_progress,
    narrate_observation as _narrate_observation,
    narrate_outcome as _narrate_outcome,
    narrate_proposal as _narrate_proposal,
    narrate_recovery as _narrate_recovery,
    narrate_refusal as _narrate_refusal,
    trace_authorization as _trace_authorization,
    trace_dispatch as _trace_dispatch,
)
from yewo.agent.planner import (
    ActionProposalSource,
    PlanningHistoryEntry,
    PlanningRequest,
    build_planning_history_entry,
    build_target_rejection_history_entry,
    grounding_correction_failure_reason,
)
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentStepStore,
    AgentTask,
    AgentTerminalReason,
    available_runtime_actions,
    build_agent_result,
    grounding_budget_reason,
    invocation_budget_reason,
    proposal_terminal,
    required_grounding_requests,
)
from yewo.agent.settlement import SettlementInterrupted, capture_settled_observation
from yewo.artifacts import new_record_identifier
from yewo.devices.focus import FocusedFieldReading
from yewo.grounding import (
    GroundingCorrection,
    GroundingEvidence,
    GroundingOutputShape,
    GroundingRequest,
    GroundingTargetOperation,
    VisualGrounder,
)
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted


class DeviceAgent:
    """Coordinate the shared bounded mobile-operation loop."""

    def __init__(
        self,
        planner: ActionProposalSource,
        grounder: VisualGrounder,
        observations: ObservationService,
        executor: ActionExecutor,
        outcome_verifier: ActionOutcomeVerifier,
        step_store: AgentStepStore,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        application_recovery: Callable[[], bool] | None = None,
        screen_comparison: ScreenComparisonPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._grounder = grounder
        self._observations = observations
        self._executor = executor
        self._outcome_verifier = outcome_verifier
        self._step_store = step_store
        self._identifier_factory = identifier_factory
        self._sleep = sleep
        self._monotonic = monotonic
        self._application_recovery = application_recovery
        self._screen_comparison = screen_comparison or ScreenComparisonPolicy()

    def run(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
    ) -> AgentResult:
        """Operate until the runtime reaches one bounded terminal."""
        started_at = self._monotonic()
        _narrate(f"Starting task: {task.objective}")
        self._require_continuation(interruption)
        current_observation = self._observations.capture()
        _narrate_observation(current_observation, label="Initial")
        initial_terminal = observation_terminal(current_observation)
        application_recovery = ApplicationRecovery(
            self._application_recovery,
            self._observations,
        )
        current_observation, initial_terminal, unused_recovery_id = application_recovery.apply(
            current_observation,
            initial_terminal,
            interruption,
        )
        if initial_terminal is not None:
            return build_agent_result(
                initial_terminal,
                application_terminal_reason(initial_terminal, initial=True),
                current_observation,
                0,
                0,
                0,
                [],
            )
        steps: list[AgentStep] = []
        history: list[PlanningHistoryEntry] = []
        no_progress = ActionNoProgressTracker(self._screen_comparison)
        no_progress.start(current_observation)
        planning_requests = 0
        grounding_requests = 0
        dispatched_actions = 0
        action_policy = ActionPolicy(mode_policy.action_limits, self._identifier_factory)

        def conclude(terminal: AgentTerminalReason, reason: str) -> AgentResult:
            _narrate(f"Terminal {terminal.value}: {reason}")
            return build_agent_result(
                terminal,
                reason,
                current_observation,
                planning_requests,
                grounding_requests,
                dispatched_actions,
                steps,
            )

        for step_index in range(1, mode_policy.maximum_steps + 1):
            self._require_continuation(interruption)
            budget_reason = invocation_budget_reason(
                mode_policy,
                self._monotonic() - started_at,
                planning_requests + grounding_requests,
            )
            if budget_reason is not None:
                return conclude(AgentTerminalReason.BUDGET_EXHAUSTED, budget_reason)
            planning_requests += 1
            available_actions = available_runtime_actions(
                mode_policy.action_limits, current_observation,
                targeted_type_available=task.targeted_type_available,
            )
            proposal = self._planner.plan(
                PlanningRequest(
                    observation=current_observation,
                    objective=task.objective,
                    known_facts=(*task.known_facts, *no_progress.known_facts()),
                    open_work=task.open_work,
                    completion_conditions=task.completion_conditions,
                    declared_value_names=task.declared_value_names,
                    history=tuple(history),
                    available_actions=available_actions,
                )
            )
            _narrate_proposal(step_index, proposal)
            terminal = proposal_terminal(proposal)
            if terminal is not None:
                step = self._step_store.record(step_index, current_observation, proposal, terminal)
                steps.append(step)
                return conclude(
                    terminal,
                    proposal.reason or "The planner requested a terminal.",
                )
            if dispatched_actions >= mode_policy.action_limits.maximum_actions:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
                steps.append(step)
                return conclude(
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                    "No device-action budget remains.",
                )

            refusal = _proposal_refusal(
                no_progress,
                action_policy,
                proposal,
                current_observation,
                ActionBudget(dispatched_actions=dispatched_actions),
            )
            if refusal is not None:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    refusal=refusal,
                )
                steps.append(step)
                _narrate_refusal(step_index, refusal)
                history.append(PlanningHistoryEntry(proposal=proposal, refusal=refusal))
                if reason := _record_refusal(
                    no_progress, proposal, current_observation, refusal, _narrate_no_progress
                ):
                    return conclude(AgentTerminalReason.CANNOT_PROGRESS, reason)
                continue

            grounding_count = required_grounding_requests(proposal)
            grounding_budget = grounding_budget_reason(
                mode_policy,
                planning_requests,
                grounding_requests,
                grounding_count,
            )
            if grounding_budget is not None:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
                steps.append(step)
                return conclude(
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                    grounding_budget,
                )

            target_grounding: GroundingEvidence | None = None
            destination_grounding: GroundingEvidence | None = None
            try:
                correction = grounding_correction_for_history(proposal, history)
                target_grounding = self._ground_target(
                    proposal, current_observation, correction
                )
                destination_grounding = self._ground_destination(proposal, current_observation)
            except BaseException as error:
                _narrate(f"Step {step_index} grounding failed: {type(error).__name__}.")
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding,
                    destination_grounding,
                    None,
                    None,
                    AgentTerminalReason.RUNTIME_FAILURE,
                    steps,
                    error,
                )
                raise
            grounding_requests += grounding_count
            _narrate_grounding(step_index, "target", target_grounding)
            _narrate_grounding(step_index, "destination", destination_grounding)
            authorization = action_policy.authorize(
                proposal,
                current_observation,
                ActionBudget(
                    dispatched_actions=dispatched_actions,
                ),
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                source=ActionSource.PLANNER,
            )
            if isinstance(authorization, ActionRefusal):
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding=target_grounding,
                    destination_grounding=destination_grounding,
                    refusal=authorization,
                )
                steps.append(step)
                _narrate_refusal(step_index, authorization)
                history.append(PlanningHistoryEntry(proposal=proposal, refusal=authorization))
                _record_refusal(
                    no_progress, proposal, current_observation, authorization, _narrate_no_progress
                )
                continue

            _trace_authorization(step_index, authorization)
            try:
                _narrate(f"Step {step_index} dispatching {proposal.action.value}.")
                before_focus = self._focus_reading(authorization)
                receipt = self._executor.dispatch(authorization)
                _trace_dispatch(step_index, receipt)
            except BaseException as error:
                _narrate(f"Step {step_index} dispatch failed: {type(error).__name__}.")
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding,
                    destination_grounding,
                    authorization,
                    None,
                    AgentTerminalReason.RUNTIME_FAILURE,
                    steps,
                    error,
                )
                raise
            if receipt.disposition is DispatchDisposition.DEVICE_REFUSED:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    AgentTerminalReason.DEVICE_REFUSED,
                    target_grounding=target_grounding,
                    destination_grounding=destination_grounding,
                    authorization=authorization,
                    receipt=receipt,
                )
                steps.append(step)
                _narrate(
                    f"Step {step_index} device refusal: "
                    f"{receipt.reason or 'The device refused the action.'}"
                )
                return conclude(
                    AgentTerminalReason.DEVICE_REFUSED,
                    receipt.reason or "The device refused the authorized action.",
                )
            if receipt.disposition is DispatchDisposition.TARGET_REJECTED:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding=target_grounding,
                    destination_grounding=destination_grounding,
                    authorization=authorization,
                    receipt=receipt,
                )
                steps.append(step)
                reason = receipt.reason or "The grounded target rejected the action."
                _narrate(f"Step {step_index} target rejected: {reason}")
                history_entry = build_target_rejection_history_entry(
                    proposal, receipt, target_grounding
                )
                history.append(history_entry)
                refusal = ActionRefusal(
                    proposal_id=proposal.proposal_id,
                    code=ActionRefusalCode.INVALID_GROUNDING,
                    reason=reason,
                )
                _record_refusal(
                    no_progress, proposal, current_observation, refusal, _narrate_no_progress
                )
                continue
            dispatched_actions += 1
            try:
                settlement = capture_settled_observation(
                    self._observations,
                    current_observation,
                    mode_policy,
                    interruption,
                    self._sleep,
                )
                after = settlement.observation
                settlement_observation_ids = settlement.observation_ids
                if not settlement.settled:
                    self._executor.discard_text_delivery(authorization)
                    before_focus = None
                    after_focus = None
                    text_delivery = None
                else:
                    after_focus = self._focus_reading(authorization)
                    text_delivery = self._executor.verify_text_delivery(authorization)
                outcome = self._outcome_verifier.verify(
                    authorization,
                    receipt,
                    current_observation,
                    after if settlement.settled else None,
                    before_focus=before_focus,
                    after_focus=after_focus,
                    text_delivery=text_delivery,
                )
                settled_observation = after
                step_terminal = observation_terminal(after)
                after, step_terminal, recovery_observation_id = application_recovery.apply(
                    after,
                    step_terminal,
                    interruption,
                )
            except RunWorkflowInterrupted as error:
                _narrate(f"The run interrupted step {step_index} during settlement.")
                self._executor.discard_text_delivery(authorization)
                settlement_ids = (
                    error.observation_ids if isinstance(error, SettlementInterrupted) else ()
                )
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding,
                    destination_grounding,
                    authorization,
                    receipt,
                    AgentTerminalReason.INTERRUPTED,
                    steps,
                    error,
                    settlement_observation_ids=settlement_ids,
                )
                raise
            except BaseException as error:
                _narrate(f"Step {step_index} outcome capture failed: {type(error).__name__}.")
                self._executor.discard_text_delivery(authorization)
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding,
                    destination_grounding,
                    authorization,
                    receipt,
                    AgentTerminalReason.RUNTIME_FAILURE,
                    steps,
                    error,
                )
                raise

            step = self._step_store.record(
                step_index,
                current_observation,
                proposal,
                step_terminal,
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                authorization=authorization,
                receipt=receipt,
                settlement_observation_ids=settlement_observation_ids,
                settled_observation_id=(
                    settled_observation.observation_id if settlement.settled else None
                ),
                recovery_observation_id=recovery_observation_id,
                outcome=outcome,
            )
            steps.append(step)
            _narrate_observation(after, label=f"Step {step_index}")
            _narrate_outcome(step_index, outcome)
            _narrate_recovery(step_index, recovery_observation_id)
            progress_terminal = _record_outcome(
                no_progress, proposal, current_observation, after, outcome, _narrate_no_progress
            )
            current_observation = after
            if step_terminal is AgentTerminalReason.APPLICATION_EXITED:
                return conclude(
                    AgentTerminalReason.APPLICATION_EXITED,
                    "The application exited after the dispatched action.",
                )
            if step_terminal is AgentTerminalReason.APPLICATION_INACTIVE:
                return conclude(
                    AgentTerminalReason.APPLICATION_INACTIVE,
                    "The application left the foreground after the dispatched action.",
                )
            history.append(build_planning_history_entry(proposal, outcome, target_grounding))
            progress_terminal = grounding_correction_failure_reason(proposal, outcome) or progress_terminal
            if progress_terminal is not None:
                return conclude(AgentTerminalReason.CANNOT_PROGRESS, progress_terminal)

        reason = "The Device Agent used all available steps."
        return conclude(AgentTerminalReason.BUDGET_EXHAUSTED, reason)

    def _ground_target(
        self, proposal: ActionProposal, observation: Observation,
        correction: GroundingCorrection | None,
    ) -> GroundingEvidence | None:
        definition = ACTION_DEFINITIONS[proposal.action]
        return self._ground_expression(
            proposal.target_expression,
            definition.target_operation,
            observation,
            correction,
        )

    def _ground_destination(
        self, proposal: ActionProposal, observation: Observation,
    ) -> GroundingEvidence | None:
        definition = ACTION_DEFINITIONS[proposal.action]
        return self._ground_expression(
            proposal.destination_expression,
            definition.destination_operation,
            observation,
            None,
        )

    def _ground_expression(
        self, expression: str | None,
        operation: GroundingTargetOperation | None,
        observation: Observation, correction: GroundingCorrection | None,
    ) -> GroundingEvidence | None:
        if expression is None or operation is None:
            return None
        output_shape = (
            GroundingOutputShape.REGION
            if operation is GroundingTargetOperation.SCROLL_REGION
            else GroundingOutputShape.POINT
        )
        request = GroundingRequest(
            grounding_id=self._identifier_factory(),
            screenshot=observation.redacted_screenshot,
            expression=expression,
            operation=operation,
            output_shape=output_shape,
            excluded_regions=observation.system_regions,
            correction=correction,
        )
        return GroundingEvidence(request=request, result=self._grounder.ground(request))

    def _focus_reading(self, action: AuthorizedAction) -> FocusedFieldReading | None:
        if action.action not in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}:
            return None
        return self._executor.focused_field_reading()

    @staticmethod
    def _require_continuation(interruption: InterruptionSignal) -> None:
        if interruption.is_set():
            raise RunWorkflowInterrupted

__all__ = ["DeviceAgent"]
