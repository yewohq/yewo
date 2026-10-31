"""Run one bounded observe-plan-act loop across workflow task briefs."""

from __future__ import annotations

from collections.abc import Callable
import time

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionExecutor,
    ActionHandling,
    ActionKind,
    ActionOutcome,
    ActionOutcomeVerifier,
    ActionPolicy,
    ActionPolicyLimits,
    ActionProposal,
    ActionRefusal,
    ActionSource,
    AuthorizedAction,
    DispatchDisposition,
    DispatchReceipt,
)
from yewo.agent.application_state import application_terminal_reason, observation_terminal
from yewo.agent.planner import PlanningHistoryEntry, PlanningRequest, TaskPlanner
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentStepPersistenceError,
    AgentStepStore,
    AgentTask,
    AgentTerminalReason,
    AgentUsage,
)
from yewo.agent.settlement import SettlementInterrupted, capture_settled_observation
from yewo.artifacts import new_record_identifier
from yewo.devices.focus import FocusedFieldReading
from yewo.grounding import (
    GroundingEvidence,
    GroundingOutputShape,
    GroundingRequest,
    GroundingTargetOperation,
    VisualGrounder,
)
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowEvidenceError, RunWorkflowInterrupted


class DeviceAgent:
    """Coordinate the shared bounded mobile-operation loop."""

    def __init__(
        self,
        planner: TaskPlanner,
        grounder: VisualGrounder,
        observations: ObservationService,
        executor: ActionExecutor,
        outcome_verifier: ActionOutcomeVerifier,
        step_store: AgentStepStore,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._planner = planner
        self._grounder = grounder
        self._observations = observations
        self._executor = executor
        self._outcome_verifier = outcome_verifier
        self._step_store = step_store
        self._identifier_factory = identifier_factory
        self._sleep = sleep

    def run(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
    ) -> AgentResult:
        """Operate until one bounded runtime terminal is reached."""
        self._require_continuation(interruption)
        current_observation = self._observations.capture()
        initial_terminal = observation_terminal(current_observation)
        if initial_terminal is not None:
            return _result(
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
        planning_requests = 0
        grounding_requests = 0
        dispatched_actions = 0
        available_actions = _available_actions(mode_policy.action_limits)
        action_policy = ActionPolicy(mode_policy.action_limits, self._identifier_factory)

        for step_index in range(1, mode_policy.maximum_steps + 1):
            self._require_continuation(interruption)
            planning_requests += 1
            proposal = self._planner.plan(
                PlanningRequest(
                    observation=current_observation,
                    objective=task.objective,
                    known_facts=task.known_facts,
                    open_work=task.open_work,
                    completion_conditions=task.completion_conditions,
                    declared_value_names=task.declared_value_names,
                    history=tuple(history),
                    available_actions=available_actions,
                )
            )
            terminal = _proposal_terminal(proposal)
            if terminal is not None:
                step = self._record_step(step_index, current_observation, proposal, terminal)
                steps.append(step)
                return _result(
                    terminal,
                    proposal.reason or "The planner requested a terminal.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )
            if dispatched_actions >= mode_policy.action_limits.maximum_actions:
                step = self._record_step(
                    step_index,
                    current_observation,
                    proposal,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
                steps.append(step)
                return _result(
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                    "The device-action budget is exhausted.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )

            grounding_count = _required_grounding_count(proposal)
            if grounding_requests + grounding_count > mode_policy.maximum_grounding_requests:
                step = self._record_step(
                    step_index,
                    current_observation,
                    proposal,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
                steps.append(step)
                return _result(
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                    "The visual-grounding request budget is exhausted.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )

            target_grounding: GroundingEvidence | None = None
            destination_grounding: GroundingEvidence | None = None
            try:
                target_grounding = self._ground_target(proposal, current_observation)
                destination_grounding = self._ground_destination(proposal, current_observation)
            except BaseException as error:
                self._record_partial_or_raise(
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
                step = self._record_step(
                    step_index,
                    current_observation,
                    proposal,
                    target_grounding=target_grounding,
                    destination_grounding=destination_grounding,
                    refusal=authorization,
                )
                steps.append(step)
                history.append(PlanningHistoryEntry(proposal=proposal, refusal=authorization))
                continue

            try:
                before_focus = self._focus_reading(authorization)
                receipt = self._executor.dispatch(authorization)
            except BaseException as error:
                self._record_partial_or_raise(
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
                step = self._record_step(
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
                return _result(
                    AgentTerminalReason.DEVICE_REFUSED,
                    receipt.reason or "The device refused the authorized action.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )
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
            except RunWorkflowInterrupted as error:
                self._executor.discard_text_delivery(authorization)
                settlement_ids = (
                    error.observation_ids if isinstance(error, SettlementInterrupted) else ()
                )
                self._record_partial_or_raise(
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
                self._executor.discard_text_delivery(authorization)
                self._record_partial_or_raise(
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

            step_terminal = observation_terminal(after)
            step = self._record_step(
                step_index,
                current_observation,
                proposal,
                step_terminal,
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                authorization=authorization,
                receipt=receipt,
                settlement_observation_ids=settlement_observation_ids,
                settled_observation_id=(after.observation_id if settlement.settled else None),
                outcome=outcome,
            )
            steps.append(step)
            current_observation = after
            if step_terminal is AgentTerminalReason.APPLICATION_EXITED:
                return _result(
                    AgentTerminalReason.APPLICATION_EXITED,
                    "The application exited after the dispatched action.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )
            if step_terminal is AgentTerminalReason.APPLICATION_INACTIVE:
                return _result(
                    AgentTerminalReason.APPLICATION_INACTIVE,
                    "The application left the foreground after the dispatched action.",
                    current_observation,
                    planning_requests,
                    grounding_requests,
                    dispatched_actions,
                    steps,
                )
            history.append(PlanningHistoryEntry(proposal=proposal, outcome=outcome))

        return _result(
            AgentTerminalReason.BUDGET_EXHAUSTED,
            "The Device Agent step budget is exhausted.",
            current_observation,
            planning_requests,
            grounding_requests,
            dispatched_actions,
            steps,
        )

    def _ground_target(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> GroundingEvidence | None:
        definition = ACTION_DEFINITIONS[proposal.action]
        return self._ground_expression(
            proposal.target_expression,
            definition.target_operation,
            observation,
        )

    def _ground_destination(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> GroundingEvidence | None:
        definition = ACTION_DEFINITIONS[proposal.action]
        return self._ground_expression(
            proposal.destination_expression,
            definition.destination_operation,
            observation,
        )

    def _ground_expression(
        self,
        expression: str | None,
        operation: GroundingTargetOperation | None,
        observation: Observation,
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
        )
        return GroundingEvidence(request=request, result=self._grounder.ground(request))

    def _focus_reading(self, action: AuthorizedAction) -> FocusedFieldReading | None:
        if action.action not in {ActionKind.TYPE, ActionKind.CLEAR}:
            return None
        return self._executor.focused_field_reading()

    def _record_partial_or_raise(
        self,
        step_index: int,
        observation: Observation,
        proposal: ActionProposal,
        target_grounding: GroundingEvidence | None,
        destination_grounding: GroundingEvidence | None,
        authorization: AuthorizedAction | None,
        receipt: DispatchReceipt | None,
        terminal: AgentTerminalReason,
        steps: list[AgentStep],
        original_error: BaseException,
        *,
        settlement_observation_ids: tuple[str, ...] = (),
    ) -> None:
        try:
            step = self._record_step(
                step_index,
                observation,
                proposal,
                terminal,
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                authorization=authorization,
                receipt=receipt,
                settlement_observation_ids=settlement_observation_ids,
            )
        except AgentStepPersistenceError as persistence_error:
            raise RunWorkflowEvidenceError(
                interrupted=isinstance(original_error, RunWorkflowInterrupted),
                original_error=original_error,
            ) from persistence_error
        steps.append(step)

    def _record_step(
        self,
        step_index: int,
        observation: Observation,
        proposal: ActionProposal,
        terminal_reason: AgentTerminalReason | None = None,
        *,
        target_grounding: GroundingEvidence | None = None,
        destination_grounding: GroundingEvidence | None = None,
        authorization: AuthorizedAction | None = None,
        refusal: ActionRefusal | None = None,
        receipt: DispatchReceipt | None = None,
        settlement_observation_ids: tuple[str, ...] = (),
        settled_observation_id: str | None = None,
        outcome: ActionOutcome | None = None,
    ) -> AgentStep:
        step = AgentStep(
            step_id=self._identifier_factory(),
            step_index=step_index,
            source_observation_id=observation.observation_id,
            proposal=proposal,
            target_grounding=target_grounding,
            destination_grounding=destination_grounding,
            authorization=authorization,
            refusal=refusal,
            receipt=receipt,
            settlement_observation_ids=settlement_observation_ids,
            settled_observation_id=settled_observation_id,
            outcome=outcome,
            terminal_reason=terminal_reason,
        )
        self._step_store.persist(step)
        return step

    @staticmethod
    def _require_continuation(interruption: InterruptionSignal) -> None:
        if interruption.is_set():
            raise RunWorkflowInterrupted


def _available_actions(limits: ActionPolicyLimits) -> frozenset[ActionKind]:
    return frozenset(
        action
        for action in limits.allowed_actions & limits.supported_actions
        if ACTION_DEFINITIONS[action].handling is ActionHandling.EXECUTE
        and ACTION_DEFINITIONS[action].side_effect in limits.allowed_side_effects
        and (action is not ActionKind.BACK or limits.allow_exploratory_back)
    )


def _proposal_terminal(proposal: ActionProposal) -> AgentTerminalReason | None:
    if proposal.action is ActionKind.DONE:
        return AgentTerminalReason.DONE_REQUESTED
    if proposal.action is ActionKind.CANNOT_PROGRESS:
        return AgentTerminalReason.CANNOT_PROGRESS
    return None


def _required_grounding_count(proposal: ActionProposal) -> int:
    return int(proposal.target_expression is not None) + int(
        proposal.destination_expression is not None
    )


def _result(
    terminal: AgentTerminalReason,
    reason: str,
    observation: Observation,
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
    steps: list[AgentStep],
) -> AgentResult:
    return AgentResult(
        terminal_reason=terminal,
        reason=reason,
        final_observation_id=observation.observation_id,
        usage=AgentUsage(
            planning_requests=planning_requests,
            grounding_requests=grounding_requests,
            dispatched_actions=dispatched_actions,
        ),
        steps=tuple(steps),
    )
