"""Run one bounded observe-plan-act loop across workflow task briefs."""

from __future__ import annotations
from collections.abc import Callable
import time
from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionExecutor,
    ActionOutcomeVerifier,
    ActionPolicy,
    ActionRefusal,
    ActionSource,
    DispatchDisposition,
)
from yewo.agent.actuation.verifier import ActionVerifier
from yewo.agent.runtime.application_state import ApplicationRecovery, observation_terminal
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.runtime.no_progress import (
    ActionNoProgressTracker,
    grounding_correction_for_history,
    proposal_refusal as _proposal_refusal,
    record_outcome_with_facts as _record_outcome,
)
import yewo.agent.narration as runtime_narration
from yewo.agent.planning.planner import (
    ActionProposalSource,
    PlanningHistoryEntry,
    PlanningRequest,
    TaskProgressContext,
    plan_or_stop,
)
from yewo.agent.planning.history import grounding_correction_failure_reason
from yewo.agent.runtime.refusals import record_action_refusal, record_dispatch_rejection
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentStepStore,
    AgentTask,
    AgentTerminalReason,
    available_runtime_actions,
    build_runtime_grounding_request,
    grounding_budget_reason,
    invocation_budget_reason,
    proposal_terminal,
)
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.agent.runtime import control_evidence, dispatch_timing, result as rr
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.recovery.advisor import RecoveryAdvisor, RecoveryDisposition
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.verification import assess_recovered_failure, verify_discovery_action
import yewo.agent.runtime.settlement as runtime_settlement
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingEvidence, VisualGrounder, resolve_grounding_request
from yewo.observations import ObservationService
from yewo.runs import InterruptionSignal


class DeviceAgent:
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
        application_recovery: ApplicationRecovery,
        screen_comparison: ScreenComparisonPolicy | None = None,
        planner_decision_source: PlannerDecisionSource = PlannerDecisionSource.MODEL,
        reused_flow_id: str | None = None,
        progress_context: TaskProgressContext | None = None,
        recovery_advisor: RecoveryAdvisor | None = None,
        action_verifier: ActionVerifier | None = None,
        initial_planning_requests: int = 0,
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
        self._planner_decision_source = planner_decision_source
        self._reused_flow_id = reused_flow_id
        self._progress_context = progress_context
        self._recovery = RecoveryRunState(recovery_advisor, self._screen_comparison)
        self._action_verifier = action_verifier
        self._initial_planning_requests = initial_planning_requests
        self._progress_ledger = RunProgressLedger(step_store.root, identifier_factory=identifier_factory)

    def run(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
    ) -> AgentResult:
        started_at = self._monotonic()
        runtime_narration.narrate(f"Starting task: {task.objective}")
        runtime_settlement.require_agent_continuation(interruption)
        current_observation = self._observations.capture()
        runtime_narration.narrate_observation(current_observation, label="Initial")
        initial_terminal = observation_terminal(current_observation)
        current_observation, initial_terminal, unused_recovery_id = (
            self._application_recovery.apply(current_observation, initial_terminal, interruption)
        )
        planning_requests = self._initial_planning_requests
        if initial_terminal is not None:
            return rr.build_initial_agent_result(
                initial_terminal,
                runtime_settlement.application_terminal_reason(initial_terminal, initial=True),
                current_observation,
                planning_requests,
            )
        self._progress_ledger.start(task.open_work, current_observation, self._progress_context)
        steps: list[AgentStep] = []
        visible_controls: list[control_evidence.VisibleControlObservation] = []
        history: list[PlanningHistoryEntry] = []
        no_progress = ActionNoProgressTracker(self._screen_comparison)
        no_progress.start(current_observation)
        grounding_requests = 0
        dispatched_actions = 0
        action_policy = ActionPolicy(mode_policy.action_limits, self._identifier_factory)
        def conclude(terminal: AgentTerminalReason, reason: str) -> AgentResult:
            runtime_narration.narrate(f"Terminal {terminal.value}: {reason}")
            credit_time = self._monotonic() if self._recovery.assessment else started_at
            self._recovery.settle(len(steps), credit_time)
            recovery_credit = self._recovery.credit(len(steps), credit_time)
            runtime_narration.narrate_recovery_credit(recovery_credit)
            return rr.build_agent_result(
                terminal,
                reason,
                current_observation,
                planning_requests,
                grounding_requests,
                dispatched_actions,
                steps,
                recovery_credit,
                tuple(visible_controls),
            )
        while len(steps) - self._recovery.credited_steps(len(steps)) < mode_policy.maximum_steps:
            step_index = len(steps) + 1
            runtime_settlement.require_agent_continuation(interruption)
            budget_time = self._monotonic()
            available_actions = available_runtime_actions(
                mode_policy.action_limits,
                current_observation,
                targeted_type_available=task.targeted_type_available,
            )
            progress_context = self._progress_ledger.progress_context(self._recovery.progress_context(self._progress_context) or self._progress_context)
            planning_request = PlanningRequest(
                observation=current_observation,
                objective=task.objective,
                known_facts=task.known_facts,
                screen_facts=no_progress.screen_facts(),
                open_work=progress_context.remaining_work,
                completion_conditions=task.completion_conditions,
                declared_value_names=task.declared_value_names,
                history=tuple(history),
                available_actions=available_actions,
                progress_context=progress_context,
                discovery_mode=task.discovery_mode,
            )
            requested_planning = self._planner.required_model_requests(planning_request)
            self._recovery.settle(len(steps), budget_time)
            recovery_was_active = self._recovery.is_active
            credit = self._recovery.credit(len(steps), budget_time)
            budget_reason = invocation_budget_reason(
                mode_policy,
                max(
                    0.0,
                    budget_time - started_at - credit.duration_seconds,
                ),
                planning_requests
                + grounding_requests
                - credit.planning_requests
                - credit.grounding_requests,
                0 if recovery_was_active else requested_planning,
            )
            if budget_reason is not None:
                return conclude(AgentTerminalReason.BUDGET_EXHAUSTED, budget_reason)
            proposal, planning_stopped = plan_or_stop(
                self._planner, planning_request, self._identifier_factory
            )
            planning_requests += rr.record_planning_model_usage(
                self._recovery, self._planner, requested_planning
            )
            proposal, decision = runtime_narration.narrate_selected_proposal(
                self._planner,
                step_index,
                proposal,
                self._planner_decision_source,
                self._reused_flow_id,
                self._identifier_factory,
                system_generated=planning_stopped,
            )
            terminal = proposal_terminal(proposal)
            pre_dispatch = verify_discovery_action(
                self._action_verifier,
                task.discovery_mode,
                decision.source,
                planning_request,
                proposal,
                mode_policy,
                self._recovery,
                len(steps),
                started_at,
                planning_requests,
                grounding_requests,
                self._monotonic(),
            )
            control_evidence.retain_visible_controls(
                visible_controls,
                current_observation.observation_id,
                proposal.proposal_id,
                pre_dispatch.visible_controls,
            )
            self._progress_ledger.record_verification(pre_dispatch, current_observation, proposal)
            if terminal is not None or pre_dispatch.budget_reason is not None:
                terminal = terminal or AgentTerminalReason.BUDGET_EXHAUSTED
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    terminal,
                )
                steps.append(step)
                return conclude(
                    terminal,
                    pre_dispatch.budget_reason
                    or proposal.reason
                    or "The planner requested a terminal.",
                )
            planning_requests += pre_dispatch.request_count
            credit = self._recovery.credit(len(steps), budget_time)
            if (
                dispatched_actions - credit.dispatched_actions
                >= mode_policy.action_limits.maximum_actions
            ):
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
                steps.append(step)
                return conclude(
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                    "No device-action budget remains.",
                )
            refusal = pre_dispatch.refusal or _proposal_refusal(
                no_progress,
                action_policy,
                proposal,
                current_observation,
                ActionBudget(
                    dispatched_actions=(
                        dispatched_actions
                        - self._recovery.credit(len(steps), budget_time).dispatched_actions
                    )
                ),
            )
            if refusal is not None:
                reason = record_action_refusal(
                    self._step_store,
                    steps,
                    history,
                    no_progress,
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    refusal,
                )
                if reason:
                    return conclude(AgentTerminalReason.CANNOT_PROGRESS, reason)
                continue
            correction = grounding_correction_for_history(proposal, history)
            target_request = build_runtime_grounding_request(
                self._identifier_factory(),
                proposal.target_expression,
                ACTION_DEFINITIONS[proposal.action].target_operation,
                current_observation,
                correction,
            )
            destination_request = build_runtime_grounding_request(
                self._identifier_factory(),
                proposal.destination_expression,
                ACTION_DEFINITIONS[proposal.action].destination_operation,
                current_observation,
                None,
            )
            target_grounding: GroundingEvidence | None = None
            destination_grounding: GroundingEvidence | None = None
            try:
                grounding_count = sum(
                    self._grounder.required_model_requests(request)
                    for request in (target_request, destination_request)
                    if request is not None
                )
                grounding_was_recovery = self._recovery.is_active
                self._recovery.credit_grounding_requests(grounding_count)
                credit = self._recovery.credit(len(steps), budget_time)
                grounding_budget = grounding_budget_reason(
                    mode_policy,
                    planning_requests - credit.planning_requests,
                    grounding_requests - credit.grounding_requests,
                    0 if grounding_was_recovery else grounding_count,
                )
                if grounding_budget is not None:
                    step = self._step_store.record(
                        step_index,
                        current_observation,
                        proposal,
                        decision,
                        AgentTerminalReason.BUDGET_EXHAUSTED,
                    )
                    steps.append(step)
                    return conclude(
                        AgentTerminalReason.BUDGET_EXHAUSTED,
                        grounding_budget,
                    )
                target_grounding = resolve_grounding_request(self._grounder, target_request)
                destination_grounding = resolve_grounding_request(
                    self._grounder, destination_request
                )
            except BaseException as error:
                runtime_narration.narrate(
                    f"Step {step_index} grounding failed: {type(error).__name__}."
                )
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    decision,
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
            runtime_narration.narrate_grounding(step_index, "target", target_grounding)
            runtime_narration.narrate_grounding(step_index, "destination", destination_grounding)
            authorization = action_policy.authorize(
                proposal,
                current_observation,
                ActionBudget(
                    dispatched_actions=(
                        dispatched_actions
                        - self._recovery.credit(len(steps), budget_time).dispatched_actions
                    ),
                ),
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                source=ActionSource.PLANNER,
            )
            if isinstance(authorization, ActionRefusal):
                record_action_refusal(
                    self._step_store,
                    steps,
                    history,
                    no_progress,
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    authorization,
                    target_grounding,
                    destination_grounding,
                )
                continue
            runtime_narration.trace_authorization(step_index, authorization)
            try:
                runtime_narration.narrate(f"Step {step_index} dispatching {proposal.action.value}.")
                timed_dispatch = dispatch_timing.dispatch_authorized_action(
                    self._executor, authorization, step_index, self._monotonic
                )
                before_focus, receipt = timed_dispatch.before_focus, timed_dispatch.receipt
            except BaseException as error:
                runtime_narration.narrate(
                    f"Step {step_index} dispatch failed: {type(error).__name__}."
                )
                self._step_store.record_partial_or_raise(
                    step_index,
                    current_observation,
                    proposal,
                    decision,
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
                    decision,
                    AgentTerminalReason.DEVICE_REFUSED,
                    target_grounding=target_grounding,
                    destination_grounding=destination_grounding,
                    authorization=authorization,
                    receipt=receipt,
                )
                steps.append(step)
                runtime_narration.narrate(
                    f"Step {step_index} device refusal: "
                    f"{receipt.reason or 'The device refused the action.'}"
                )
                return conclude(
                    AgentTerminalReason.DEVICE_REFUSED,
                    receipt.reason or "The device refused the authorized action.",
                )
            if receipt.disposition in {
                DispatchDisposition.INPUT_REJECTED,
                DispatchDisposition.TARGET_REJECTED,
            }:
                repeated_rejection = record_dispatch_rejection(
                    self._step_store,
                    steps,
                    history,
                    no_progress,
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    target_grounding,
                    destination_grounding,
                    authorization,
                    receipt,
                )
                if repeated_rejection is not None:
                    return conclude(AgentTerminalReason.CANNOT_PROGRESS, repeated_rejection)
                continue
            dispatched_actions += 1
            self._recovery.credit_dispatched_action()
            dispatch_timing.trace_settlement_start(step_index, timed_dispatch, self._monotonic())
            settled = runtime_settlement.settle_dispatched_action(
                self._observations,
                self._executor,
                self._outcome_verifier,
                self._application_recovery,
                self._step_store,
                step_index,
                current_observation,
                proposal,
                decision,
                target_grounding,
                destination_grounding,
                authorization,
                receipt,
                mode_policy,
                self._planner,
                interruption,
                self._sleep,
                steps,
                before_focus,
            )
            after = settled.observation
            outcome = settled.outcome
            application_failure = settled.application_failure

            step = runtime_settlement.record_settled_action(
                self._step_store,
                step_index,
                current_observation,
                proposal,
                decision,
                target_grounding,
                destination_grounding,
                authorization,
                receipt,
                settled,
            )
            steps.append(step)
            if outcome is None:
                runtime_narration.narrate_application_failure(step_index, application_failure)
                reason = runtime_settlement.unsettled_application_failure_reason(settled)
                return conclude(AgentTerminalReason.APPLICATION_EXITED, reason)
            runtime_narration.narrate_observation(after, label=f"Step {step_index}")
            runtime_narration.narrate_outcome(step_index, outcome)
            runtime_narration.narrate_recovery(step_index, settled.recovery_observation_id)
            runtime_narration.narrate_application_failure(step_index, application_failure)
            source_observation = current_observation
            progress_terminal = _record_outcome(
                no_progress,
                proposal,
                current_observation,
                after,
                outcome,
                runtime_narration.narrate_no_progress,
            )
            self._progress_ledger.record_step(step, pre_dispatch, current_observation, after)
            current_observation = after
            self._recovery.observe_if_active(after, len(steps), self._monotonic)
            terminal_conclusion = runtime_settlement.application_terminal_conclusion(settled)
            if terminal_conclusion is not None:
                return conclude(*terminal_conclusion)
            history.append(
                settled.planning_history_entry(proposal, target_grounding, source_observation)
            )
            assessment = assess_recovered_failure(
                self._recovery,
                application_failure,
                planning_request,
                step,
                current_observation,
                len(steps),
                self._monotonic,
            )
            if assessment is not None:
                planning_requests += 1
                runtime_narration.narrate_recovery_assessment(assessment)
                if assessment.disposition is RecoveryDisposition.STOP:
                    return conclude(AgentTerminalReason.APPLICATION_EXITED, assessment.reason)
            progress_terminal = (
                grounding_correction_failure_reason(proposal, outcome) or progress_terminal
            )
            if progress_terminal is not None:
                return conclude(AgentTerminalReason.CANNOT_PROGRESS, progress_terminal)
        return conclude(
            AgentTerminalReason.BUDGET_EXHAUSTED, "The Device Agent used all available steps."
        )


__all__ = ["DeviceAgent"]
