"""Run one bounded observe-plan-act loop across workflow task briefs."""

from __future__ import annotations

from collections.abc import Callable
import time

import yewo.agent.narration as runtime_narration
import yewo.agent.runtime.settlement as runtime_settlement
from yewo.agent.actuation import (
    ActionBudget,
    ActionExecutor,
    ActionOutcomeVerifier,
    ActionPolicy,
    ActionRefusal,
    ActionSource,
    DispatchDisposition,
)
from yewo.agent.actuation.verifier import ActionVerifier
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.planning.planner import ActionProposalSource, PlanningHistoryEntry, plan_or_stop
from yewo.agent.planning.records import TaskProgressContext
from yewo.agent.recovery.advisor import RecoveryAdvisor
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime import agent_startup, dispatch_timing, grounding, result as rr
from yewo.agent.runtime.application_state import ApplicationRecovery
from yewo.agent.runtime.pre_dispatch import pre_dispatch_terminal, prepare_action_for_grounding
from yewo.agent.runtime.refusals import record_action_refusal, record_dispatch_rejection
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentStepStore,
    AgentTask,
    AgentTerminalReason,
    grounding_budget_reason,
    invocation_budget_reason,
)
from yewo.agent.runtime.progress import RunProgressLedger, resolve_progress_terminal
from yewo.agent.runtime.planning import build_current_planning_view
from yewo.agent.runtime.verification import recovery_stop_reason
from yewo.agent.screen_analysis import CurrentScreenAnalysisManager
from yewo.agent.screen_analysis import runtime as screen_runtime
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingEvidence, VisualGrounder
from yewo.observations import ObservationService
from yewo.runs import InterruptionSignal


__all__ = ["DeviceAgent"]


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
        screen_analysis: CurrentScreenAnalysisManager | None = None,
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
        self._screen_analysis = screen_analysis
        self._initial_planning_requests = initial_planning_requests
        self._progress_ledger = RunProgressLedger(
            step_store.root,
            screen_comparison=self._screen_comparison,
            identifier_factory=identifier_factory,
        )

    def run(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
    ) -> AgentResult:
        started_at = self._monotonic()
        runtime_narration.narrate(f"Starting task: {task.objective}")
        current_observation, initial_terminal = agent_startup.capture_initial_application_state(
            self._observations,
            self._application_recovery,
            interruption,
        )
        planning_requests = self._initial_planning_requests
        if initial_terminal is not None:
            return agent_startup.build_initial_result(
                initial_terminal, current_observation, planning_requests
            )
        self._progress_ledger.start(task.open_work, current_observation, self._progress_context)
        steps: list[AgentStep] = []
        screen_analysis = screen_runtime.start_screen_analysis(
            self._screen_analysis,
            self._recovery,
            mode_policy,
            started_at,
            self._planner,
        )
        history: list[PlanningHistoryEntry] = []
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
                screen_analysis_ids=screen_analysis.record_ids,
            )

        while len(steps) - self._recovery.credited_steps(len(steps)) < mode_policy.maximum_steps:
            step_index = len(steps) + 1
            runtime_settlement.require_agent_continuation(interruption)
            budget_time = self._monotonic()
            analysis_attempt = screen_analysis.analyze(
                current_observation,
                screen_runtime.ScreenAnalysisBudget(
                    completed_steps=len(steps),
                    planning_requests=planning_requests,
                    grounding_requests=grounding_requests,
                    budget_time=budget_time,
                ),
            )
            planning_requests += analysis_attempt.request_count
            current_screen_analysis = analysis_attempt.analysis
            planning_view = build_current_planning_view(
                self._planner,
                self._progress_ledger,
                self._recovery,
                self._progress_context,
                task,
                mode_policy,
                current_observation,
                current_screen_analysis,
                tuple(history),
            )
            planning_request = planning_view.request
            pending_focus = planning_view.pending_focus
            requested_planning = self._planner.required_model_requests(planning_request)
            self._recovery.settle(len(steps), budget_time)
            recovery_was_active = self._recovery.is_active
            credit = self._recovery.credit(len(steps), budget_time)
            budget_reason = analysis_attempt.budget_reason or invocation_budget_reason(
                mode_policy,
                max(0.0, budget_time - started_at - credit.duration_seconds),
                planning_requests
                + grounding_requests
                - credit.planning_requests
                - credit.analysis_requests
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
            proposal, element_resolution_stopped = screen_runtime.resolve_element_refs_or_stop(
                proposal,
                current_screen_analysis,
                current_observation.observation_id,
                self._identifier_factory,
            )
            planning_stopped = planning_stopped or element_resolution_stopped
            proposal, decision = runtime_narration.narrate_selected_proposal(
                self._planner,
                step_index,
                proposal,
                self._planner_decision_source,
                self._reused_flow_id,
                self._identifier_factory,
                system_generated=planning_stopped,
            )
            committed_focus = self._progress_ledger.commit_planning_focus(
                current_observation, pending_focus, decision.source
            )
            runtime_narration.narrate_planning_focus(committed_focus)
            credit = self._recovery.credit(len(steps), budget_time)
            terminal = pre_dispatch_terminal(
                proposal,
                dispatched_actions,
                credit.dispatched_actions,
                mode_policy.action_limits.maximum_actions,
            )
            if terminal is not None:
                step = self._step_store.record(
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    terminal.terminal_reason,
                )
                steps.append(step)
                return conclude(terminal.terminal_reason, terminal.reason)
            action_budget = ActionBudget(
                dispatched_actions=dispatched_actions - credit.dispatched_actions
            )
            prepared = prepare_action_for_grounding(
                self._progress_ledger,
                action_policy,
                action_budget,
                self._action_verifier,
                task.discovery_mode,
                decision.source,
                self._planner,
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
            proposal = prepared.proposal
            refusal = prepared.refusal
            system_declared_correction = prepared.system_declared_correction
            if prepared.budget_reason is not None:
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
                    prepared.budget_reason,
                )
            planning_requests += prepared.verification_request_count
            if refusal is not None:
                reason = record_action_refusal(
                    self._step_store,
                    steps,
                    history,
                    self._progress_ledger,
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    refusal,
                    planning_requests=planning_requests,
                    grounding_requests=grounding_requests,
                    dispatched_actions=dispatched_actions,
                )
                if reason:
                    return conclude(AgentTerminalReason.CANNOT_PROGRESS, reason)
                continue
            target_request, destination_request = grounding.build_proposal_grounding_requests(
                self._identifier_factory,
                proposal,
                current_observation,
                history,
                current_screen_analysis,
            )
            target_grounding: GroundingEvidence | None = None
            destination_grounding: GroundingEvidence | None = None
            try:
                grounding_reservation = sum(
                    self._grounder.required_model_requests(request)
                    for request in (target_request, destination_request)
                    if request is not None
                )
                grounding_was_recovery = self._recovery.is_active
                credit = self._recovery.credit(len(steps), budget_time)
                grounding_budget = grounding_budget_reason(
                    mode_policy,
                    planning_requests - credit.planning_requests - credit.analysis_requests,
                    grounding_requests - credit.grounding_requests,
                    0 if grounding_was_recovery else grounding_reservation,
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
                target_grounding, destination_grounding = grounding.resolve_proposal_grounding(
                    self._grounder,
                    target_request,
                    destination_request,
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
            grounding_count = grounding.consumed_model_requests(
                target_grounding, destination_grounding
            )
            self._recovery.credit_grounding_requests(grounding_count)
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
                reason = record_action_refusal(
                    self._step_store,
                    steps,
                    history,
                    self._progress_ledger,
                    step_index,
                    current_observation,
                    proposal,
                    decision,
                    authorization,
                    target_grounding,
                    destination_grounding,
                    planning_requests=planning_requests,
                    grounding_requests=grounding_requests,
                    dispatched_actions=dispatched_actions,
                )
                if reason:
                    return conclude(AgentTerminalReason.CANNOT_PROGRESS, reason)
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
                screen_analysis.observe_dispatch_rejection(receipt.disposition, current_observation)
                repeated_rejection = record_dispatch_rejection(
                    self._step_store,
                    steps,
                    history,
                    self._progress_ledger,
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
                screen_analysis.invalidate(current_observation, "application_failure")
                runtime_narration.narrate_application_failure(step_index, application_failure)
                reason = runtime_settlement.unsettled_application_failure_reason(settled)
                return conclude(AgentTerminalReason.APPLICATION_EXITED, reason)
            runtime_narration.narrate_dispatched_step(
                step_index, after, outcome, settled.recovery_observation_id, application_failure
            )
            source_observation = current_observation
            screen_analysis.observe_outcome(proposal, current_observation, after, outcome)
            new_progress_facts, progress_terminal = self._progress_ledger.record_step(
                step,
                prepared.verification,
                current_observation,
                after,
                current_screen_analysis,
            )
            runtime_narration.narrate_no_progress_facts(new_progress_facts)
            current_observation = after
            self._recovery.observe_if_active(after, len(steps), self._monotonic)
            terminal_conclusion = runtime_settlement.application_terminal_conclusion(settled)
            if terminal_conclusion is not None:
                return conclude(*terminal_conclusion)
            history.append(
                settled.planning_history_entry(proposal, target_grounding, source_observation)
            )
            stop_reason, assessment_requests = recovery_stop_reason(
                self._recovery,
                application_failure,
                planning_request,
                step,
                current_observation,
                len(steps),
                self._monotonic,
            )
            planning_requests += assessment_requests
            if stop_reason is not None:
                return conclude(AgentTerminalReason.APPLICATION_EXITED, stop_reason)
            progress_terminal = resolve_progress_terminal(
                proposal, outcome, system_declared_correction, progress_terminal
            )
            if progress_terminal is not None:
                return conclude(AgentTerminalReason.CANNOT_PROGRESS, progress_terminal)
        return conclude(
            AgentTerminalReason.BUDGET_EXHAUSTED, "The Device Agent used all available steps."
        )
