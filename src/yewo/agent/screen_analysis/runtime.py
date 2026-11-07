"""Runtime accounting and identity resolution for current-screen analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionProposal,
    DispatchDisposition,
)
from yewo.agent.narration import narrate
from yewo.agent.planning.evidence import first_source_of
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.records import AgentModePolicy, invocation_budget_reason
from yewo.agent.screen_analysis.manager import CurrentScreenAnalysisManager
from yewo.agent.screen_analysis.records import CurrentScreenAnalysis, resolve_element_expression
from yewo.observations import Observation


@dataclass(frozen=True)
class ScreenAnalysisAttempt:
    """One optional analysis and its budget effect."""

    analysis: CurrentScreenAnalysis | None = None
    request_count: int = 0
    budget_reason: str | None = None


@dataclass(frozen=True)
class ScreenAnalysisBudget:
    """Current counters needed to reserve one analyzer request."""

    completed_steps: int
    planning_requests: int
    grounding_requests: int
    budget_time: float


@runtime_checkable
class ScreenAnalysisRequirementSource(Protocol):
    """Declare whether the active action source needs semantic perception."""

    def requires_screen_analysis(self, observation: Observation) -> bool: ...


@dataclass
class ScreenAnalysisRunState:
    """Own analysis reuse, budget accounting, and retained run evidence."""

    manager: CurrentScreenAnalysisManager | None
    recovery: RecoveryRunState
    policy: AgentModePolicy
    started_at: float
    requirement_source: object | None = None
    _retained: list[CurrentScreenAnalysis] = field(default_factory=list)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """Return persisted analysis identities referenced by the run result."""
        return tuple(item.analysis_id for item in self._retained)

    def analyze(
        self,
        observation: Observation,
        budget: ScreenAnalysisBudget,
    ) -> ScreenAnalysisAttempt:
        """Analyze one observation and retain its current result once."""
        if not self._is_required(observation):
            return ScreenAnalysisAttempt()
        attempt = _analyze_current_screen(
            self.manager,
            self.recovery,
            observation,
            self.policy,
            self.started_at,
            budget,
        )
        retain_screen_analysis(self._retained, attempt.analysis)
        return attempt

    def observe_outcome(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
    ) -> None:
        """Update structural reuse after one verified device outcome."""
        if self.manager is not None and self._is_required(after):
            self.manager.observe_outcome(proposal, before, after, outcome)

    def invalidate(self, observation: Observation, reason: str) -> None:
        """Invalidate one cached hypothesis after contradictory runtime evidence."""
        if self.manager is not None and self._is_required(observation):
            self.manager.invalidate(observation, reason)

    def observe_dispatch_rejection(
        self,
        disposition: DispatchDisposition,
        observation: Observation,
    ) -> None:
        """Invalidate perception only when the device rejected the target itself."""
        if disposition is DispatchDisposition.TARGET_REJECTED:
            self.invalidate(observation, "interaction_contradicted_screen_analysis")

    def _is_required(self, observation: Observation) -> bool:
        requirement = (
            first_source_of(self.requirement_source, ScreenAnalysisRequirementSource)
            if self.requirement_source is not None
            else None
        )
        return requirement is None or requirement.requires_screen_analysis(observation)


def _analyze_current_screen(
    manager: CurrentScreenAnalysisManager | None,
    recovery: RecoveryRunState,
    observation: Observation,
    policy: AgentModePolicy,
    started_at: float,
    budget: ScreenAnalysisBudget,
) -> ScreenAnalysisAttempt:
    """Return one cached/current analysis while accounting for its model budget."""
    if manager is None:
        return ScreenAnalysisAttempt()
    request_count = manager.required_model_requests(observation)
    request_reason = manager.request_reason(observation)
    recovery.settle(budget.completed_steps, budget.budget_time)
    recovery_active = recovery.is_active
    credit = recovery.credit(budget.completed_steps, budget.budget_time)
    budget_reason = invocation_budget_reason(
        policy,
        max(0.0, budget.budget_time - started_at - credit.duration_seconds),
        budget.planning_requests
        + budget.grounding_requests
        - credit.planning_requests
        - credit.analysis_requests
        - credit.grounding_requests,
        0 if recovery_active else request_count,
    )
    if budget_reason is not None:
        return ScreenAnalysisAttempt(budget_reason=budget_reason)
    analysis = manager.analyze(observation)
    capacity_reason = manager.capacity_failure_reason(observation)
    if recovery_active:
        recovery.credit_analysis_requests(request_count)
    if request_count:
        narrate(
            "Current-screen analysis "
            + ("completed" if analysis else "unavailable")
            + f"; reason={request_reason}."
        )
    return ScreenAnalysisAttempt(
        analysis=analysis,
        request_count=request_count,
        budget_reason=capacity_reason,
    )


def resolve_proposal_element_refs(
    proposal: ActionProposal,
    analysis: CurrentScreenAnalysis | None,
) -> ActionProposal:
    """Resolve planner identities to ephemeral TARS descriptions for this observation."""
    target = resolve_element_expression(analysis, proposal.target_element_ref)
    destination = resolve_element_expression(analysis, proposal.destination_element_ref)
    if target is None and destination is None:
        return proposal
    return proposal.model_copy(
        update={
            "target_expression": target or proposal.target_expression,
            "destination_expression": destination or proposal.destination_expression,
        }
    )


def resolve_element_refs_or_stop(
    proposal: ActionProposal,
    analysis: CurrentScreenAnalysis | None,
    observation_id: str,
    identifier_factory: Callable[[], str],
) -> tuple[ActionProposal, bool]:
    """Resolve analyzed identities or replace an invalid proposal with a safe terminal."""
    try:
        return resolve_proposal_element_refs(proposal, analysis), False
    except ValueError as error:
        return (
            ActionProposal(
                proposal_id=identifier_factory(),
                source_observation_id=observation_id,
                action=ActionKind.CANNOT_PROGRESS,
                reason=f"Planning stopped before dispatch: {error}",
            ),
            True,
        )


def retain_screen_analysis(
    retained: list[CurrentScreenAnalysis],
    analysis: CurrentScreenAnalysis | None,
) -> None:
    """Retain one analysis per observation without duplicating reused variants."""
    if analysis is not None and all(
        item.observation_id != analysis.observation_id for item in retained
    ):
        retained.append(analysis)


def start_screen_analysis(
    manager: CurrentScreenAnalysisManager | None,
    recovery: RecoveryRunState,
    policy: AgentModePolicy,
    started_at: float,
    requirement_source: object | None = None,
) -> ScreenAnalysisRunState:
    """Create run-scoped screen-analysis orchestration."""
    return ScreenAnalysisRunState(manager, recovery, policy, started_at, requirement_source)
