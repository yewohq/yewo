"""Assemble the planner's current-step view from runtime state."""

from __future__ import annotations

from dataclasses import dataclass

from yewo.agent.planning.planner import (
    ActionProposalSource,
    PlanningHistoryEntry,
)
from yewo.agent.planning.records import (
    PlanningFocusContext,
    PlanningRequest,
    TaskProgressContext,
)
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.agent.runtime.records import AgentModePolicy, AgentTask, available_runtime_actions
from yewo.agent.screen_analysis import CurrentScreenAnalysis
from yewo.observations import Observation


@dataclass(frozen=True)
class CurrentPlanningView:
    """One planner request and the focus claim it may consume."""

    request: PlanningRequest
    pending_focus: PlanningFocusContext | None


def build_current_planning_view(
    planner: ActionProposalSource,
    progress_ledger: RunProgressLedger,
    recovery: RecoveryRunState,
    base_progress_context: TaskProgressContext | None,
    task: AgentTask,
    mode_policy: AgentModePolicy,
    observation: Observation,
    analysis: CurrentScreenAnalysis | None,
    history: tuple[PlanningHistoryEntry, ...],
) -> CurrentPlanningView:
    """Build the authoritative, non-duplicated context for one planning call."""
    pending_focus = progress_ledger.pending_planning_focus(observation, planner)
    progress_context = progress_ledger.progress_context(
        recovery.progress_context(base_progress_context) or base_progress_context
    )
    request = PlanningRequest(
        observation=observation,
        current_screen_analysis=analysis,
        objective=task.objective,
        known_facts=task.known_facts,
        screen_facts=progress_ledger.screen_facts(observation, history),
        open_work=progress_ledger.remaining_work(),
        completion_conditions=task.completion_conditions,
        declared_value_names=task.declared_value_names,
        history=history,
        available_actions=available_runtime_actions(
            mode_policy.action_limits,
            observation,
            targeted_type_available=task.targeted_type_available,
        ),
        progress_context=progress_context,
        planning_focus=pending_focus,
        discovery_mode=task.discovery_mode,
    )
    return CurrentPlanningView(request=request, pending_focus=pending_focus)
