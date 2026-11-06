"""Build the Device Agent's terminal result and gross usage record."""

from yewo.agent.recovery.records import RecoveryBudgetCredit
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.planning.evidence import resolve_planning_model_request_count
from yewo.agent.runtime.records import (
    AgentResult,
    AgentStep,
    AgentTerminalReason,
    AgentUsage,
)
from yewo.agent.screen_analysis import CurrentScreenAnalysis
from yewo.observations import Observation


def build_initial_agent_result(
    terminal: AgentTerminalReason,
    reason: str,
    observation: Observation,
    planning_requests: int,
) -> AgentResult:
    """Build a result for an application terminal observed before the loop."""
    return build_agent_result(terminal, reason, observation, planning_requests, 0, 0, [])


def record_planning_model_usage(
    recovery: RecoveryRunState,
    source: object,
    reserved_count: int,
) -> int:
    """Credit recovery and return actual model calls after one plan operation."""
    consumed_count = resolve_planning_model_request_count(source, reserved_count)
    recovery.credit_planning_requests(consumed_count)
    return consumed_count


def build_agent_result(
    terminal: AgentTerminalReason,
    reason: str,
    observation: Observation,
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
    steps: list[AgentStep],
    recovery_budget_credit: RecoveryBudgetCredit | None = None,
    screen_analyses: tuple[CurrentScreenAnalysis, ...] = (),
) -> AgentResult:
    """Build one Device Agent conclusion from gross counters and explicit credit."""
    return AgentResult(
        terminal_reason=terminal,
        reason=reason,
        final_observation_id=observation.observation_id,
        usage=AgentUsage(
            steps=len(steps),
            planning_requests=planning_requests,
            grounding_requests=grounding_requests,
            dispatched_actions=dispatched_actions,
        ),
        recovery_budget_credit=recovery_budget_credit or RecoveryBudgetCredit(),
        steps=tuple(steps),
        screen_analyses=screen_analyses,
    )
