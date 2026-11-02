"""Build the Device Agent's terminal result and gross usage record."""

from yewo.agent.recovery_records import RecoveryBudgetCredit
from yewo.agent.runtime_records import (
    AgentResult,
    AgentStep,
    AgentTerminalReason,
    AgentUsage,
)
from yewo.observations import Observation


def build_agent_result(
    terminal: AgentTerminalReason,
    reason: str,
    observation: Observation,
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
    steps: list[AgentStep],
    recovery_budget_credit: RecoveryBudgetCredit | None = None,
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
    )
