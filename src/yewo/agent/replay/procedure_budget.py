"""Shared runtime-budget admission for compiled procedures."""

from dataclasses import dataclass

from yewo.agent.replay.procedure_repair import PROCEDURE_REPAIR_MAXIMUM_TOTAL_ACTIONS
from yewo.agent.replay.reconciliation_records import ReplayProcedure
from yewo.agent.runtime.observation_settlement import (
    default_action_settlement_policy,
    maximum_settlement_window_seconds,
)
from yewo.agent.runtime.records import AgentModePolicy


PROCEDURE_TERMINAL_STEP_RESERVE = 1


@dataclass(frozen=True)
class ProcedureBudgetRequirement:
    """Minimum local runtime capacity reserved before procedure replay."""

    action_count: int
    step_count: int
    duration_seconds: float


def build_procedure_budget_requirement(
    policy: AgentModePolicy,
    procedure: ReplayProcedure,
) -> ProcedureBudgetRequirement:
    """Calculate procedure, repair, terminal, and settlement capacity."""
    action_count = len(procedure.steps) + PROCEDURE_REPAIR_MAXIMUM_TOTAL_ACTIONS
    explicit_duration_seconds = sum(step.proposal.duration_seconds or 0 for step in procedure.steps)
    settlement_window_seconds = maximum_settlement_window_seconds(
        default_action_settlement_policy(policy)
    )
    return ProcedureBudgetRequirement(
        action_count=action_count,
        step_count=action_count + PROCEDURE_TERMINAL_STEP_RESERVE,
        duration_seconds=(explicit_duration_seconds + action_count * settlement_window_seconds),
    )


def procedure_budget_refusal_reason(
    policy: AgentModePolicy,
    procedure: ReplayProcedure,
) -> str | None:
    """Explain why one mode policy cannot safely admit the procedure."""
    requirement = build_procedure_budget_requirement(policy, procedure)
    if policy.maximum_steps < requirement.step_count:
        return "The mode step budget cannot execute the procedure and repair reserve."
    if policy.action_limits.maximum_actions < requirement.action_count:
        return "The action budget cannot execute the procedure and repair reserve."
    if policy.maximum_duration_seconds < requirement.duration_seconds:
        return "The duration budget cannot cover procedure action and settlement reserves."
    return None
