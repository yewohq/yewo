"""Separate scenario-owned Device Agent budget accounting."""

import pytest

from yewo.agent import AgentModePolicy, AgentUsage
from yewo.scenarios import ScenarioBudgetBank, ScenarioBudgetExhausted

from agent.actuation.support import limits


def _policy() -> AgentModePolicy:
    return AgentModePolicy(
        action_limits=limits(maximum_actions=100),
        maximum_steps=100,
        maximum_grounding_requests=200,
        maximum_model_requests=300,
        maximum_duration_seconds=1_200,
    )


def test_route_usage_is_subtracted_before_scenario_task_work() -> None:
    bank = ScenarioBudgetBank(_policy())

    bank.record(
        AgentUsage(
            steps=10,
            planning_requests=20,
            grounding_requests=2,
            dispatched_actions=10,
        ),
        elapsed_seconds=300,
    )

    remaining = bank.require_policy()
    assert remaining.maximum_steps == 90
    assert remaining.action_limits.maximum_actions == 90
    assert remaining.maximum_grounding_requests == 198
    assert remaining.maximum_model_requests == 278
    assert remaining.maximum_duration_seconds == 900


def test_exhausted_route_prevents_a_second_scenario_agent_allowance() -> None:
    bank = ScenarioBudgetBank(_policy())

    bank.record(
        AgentUsage(
            steps=100,
            planning_requests=100,
            grounding_requests=0,
            dispatched_actions=100,
        ),
        elapsed_seconds=600,
    )

    with pytest.raises(ScenarioBudgetExhausted, match="budget is exhausted"):
        bank.require_policy()
