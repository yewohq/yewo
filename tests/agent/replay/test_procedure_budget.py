"""Compiled-procedure admission budget tests."""

from yewo.agent import AgentModePolicy
from yewo.agent.discovery.procedure_handoff import _policy_can_execute
from yewo.agent.replay.procedure.budget import (
    build_procedure_budget_requirement,
    procedure_budget_refusal_reason,
)
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure

from agent.actuation.support import limits
from agent.replay.test_procedure_compiler import _evidence, _response


def _procedure() -> ReplayProcedure:
    procedure = compile_replay_procedure(_evidence(), _response())
    assert isinstance(procedure, ReplayProcedure)
    return procedure


def _policy(**updates: object) -> AgentModePolicy:
    values: dict[str, object] = {
        "action_limits": limits(maximum_actions=100),
        "maximum_steps": 100,
        "maximum_grounding_requests": 100,
        "maximum_duration_seconds": 300,
        "maximum_settlement_observations": 6,
        "settlement_interval_seconds": 0.5,
    }
    values.update(updates)
    return AgentModePolicy.model_validate(values)


def test_requirement_reserves_repair_terminal_and_settlement_capacity() -> None:
    procedure = _procedure()

    requirement = build_procedure_budget_requirement(_policy(), procedure)

    assert requirement.action_count == len(procedure.steps) + 32
    assert requirement.step_count == requirement.action_count + 1
    assert requirement.duration_seconds == requirement.action_count * 3


def test_duration_reserve_can_refuse_an_undersized_policy() -> None:
    procedure = _procedure()
    requirement = build_procedure_budget_requirement(_policy(), procedure)
    policy = _policy(maximum_duration_seconds=requirement.duration_seconds - 0.1)

    assert procedure_budget_refusal_reason(policy, procedure) == (
        "The duration budget cannot cover procedure action and settlement reserves."
    )
    assert not _policy_can_execute(policy, procedure)


def test_practical_policy_defaults_remain_bounded() -> None:
    policy = AgentModePolicy(
        action_limits=limits(),
        maximum_grounding_requests=20,
    )

    assert policy.maximum_steps == 100
    assert policy.maximum_duration_seconds == 300
