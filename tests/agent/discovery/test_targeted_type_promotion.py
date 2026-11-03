"""Targeted text-entry promotion regressions."""

from datetime import datetime, timezone

from yewo.agent import InputValueShape
from yewo.agent.actuation import ActionKind
from yewo.agent.discovery import AppMemoryBuilder

from agent.discovery.support import agent_result, observation
from agent.discovery.test_promotion import _compatibility, _request


def test_targeted_literal_type_builds_input_contract_before_validation() -> None:
    before = observation("obs_1", "0000000000000000")
    typed = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(before, typed, action=ActionKind.TYPE)
    type_step = result.steps[0]
    targeted_proposal = type_step.proposal.model_copy(
        update={"target_expression": "the Email Address text input field"}
    )
    targeted_step = type_step.model_copy(update={"proposal": targeted_proposal})
    result = result.model_copy(update={"steps": (targeted_step, *result.steps[1:])})

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before, typed.observation_id: typed},
        _compatibility(),
        verified_objective_ids=frozenset({"account_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    assert len(promotion.memory.transitions) == 1
    action = promotion.memory.transitions[0].action
    assert action.action is ActionKind.TYPE
    assert action.target_description == "the Email Address text input field"
    assert action.input_contract is not None
    assert action.input_contract.value_shape is InputValueShape.TEXT
