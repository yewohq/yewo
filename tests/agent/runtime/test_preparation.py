"""Runtime accounting for work completed before live observation."""

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind
from yewo.runs import NeverInterrupted

from agent.actuation.support import limits, proposal
from agent.runtime.test_agent import (
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


def test_runtime_accounts_for_pre_observation_reconciliation_request(tmp_path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)], model_requests_per_plan=0)
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        initial_planning_requests=1,
    ).run(_task(), _mode_policy(maximum_model_requests=1), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert result.usage.planning_requests == 1


def test_action_budget_still_allows_planner_to_confirm_completion(tmp_path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(action_limits=limits(maximum_actions=1)), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert result.usage.dispatched_actions == 1
    assert result.usage.planning_requests == 2
