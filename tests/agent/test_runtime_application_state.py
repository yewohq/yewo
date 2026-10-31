"""Application-state and settlement tests for the shared runtime."""

from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionOutcomeSignal

from agent.actuation.support import proposal
from agent.test_runtime import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


def test_application_exit_before_planning_is_a_distinct_terminal(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", application_state=1)]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.usage.planning_requests == 0
    assert result.steps == ()
    assert planner.requests == []
    assert not (tmp_path / "assets/agent/steps.jsonl").exists()


def test_background_application_stops_before_planning(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", application_state=3)]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
    assert result.usage.planning_requests == 0
    assert planner.requests == []


def test_unchanged_screen_settles_only_on_the_final_attempt(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "0000000000000000", offset_seconds=1),
            _observation("obs_3", "0000000000000000", offset_seconds=2),
            _observation("obs_4", "0000000000000000", offset_seconds=3),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(maximum_settlement_observations=3),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert observations.capture_count == 4
    assert result.steps[0].settlement_observation_ids == ("obs_2", "obs_3", "obs_4")
    assert result.steps[0].settled_observation_id == "obs_4"


def test_application_exit_after_action_is_recorded_as_an_observed_outcome(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", application_state=1, offset_seconds=1),
            _observation("obs_3", "1111111111111111", application_state=1, offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.steps[0].outcome is not None
    assert ActionOutcomeSignal.APP_EXITED in result.steps[0].outcome.signals
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_EXITED


def test_background_application_after_action_is_terminal(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", application_state=3, offset_seconds=1),
            _observation("obs_3", "1111111111111111", application_state=3, offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
