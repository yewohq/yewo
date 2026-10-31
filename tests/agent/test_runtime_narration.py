"""Human-readable narration from the Device Agent runtime."""

from pathlib import Path

from yewo.agent.actuation import ActionKind
from yewo.run_narration import RunNarrator, narration_scope

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


def test_runtime_narrates_agent_decisions_actions_and_outcomes(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    with narration_scope(RunNarrator(tmp_path)):
        _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
            _task(), _mode_policy(), NeverInterrupted()
        )

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "device-agent: Starting task: Open the account form." in narration
    assert "Step 1 decision: This is the next useful action" in narration
    assert "Step 1 planned tap on the add button." in narration
    assert "Step 1 grounded the target: accepted." in narration
    assert "Step 1 dispatching tap." in narration
    assert "Step 1 outcome observed. Signals: screen_changed." in narration
    assert "Step 2 decision: Task is complete." in narration
    assert "Terminal done_requested: Task is complete." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "observation=obs_1 label=Initial" in trace
    assert "proposal=proposal-1 observation=obs_1 action=tap" in trace
    assert "grounding=" in trace
    assert "point=(100,200)" in trace
    assert "authorization=" in trace
    assert "receipt=" in trace
    assert "outcome=" in trace
    assert "fingerprint=1111111111111111" in trace
