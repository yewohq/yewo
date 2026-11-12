"""A correction that reaches nothing retires its route, not the run."""

from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind

from agent.actuation.support import proposal
from agent.runtime.support import RuntimeDevice
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


def test_a_correction_that_changes_nothing_retires_the_route_not_the_run(
    tmp_path: Path,
) -> None:
    # The planner's correction can be right while the grounder still misses. Ending the run on
    # one unreachable target discards every other route the planner can still see.
    planner = SequencePlanner(
        [
            proposal(ActionKind.TAP, target_expression="the Edit Record item"),
            proposal(
                ActionKind.TAP,
                proposal_id="proposal-2",
                grounding_correction_for_proposal_id="proposal-1",
                target_expression=(
                    "the Edit Record row near the top of the drawer, not the blank lower area"
                ),
            ),
            proposal(ActionKind.CANNOT_PROGRESS, proposal_id="proposal-3"),
        ]
    )
    observations = ObservationSequence(
        [
            _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
            for index in range(1, 6)
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(),
        NeverInterrupted(),
    )

    # The run still ends, but because the planner said so, not because one aim failed.
    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.reason != "The grounding correction produced no observable change."
    assert result.usage.dispatched_actions == 2
    # The planner is asked again, and the failed point travels with the question.
    assert len(planner.requests) == 3
    failed_point = planner.requests[1].history[0].grounded_target_point
    assert failed_point is not None
    assert (failed_point.point.x, failed_point.point.y) == (100, 200)
