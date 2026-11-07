"""Forward-only procedure repair and input-preservation policy."""

from agent.replay.support import observations
from agent.replay.test_procedure_repair import (
    FixedRepairPlanner,
    _repair_source,
    _request,
    _unknown_observation,
)
from yewo.agent.actuation import ActionKind, ActionProposal


def test_repair_rejoins_at_a_later_distinct_compiled_state() -> None:
    source, planner = _repair_source()
    later_state = observations()[3]
    wrong = _unknown_observation()

    assert source.plan(_request(wrong)).action is ActionKind.WAIT
    resumed = source.plan(_request(later_state))

    assert resumed.action is ActionKind.TAP
    assert source.completed_step_count == 2
    assert planner.request_count == 0
    assert source.repair_evidence is not None
    assert source.repair_evidence.succeeded


def test_repair_refuses_uncompiled_clear_that_would_undo_progress() -> None:
    wrong = _unknown_observation()
    planner = FixedRepairPlanner(
        ActionProposal(
            proposal_id="regressive_clear",
            source_observation_id=wrong.observation_id,
            action=ActionKind.CLEAR,
            expected_outcome="The focused field becomes empty.",
            reason="Restore the older empty checkpoint.",
        )
    )
    source, unused_planner = _repair_source(planner=planner)

    assert source.plan(_request(wrong)).action is ActionKind.WAIT
    stopped = source.plan(_request(wrong))

    assert stopped.action is ActionKind.CANNOT_PROGRESS
    assert stopped.reason is not None and "refused to clear" in stopped.reason
    assert source.repair_evidence is not None
    assert source.repair_evidence.action_count == 1
