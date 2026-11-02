"""Cross-run replay bypass and repair-rejoin integration tests."""

from yewo.agent import RouteActionSource
from yewo.agent.actuation import ActionKind
from yewo.agent.replay_rejoin import ReplayRejoinActionSource, build_replay_rejoin_plans

from agent.test_replay_rejoin import RecordingRepairPlanner, _chain_memory, _request


def test_known_divergent_transition_is_not_reintroduced_during_repair() -> None:
    memory, observations, unused_evidence = _chain_memory()
    checkpoint = memory.checkpoints[0]
    transition_id = checkpoint.transition_ids[0]
    replay = RouteActionSource(
        memory,
        checkpoint,
        known_divergent_transition_id=transition_id,
    )

    terminal = replay.plan(_request(observations[2]))
    plans = build_replay_rejoin_plans(
        checkpoint.checkpoint_id,
        replay.decision_metadata().source,
        replay.replay_transition_ids,
        replay.downstream_rejoin_transition_ids,
        replay.rejoin_segment_index,
        replay_diverged=True,
    )
    repair = RecordingRepairPlanner()
    source = ReplayRejoinActionSource(repair, memory, plans[0])

    repair_action = source.plan(_request(observations[2]))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert plans[0].minimum_segment_index == 1
    assert repair_action.action is ActionKind.WAIT
    assert len(repair.requests) == 1
