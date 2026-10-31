"""Shared runtime no-progress suppression integration test."""

from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    SideEffectClass,
)
from yewo.agent.app_memory import ScreenComparisonPolicy
from yewo.agent.no_progress import ActionNoProgressTracker
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import limits, proposal
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


def test_runtime_refuses_an_exact_repeat_after_no_observable_progress(
    tmp_path: Path,
) -> None:
    wait_limits = limits(
        allowed_actions=frozenset({ActionKind.WAIT}),
        supported_actions=frozenset({ActionKind.WAIT}),
        allowed_side_effects=frozenset({SideEffectClass.NONE}),
    )
    planner = SequencePlanner(
        [proposal(ActionKind.WAIT), proposal(ActionKind.WAIT), proposal(ActionKind.CANNOT_PROGRESS)]
    )
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "0000000000000000", offset_seconds=1),
            _observation("obs_3", "0000000000000000", offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(action_limits=wait_limits),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.steps[1].refusal is not None
    assert result.steps[1].refusal.code.value == "repeated_no_progress"
    assert result.usage.dispatched_actions == 1
    assert result.usage.grounding_requests == 0


def test_runtime_exposes_failed_target_point_and_stops_after_one_correction(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TAP, target_expression="the Edit Schedule item"),
            proposal(
                ActionKind.TAP,
                proposal_id="proposal-2",
                grounding_correction_for_proposal_id="proposal-1",
                target_expression=(
                    "the Edit Schedule row near the top of the drawer, not the blank lower area"
                ),
            ),
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

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.reason == "The grounding correction produced no observable change."
    assert result.usage.dispatched_actions == 2
    failed_point = planner.requests[1].history[0].grounded_target_point
    assert failed_point is not None
    assert (failed_point.point.x, failed_point.point.y) == (100, 200)


def test_no_progress_suppression_uses_equivalent_screen_policy() -> None:
    first = _observation("obs_1", "0000000000000000")
    nearby = _observation("obs_2", "0000000000000001")
    first_proposal = proposal(ActionKind.WAIT, source_observation_id=first.observation_id)
    repeated = proposal(
        ActionKind.WAIT,
        proposal_id="proposal-2",
        source_observation_id=nearby.observation_id,
    )
    outcome = ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.WAIT,
        source_observation_id=first.observation_id,
        settled_observation_id=first.observation_id,
        disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        signals=frozenset(),
        reason="No application pixels or trusted state changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy(maximum_phash_distance=1))
    tracker.start(first)
    tracker.record_outcome(first_proposal, first, first, outcome)

    assert tracker.refusal_for(repeated, nearby) is not None


def test_runtime_stops_after_an_equivalent_refusal_repeats(
    tmp_path: Path,
) -> None:
    tap_only = limits(
        allowed_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
        supported_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
        allowed_side_effects=frozenset(
            {SideEffectClass.UNCLASSIFIED_INTERACTION, SideEffectClass.NONE}
        ),
    )
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.WAIT)])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(action_limits=tap_only),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert "repeatedly refused" in result.reason
    assert len(result.steps) == 2


def test_runtime_allows_distinct_routes_to_return_to_a_visited_screen(
    tmp_path: Path,
) -> None:
    wait_limits = limits(
        allowed_actions=frozenset({ActionKind.WAIT}),
        supported_actions=frozenset({ActionKind.WAIT}),
        allowed_side_effects=frozenset({SideEffectClass.NONE}),
    )
    planner = SequencePlanner(
        [
            proposal(ActionKind.WAIT),
            proposal(ActionKind.WAIT),
            proposal(ActionKind.WAIT),
            proposal(ActionKind.WAIT),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    observations = ObservationSequence(
        [
            _observation("a_1", "0000000000000000"),
            _observation("b_1", "1111111111111111", offset_seconds=1),
            _observation("b_2", "1111111111111111", offset_seconds=2),
            _observation("a_2", "0000000000000000", offset_seconds=3),
            _observation("a_3", "0000000000000000", offset_seconds=4),
            _observation("c_1", "aaaaaaaaaaaaaaaa", offset_seconds=5),
            _observation("c_2", "aaaaaaaaaaaaaaaa", offset_seconds=6),
            _observation("a_4", "0000000000000000", offset_seconds=7),
            _observation("a_5", "0000000000000000", offset_seconds=8),
        ]
    )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        screen_comparison=ScreenComparisonPolicy(maximum_phash_distance=1),
    ).run(
        _task(),
        _mode_policy(action_limits=wait_limits),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.reason == "No permitted action can progress."
    assert result.usage.dispatched_actions == 4
    assert len(planner.requests) == 5
    assert "already visited screen" in planner.requests[2].known_facts[-1]


def test_repeated_observed_transition_warns_then_stops_despite_target_rewording() -> None:
    comparison = ScreenComparisonPolicy(maximum_phash_distance=0)
    tracker = ActionNoProgressTracker(comparison)
    source_observations = tuple(
        _observation(f"source_{index}", "0000000000000000") for index in range(3)
    )
    destination_observations = tuple(
        _observation(f"destination_{index}", "1111111111111111") for index in range(3)
    )
    tracker.start(source_observations[0])
    terminal_reasons: list[str | None] = []

    for index, (source, destination) in enumerate(
        zip(source_observations, destination_observations, strict=True)
    ):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"proposal-{index}",
            source_observation_id=source.observation_id,
            target_expression=f"rewritten target description {index}",
        )
        outcome = ActionOutcome(
            outcome_id=f"outcome-{index}",
            authorization_id=f"authorization-{index}",
            receipt_id=f"receipt-{index}",
            action=ActionKind.TAP,
            source_observation_id=source.observation_id,
            settled_observation_id=destination.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.SCREEN_CHANGED}),
            reason="The settled observation contains action-outcome signals.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )
        terminal_reasons.append(
            tracker.record_outcome(candidate, source, destination, outcome)
        )

    assert terminal_reasons[:2] == [None, None]
    assert "repeated an already observed screen transition" in tracker.known_facts()[-1]
    assert terminal_reasons[2] is not None


def test_runtime_enforces_duration_before_another_model_request(tmp_path: Path) -> None:
    times = iter((0.0, 301.0))
    planner = SequencePlanner([proposal(ActionKind.CANNOT_PROGRESS)])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        monotonic=lambda: next(times),
    ).run(_task(), _mode_policy(maximum_duration_seconds=300), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.planning_requests == 0


def test_runtime_enforces_one_total_model_request_budget(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.TAP)])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(maximum_model_requests=1),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.planning_requests == 1
    assert result.usage.grounding_requests == 0
    assert result.usage.dispatched_actions == 0
