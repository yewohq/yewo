"""Shared runtime no-progress suppression integration test."""

from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionRefusal,
    ActionRefusalCode,
    SideEffectClass,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.runtime.no_progress import (
    MAXIMUM_CONSECUTIVE_REFUSALS,
    ActionNoProgressTracker,
)
from yewo.agent.runtime.interaction_intent import interaction_intent_key
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import limits, proposal
from agent.runtime.test_agent import (
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
            proposal(ActionKind.TAP, target_expression="the Edit Record item"),
            proposal(
                ActionKind.TAP,
                proposal_id="proposal-2",
                grounding_correction_for_proposal_id="proposal-1",
                target_expression=(
                    "the Edit Record row near the top of the drawer, not the blank lower area"
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


def test_runtime_allows_one_declared_interaction_refinement_then_exhausts_family(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TAP, target_expression="the Example Record card"),
            proposal(
                ActionKind.TAP,
                proposal_id="refinement-template-1",
                interaction_refinement_for_proposal_id="proposal-1",
                target_expression="the Example Record text inside the card",
            ),
            proposal(
                ActionKind.TAP,
                proposal_id="refinement-template-2",
                interaction_refinement_for_proposal_id="proposal-1",
                target_expression="the right side of the Example Record card",
            ),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    observations = ObservationSequence(
        [
            _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
            for index in range(1, 6)
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.usage.dispatched_actions == 2
    assert result.steps[2].refusal is not None
    assert result.steps[2].refusal.code.value == "invalid_constraint"
    screen_fact = planner.requests[2].screen_facts[-1]
    assert "one visible interaction refinement" in screen_fact.fact
    assert screen_fact.screen.observation_id == "obs_3"
    assert screen_fact.screen.screen_fingerprint == "0000000000000000"


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


def test_provider_unavailability_does_not_poison_target_no_progress() -> None:
    screen = _observation("obs_1", "0000000000000000")
    attempted = proposal(ActionKind.TAP, source_observation_id=screen.observation_id)
    retry = proposal(
        ActionKind.TAP,
        proposal_id="proposal-2",
        source_observation_id=screen.observation_id,
    )
    refusal = ActionRefusal(
        proposal_id=attempted.proposal_id,
        code=ActionRefusalCode.PROVIDER_UNAVAILABLE,
        reason="RateLimitError (HTTP 429)",
    )
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)

    assert tracker.record_refusal(attempted, screen, refusal) is None
    assert tracker.refusal_for(retry, screen) is None
    assert tracker.known_facts() == ()


def test_runtime_stops_only_after_consecutive_refusals_reach_the_bound(
    tmp_path: Path,
) -> None:
    tap_only = limits(
        allowed_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
        supported_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
        allowed_side_effects=frozenset(
            {SideEffectClass.UNCLASSIFIED_INTERACTION, SideEffectClass.NONE}
        ),
    )
    planner = SequencePlanner(
        [proposal(ActionKind.WAIT) for _ in range(MAXIMUM_CONSECUTIVE_REFUSALS)]
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(action_limits=tap_only),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert "consecutive refusals" in result.reason
    assert len(result.steps) == MAXIMUM_CONSECUTIVE_REFUSALS


def test_runtime_requests_an_isolated_focus_decision_after_screen_exhaustion(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [proposal(ActionKind.TAP, target_expression=f"control {index}") for index in range(3)]
        + [proposal(ActionKind.BACK), proposal(ActionKind.CANNOT_PROGRESS)]
    )
    observations = ObservationSequence(
        [
            _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
            for index in range(1, 10)
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert planner.requests[3].planning_focus is not None
    assert planner.requests[3].planning_focus.cause == "screen_no_progress"
    assert planner.requests[2].planning_focus is None
    assert planner.requests[4].planning_focus is None


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
    screen_fact = planner.requests[2].screen_facts[-1]
    assert "already visited screen" in screen_fact.fact
    assert screen_fact.screen.observation_id == "a_3"


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
    candidates = []

    for index, (source, destination) in enumerate(
        zip(source_observations, destination_observations, strict=True)
    ):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"proposal-{index}",
            source_observation_id=source.observation_id,
            target_expression=f"rewritten target description {index}",
        )
        candidates.append(candidate)
        outcome = ActionOutcome(
            outcome_id=f"outcome-{index}",
            authorization_id=f"authorization-{index}",
            receipt_id=f"receipt-{index}",
            action=ActionKind.TAP,
            source_observation_id=source.observation_id,
            settled_observation_id=destination.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
            reason="The settled observation contains action-outcome signals.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )
        terminal_reasons.append(tracker.record_outcome(candidate, source, destination, outcome))

    assert terminal_reasons == [None, None, None]
    assert any(
        "repeated an already observed screen transition" in fact for fact in tracker.known_facts()
    )
    focus = tracker.pending_planning_focus(
        destination_observations[-1],
        lambda: "focus-repeated-transition",
        ("Reach the next workflow stage.",),
        (),
    )
    if focus is not None:
        tracker.commit_planning_focus(destination_observations[-1])
    assert focus is not None
    assert focus.cause == "repeated_transition"
    assert focus.prohibited_interaction_intents == (interaction_intent_key(candidates[1]),)
    repeat = proposal(
        ActionKind.TAP,
        proposal_id="proposal-repeat",
        source_observation_id=source_observations[0].observation_id,
        target_expression="rewritten target description 0",
    )
    refusal = tracker.refusal_for(repeat, source_observations[0])
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS


def test_distinct_exact_states_do_not_look_like_one_repeated_transition() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy(maximum_phash_distance=0))
    terminal_reasons: list[str | None] = []

    for index in range(3):
        source = _observation(f"source_{index}", "0000000000000000").model_copy(
            update={
                "screen_fingerprint": _observation(
                    f"source_fingerprint_{index}", "0000000000000000"
                ).screen_fingerprint.model_copy(
                    update={"masked_pixels_sha256": f"{index + 1:064x}"}
                )
            }
        )
        destination = _observation(f"destination_{index}", "1111111111111111").model_copy(
            update={
                "screen_fingerprint": _observation(
                    f"destination_fingerprint_{index}", "1111111111111111"
                ).screen_fingerprint.model_copy(
                    update={"masked_pixels_sha256": f"{index + 4:064x}"}
                )
            }
        )
        candidate = proposal(
            ActionKind.TYPE,
            proposal_id=f"proposal-{index}",
            source_observation_id=source.observation_id,
            target_expression=f"field {index}",
        )
        outcome = ActionOutcome(
            outcome_id=f"outcome-{index}",
            authorization_id=f"authorization-{index}",
            receipt_id=f"receipt-{index}",
            action=ActionKind.TYPE,
            source_observation_id=source.observation_id,
            settled_observation_id=destination.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset(
                {ActionOutcomeSignal.PIXELS_CHANGED, ActionOutcomeSignal.TEXT_LANDED}
            ),
            reason="Text landed.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        )
        terminal_reasons.append(tracker.record_outcome(candidate, source, destination, outcome))

    assert terminal_reasons == [None, None, None]
    assert all(
        "repeated an already observed screen transition" not in fact
        for fact in tracker.known_facts()
    )


def test_exact_repeated_transition_is_refused_before_a_third_dispatch() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    before = _observation("source", "0000000000000000")
    after = _observation("destination", "1111111111111111")
    candidate = proposal(
        ActionKind.TAP,
        source_observation_id=before.observation_id,
        target_expression="the overflow menu",
    )
    outcome = ActionOutcome(
        outcome_id="outcome",
        authorization_id="authorization",
        receipt_id="receipt",
        action=ActionKind.TAP,
        source_observation_id=before.observation_id,
        settled_observation_id=after.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The menu opened.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    tracker.start(before)
    tracker.record_outcome(candidate, before, after, outcome)
    tracker.record_outcome(candidate, before, after, outcome)

    refusal = tracker.refusal_for(candidate, before)

    assert refusal is not None
    assert refusal.code.value == "repeated_no_progress"
    assert "repeated an equivalent screen transition" in refusal.reason


def test_application_exit_refuses_the_same_action_after_recovery() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    before = _observation("source", "0000000000000000")
    recovered = _observation("recovered", "1111111111111111")
    candidate = proposal(ActionKind.TAP, target_expression="the Save button")
    outcome = ActionOutcome(
        outcome_id="outcome",
        authorization_id="authorization",
        receipt_id="receipt",
        action=ActionKind.TAP,
        source_observation_id=before.observation_id,
        settled_observation_id=recovered.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.APP_EXITED}),
        reason="The application crashed after Save.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    tracker.start(before)

    tracker.record_outcome(candidate, before, recovered, outcome)

    assert tracker.refusal_for(candidate, before) is not None
    assert "application exited" in tracker.known_facts()[-1]


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


def test_runtime_reserves_all_requests_needed_for_the_next_planning_operation(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [proposal(ActionKind.CANNOT_PROGRESS)],
        model_requests_per_plan=2,
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(maximum_model_requests=1),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.planning_requests == 0
    assert planner.requests == []


def test_a_conditional_refusal_never_blocks_the_action_it_asks_yewo_to_resolve() -> None:
    comparison = ScreenComparisonPolicy(maximum_phash_distance=0)
    tracker = ActionNoProgressTracker(comparison)
    screen = _observation("obs_1", "0000000000000000")
    tracker.start(screen)
    advance = proposal(
        ActionKind.TAP,
        proposal_id="proposal-advance",
        source_observation_id=screen.observation_id,
        target_expression="the save control",
    )
    coverage_refusal = ActionRefusal(
        proposal_id=advance.proposal_id,
        code=ActionRefusalCode.DISCOVERY_INCOMPLETE,
        reason="Two dropdown controls remain unopened.",
    )

    assert tracker.record_refusal(advance, screen, coverage_refusal) is None
    assert tracker.refusal_for(advance, screen) is None
    assert any("Resolve this before you repeat" in fact for fact in tracker.known_facts())
