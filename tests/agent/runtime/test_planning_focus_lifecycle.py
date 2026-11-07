"""Planning-focus lifecycle: owed, claimed, and spent exactly once."""

from pathlib import Path

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import proposal
from agent.runtime.test_agent import _observation


def _no_change_outcome(screen, label: str) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=f"outcome-{label}",
        authorization_id=f"authorization-{label}",
        receipt_id=f"receipt-{label}",
        action=ActionKind.TAP,
        source_observation_id=screen.observation_id,
        settled_observation_id=screen.observation_id,
        disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        signals=frozenset(),
        reason="No application pixels or trusted state changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def _exhaust_screen(tracker: ActionNoProgressTracker, screen, label: str) -> None:
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"{label}-{index}",
            target_expression=f"{label} control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="The grounder did not accept this target.",
            ),
        )


def test_refused_focus_proposal_keeps_the_exemption_for_a_fresh_action() -> None:
    screen = _observation("obs_1", "0000000000000000")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    _exhaust_screen(tracker, screen, "failed")
    focus = tracker.pending_planning_focus(screen, lambda: "focus-1", ("Reopen it.",), ())
    assert focus is not None
    tracker.commit_planning_focus(screen)

    repeated = proposal(
        ActionKind.TAP,
        proposal_id="repeats-a-prohibited-intent",
        target_expression="failed control 0",
        source_observation_id=screen.observation_id,
    )
    refusal = tracker.refusal_for(repeated, screen)
    assert refusal is not None
    tracker.record_refusal(repeated, screen, refusal)

    fresh = proposal(
        ActionKind.TAP,
        proposal_id="fresh-action",
        target_expression="a genuinely untried control",
        source_observation_id=screen.observation_id,
    )

    assert tracker.refusal_for(fresh, screen) is None


def test_verifier_refusal_does_not_burn_the_focus_exemption() -> None:
    screen = _observation("obs_1", "0000000000000000")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    _exhaust_screen(tracker, screen, "failed")
    assert (
        tracker.pending_planning_focus(screen, lambda: "focus-1", ("Reopen it.",), ()) is not None
    )
    tracker.commit_planning_focus(screen)

    verified = proposal(
        ActionKind.TAP,
        proposal_id="refused-before-dispatch",
        target_expression="a control the verifier rejects",
        source_observation_id=screen.observation_id,
    )
    tracker.record_refusal(
        verified,
        screen,
        ActionRefusal(
            proposal_id=verified.proposal_id,
            code=ActionRefusalCode.VERIFICATION_UNAVAILABLE,
            reason="The action verifier could not assess this proposal.",
        ),
    )

    fresh = proposal(
        ActionKind.TAP,
        proposal_id="fresh-action",
        target_expression="a genuinely untried control",
        source_observation_id=screen.observation_id,
    )

    assert tracker.refusal_for(fresh, screen) is None


def test_unreached_pending_focus_does_not_block_a_later_stuck_screen() -> None:
    first = _observation("obs_1", "0000000000000000")
    second = _observation("obs_2", "1111111111111111")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(first)
    _exhaust_screen(tracker, first, "first")
    _exhaust_screen(tracker, second, "second")

    focus = tracker.pending_planning_focus(second, lambda: "focus-2", ("Reopen it.",), ())

    assert focus is not None
    tracker.commit_planning_focus(second)
    fresh = proposal(
        ActionKind.TAP,
        proposal_id="fresh-on-second",
        target_expression="an untried control on the second screen",
        source_observation_id=second.observation_id,
    )
    assert tracker.refusal_for(fresh, second) is None


def _observed_outcome(before, after, label: str) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=f"outcome-{label}",
        authorization_id=f"authorization-{label}",
        receipt_id=f"receipt-{label}",
        action=ActionKind.TAP,
        source_observation_id=before.observation_id,
        settled_observation_id=after.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The settled observation contains action-outcome signals.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def test_grounding_correction_cannot_bypass_an_exhausted_action() -> None:
    source = _observation("obs_1", "0000000000000000")
    destination = _observation("obs_2", "1111111111111111")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(source)
    repeated = proposal(
        ActionKind.TAP,
        proposal_id="repeated",
        target_expression="the same control",
        source_observation_id=source.observation_id,
    )
    for index in range(3):
        tracker.record_outcome(
            repeated,
            source,
            destination,
            _observed_outcome(source, destination, f"repeat-{index}"),
        )

    assert tracker.refusal_for(repeated, source) is not None

    correction = proposal(
        ActionKind.TAP,
        proposal_id="correction",
        target_expression="the same control",
        source_observation_id=source.observation_id,
        grounding_correction_for_proposal_id="repeated",
    )

    assert tracker.refusal_for(correction, source) is not None


def test_screen_counter_read_and_write_agree_under_tolerant_matching() -> None:
    # Hamming matching is not transitive, so one logical screen accrues two entries.
    first = _observation("obs_1", "0000000000000000")
    second = _observation("obs_2", "0000000000000003")
    between = _observation("obs_3", "0000000000000001")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy(maximum_phash_distance=1))
    tracker.start(first)

    for index, screen in ((0, first), (1, first), (2, second)):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"seed-{index}",
            target_expression=f"seed control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_outcome(
            candidate,
            screen,
            screen,
            _no_change_outcome(screen, f"seed-{index}"),
        )

    third = proposal(
        ActionKind.TAP,
        proposal_id="third-on-first-screen",
        target_expression="a third control",
        source_observation_id=between.observation_id,
    )
    tracker.record_outcome(third, between, between, _no_change_outcome(between, "third"))

    fresh = proposal(
        ActionKind.TAP,
        proposal_id="fresh",
        target_expression="an untried control",
        source_observation_id=between.observation_id,
    )

    assert tracker.refusal_for(fresh, between) is not None


def test_screen_facts_reach_the_planner_only_for_the_current_screen() -> None:
    first = _observation("obs_1", "0000000000000000")
    second = _observation("obs_2", "1111111111111111")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(first)
    _exhaust_screen(tracker, first, "first")

    assert tracker.screen_facts(first)
    assert tracker.screen_facts(second) == ()


def test_more_screen_facts_than_the_cap_are_truncated_not_rejected() -> None:
    from yewo.agent.planning.records import (
        MAXIMUM_SCREEN_FACTS,
        PlanningRequest,
    )

    screen = _observation("obs_1", "0000000000000000")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    for index in range(MAXIMUM_SCREEN_FACTS + 20):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"conditional-{index}",
            target_expression=f"control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.DISCOVERY_INCOMPLETE,
                reason=f"Resolve condition {index} first.",
            ),
        )

    facts = tracker.screen_facts(screen)
    assert len(facts) == MAXIMUM_SCREEN_FACTS
    request = PlanningRequest(
        observation=screen,
        objective="Finish the task.",
        completion_conditions=("The task is finished.",),
        available_actions=frozenset({ActionKind.DONE}),
        screen_facts=facts,
    )

    # The newest evidence survives, because the planner acts on what it just learned.
    assert request.screen_facts[-1] == facts[-1]


def test_more_prohibited_intents_than_the_cap_are_truncated_not_rejected() -> None:
    from yewo.agent.planning.records import (
        MAXIMUM_PROHIBITED_INTENTS,
    )

    screen = _observation("obs_1", "0000000000000000")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    for index in range(MAXIMUM_PROHIBITED_INTENTS + 20):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"refused-{index}",
            target_expression=f"control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="The grounder did not accept this target.",
            ),
        )

    focus = tracker.pending_planning_focus(screen, lambda: "focus-1", ("Reopen it.",), ())
    assert focus is not None
    assert len(focus.prohibited_interaction_intents) == MAXIMUM_PROHIBITED_INTENTS


def test_a_source_that_declines_focus_leaves_the_request_unclaimed() -> None:
    from yewo.agent.planning.planner import source_accepts_planning_focus

    class RepairingSource:
        def declines_planning_focus(self) -> bool:
            return True

    assert source_accepts_planning_focus(RepairingSource()) is False
    assert source_accepts_planning_focus(object()) is True


def test_the_agent_never_claims_a_focus_a_declining_source_cannot_receive(
    tmp_path: Path,
) -> None:
    import json

    from agent.runtime.test_agent import (
        NeverInterrupted,
        ObservationSequence,
        RuntimeDevice,
        SequencePlanner,
        _mode_policy,
        _runtime,
        _task,
    )

    class DecliningPlanner(SequencePlanner):
        def declines_planning_focus(self) -> bool:
            return True

    def run(planner_type):
        root = tmp_path / planner_type.__name__
        root.mkdir()
        stuck = [
            _observation(f"obs_{index}", "0000000000000000", offset_seconds=index)
            for index in range(40)
        ]
        planner = planner_type([proposal() for _ in range(5)])
        runtime = _runtime(root, planner, ObservationSequence(stuck), RuntimeDevice())
        runtime.run(_task(), _mode_policy(), NeverInterrupted())
        ledger = root / "agent/progress-ledger.jsonl"
        if not ledger.is_file():
            return []
        return [
            json.loads(line)["kind"] for line in ledger.read_text().splitlines() if line.strip()
        ]

    assert "planning_focus" in run(SequencePlanner)
    assert "planning_focus" not in run(DecliningPlanner)
