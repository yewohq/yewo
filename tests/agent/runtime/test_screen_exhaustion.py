"""Screen-scoped deterministic no-progress bounds."""

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
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


def test_screen_no_progress_limit_blocks_more_interactions_but_allows_exit() -> None:
    screen = _observation("obs_1", "0000000000000000")
    candidates = (
        proposal(
            ActionKind.TAP,
            target_expression="the saved record",
            source_observation_id=screen.observation_id,
        ),
        proposal(
            ActionKind.TAP,
            proposal_id="proposal-2",
            target_expression="the overflow menu",
            source_observation_id=screen.observation_id,
        ),
        proposal(
            ActionKind.TAP,
            proposal_id="proposal-3",
            target_expression="the settings command",
            source_observation_id=screen.observation_id,
        ),
    )
    outcome = ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.TAP,
        source_observation_id=screen.observation_id,
        settled_observation_id=screen.observation_id,
        disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        signals=frozenset(),
        reason="No application pixels or trusted state changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    for candidate in candidates:
        tracker.record_outcome(candidate, screen, screen, outcome)

    another_interaction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-4",
        target_expression="the floating add control",
        source_observation_id=screen.observation_id,
    )
    leave_screen = proposal(
        ActionKind.BACK,
        proposal_id="proposal-5",
        source_observation_id=screen.observation_id,
    )

    refusal = tracker.refusal_for(another_interaction, screen)

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS
    assert "no-progress limit" in refusal.reason
    assert tracker.refusal_for(leave_screen, screen) is None


def test_screen_no_progress_limit_counts_distinct_grounding_refusals() -> None:
    screen = _observation("obs_1", "0000000000000000")
    candidates = (
        proposal(
            ActionKind.TAP,
            target_expression="the saved record",
            source_observation_id=screen.observation_id,
        ),
        proposal(
            ActionKind.TAP,
            proposal_id="proposal-2",
            target_expression="the overflow menu",
            source_observation_id=screen.observation_id,
        ),
        proposal(
            ActionKind.TAP,
            proposal_id="proposal-3",
            target_expression="the settings command",
            source_observation_id=screen.observation_id,
        ),
    )
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    for index, candidate in enumerate(candidates, start=1):
        tracker.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason=f"Grounding attempt {index} was not accepted.",
            ),
        )

    another_interaction = proposal(
        ActionKind.TAP,
        proposal_id="proposal-4",
        target_expression="the floating add control",
        source_observation_id=screen.observation_id,
    )

    refusal = tracker.refusal_for(another_interaction, screen)

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS


def test_screen_limit_opens_one_fresh_action_for_the_stuck_state() -> None:
    screen = _observation("obs_1", "0000000000000000")
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    tracker.start(screen)
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"failed-{index}",
            target_expression=f"failed control {index}",
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
    tracker.commit_planning_focus(screen)

    assert focus is not None
    assert focus.cause == "screen_no_progress"
    assert focus.unresolved_work == ("Reopen it.",)
    assert len(focus.prohibited_interaction_intents) == 3
    fresh_action = proposal(
        ActionKind.TAP,
        proposal_id="fresh-action",
        target_expression="a different visible control",
        source_observation_id=screen.observation_id,
    )
    assert tracker.refusal_for(fresh_action, screen) is None
    tracker.record_outcome(fresh_action, screen, screen, _no_change_outcome(screen, "fresh"))
    later_action = proposal(
        ActionKind.TAP,
        proposal_id="later-action",
        target_expression="another different control",
        source_observation_id=screen.observation_id,
    )
    assert tracker.refusal_for(later_action, screen) is not None
    assert tracker.pending_planning_focus(screen, lambda: "focus-2", ("Reopen it.",), ()) is None


def test_focus_mode_is_available_on_every_distinct_stuck_screen() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    screens = tuple(_observation(f"obs_{index}", f"{index:016x}") for index in range(1, 4))
    tracker.start(screens[0])

    focuses = []
    for screen_index, screen in enumerate(screens):
        for attempt_index in range(3):
            candidate = proposal(
                ActionKind.TAP,
                proposal_id=f"proposal-{screen_index}-{attempt_index}",
                target_expression=f"control {screen_index} {attempt_index}",
                source_observation_id=screen.observation_id,
            )
            tracker.record_refusal(
                candidate,
                screen,
                ActionRefusal(
                    proposal_id=candidate.proposal_id,
                    code=ActionRefusalCode.INVALID_GROUNDING,
                    reason="The target was not accepted.",
                ),
            )
        focuses.append(
            tracker.pending_planning_focus(
                screen,
                lambda screen_index=screen_index: f"focus-{screen_index}",
                ("Continue the task.",),
                (),
            )
        )

    assert focuses[0] is not None
    assert focuses[1] is not None
    assert focuses[2] is not None
