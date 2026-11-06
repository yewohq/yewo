"""Bound repeated outcomes whose intended effect was not established."""

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionRefusalCode,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import proposal
from agent.runtime.test_agent import _observation


def _ambiguous_focus(screen) -> ActionOutcome:
    return ActionOutcome(
        outcome_id="outcome-ambiguous",
        authorization_id="authorization-ambiguous",
        receipt_id="receipt-ambiguous",
        action=ActionKind.FOCUS,
        source_observation_id=screen.observation_id,
        settled_observation_id=screen.observation_id,
        disposition=ActionOutcomeDisposition.AMBIGUOUS,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The intended target effect was not established.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def test_repeated_ambiguous_interaction_requests_focus_and_prohibits_repeat() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    screen = _observation("obs_1", "0000000000000000")
    candidate = proposal(
        ActionKind.FOCUS,
        proposal_id="proposal-focus",
        source_observation_id=screen.observation_id,
        target_expression="the unverified input",
    )
    outcome = _ambiguous_focus(screen)
    tracker.start(screen)

    tracker.record_outcome(candidate, screen, screen, outcome)
    assert tracker.refusal_for(candidate, screen) is None
    tracker.record_outcome(candidate, screen, screen, outcome)

    focus = tracker.pending_planning_focus(
        screen,
        lambda: "focus-ambiguous",
        ("Continue through another route.",),
        (),
    )
    assert focus is not None
    assert focus.prohibited_interaction_intents
    refusal = tracker.refusal_for(candidate, screen)
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS
    assert any("target as unverified" in fact for fact in tracker.known_facts())
    assert tracker.screen_facts(screen)[-1].supersedes_recovery_guidance


def test_observed_interaction_clears_prior_ambiguous_attempt() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    screen = _observation("obs_1", "0000000000000000")
    destination = _observation("obs_2", "1111111111111111", offset_seconds=1)
    candidate = proposal(
        ActionKind.FOCUS,
        proposal_id="proposal-focus",
        source_observation_id=screen.observation_id,
        target_expression="the verified input",
    )
    ambiguous = _ambiguous_focus(screen)
    observed = ambiguous.model_copy(
        update={
            "outcome_id": "outcome-observed",
            "settled_observation_id": destination.observation_id,
            "disposition": ActionOutcomeDisposition.OBSERVED,
            "signals": frozenset({ActionOutcomeSignal.FOCUS_CHANGED}),
            "reason": "The target accepted focus.",
        }
    )
    tracker.start(screen)

    tracker.record_outcome(candidate, screen, screen, ambiguous)
    tracker.record_outcome(candidate, screen, destination, observed)
    tracker.record_outcome(candidate, screen, screen, ambiguous)

    assert tracker.refusal_for(candidate, screen) is None
