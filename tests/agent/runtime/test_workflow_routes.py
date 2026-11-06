"""Workflow-route exhaustion and task-level progress boundaries."""

from pathlib import Path

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusalCode,
)
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.planning.records import ProgressRouteAttempt
from yewo.agent.runtime.interaction_intent import interaction_intent_key
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.agent.runtime.verification import PreDispatchVerification
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import proposal
from agent.runtime.test_agent import _observation


def test_repeated_complete_workflow_route_exhausts_each_route_interaction() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    first = _observation("screen-a", "0000000000000000").model_copy(
        update={"platform_context": "collection"}
    )
    second = _observation("screen-b", "1111111111111111").model_copy(
        update={"platform_context": "editor"}
    )
    tracker.start(first)
    open_editor = proposal(
        ActionKind.TAP,
        proposal_id="open-editor",
        source_observation_id=first.observation_id,
        target_element_ref="open-control",
        target_expression=None,
    )
    close_editor = proposal(
        ActionKind.TAP,
        proposal_id="close-editor",
        source_observation_id=second.observation_id,
        target_element_ref="close-control",
        target_expression=None,
    )
    for index, (candidate, before, after) in enumerate(
        (
            (open_editor, first, second),
            (close_editor, second, first),
            (open_editor, first, second),
            (close_editor, second, first),
        ),
        start=1,
    ):
        tracker.record_route(
            candidate,
            before,
            after,
            ProgressRouteAttempt(
                step_index=index,
                work_item="Reach the next workflow stage.",
                source_context=before.platform_context,
                source_screen_fingerprint=before.screen_fingerprint.value,
                action=candidate.action,
                target=candidate.target_expression,
                interaction_intent_key=interaction_intent_key(candidate),
                destination_context=after.platform_context,
                destination_screen_fingerprint=after.screen_fingerprint.value,
                outcome="observed",
            ),
            task_advanced=False,
        )
    rewritten = open_editor.model_copy(
        update={
            "proposal_id": "rewritten-open-editor",
            "target_expression": "a differently worded description",
        }
    )

    refusal = tracker.refusal_for(rewritten, first)

    assert refusal is None
    focus = tracker.pending_planning_focus(
        first,
        lambda: "focus-route",
        ("Reach the next workflow stage.",),
        (),
    )
    assert focus is not None
    assert interaction_intent_key(open_editor) in focus.prohibited_interaction_intents


def test_task_advancement_clears_only_its_work_item_route_state() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    first = _observation("screen-a", "0000000000000000").model_copy(
        update={"platform_context": "collection"}
    )
    second = _observation("screen-b", "1111111111111111").model_copy(
        update={"platform_context": "editor"}
    )
    tracker.start(first)
    open_editor = proposal(
        ActionKind.TAP,
        proposal_id="open-editor",
        source_observation_id=first.observation_id,
        target_expression="the add control",
    )
    close_editor = proposal(
        ActionKind.BACK,
        proposal_id="close-editor",
        source_observation_id=second.observation_id,
    )
    route_steps = (
        (open_editor, first, second),
        (close_editor, second, first),
        (open_editor, first, second),
        (close_editor, second, first),
    )
    for index, (candidate, before, after) in enumerate(route_steps, start=1):
        tracker.record_route(
            candidate,
            before,
            after,
            ProgressRouteAttempt(
                step_index=index,
                work_item="Add item A.",
                source_context=before.platform_context,
                source_screen_fingerprint=before.screen_fingerprint.value,
                action=candidate.action,
                target=candidate.target_expression,
                interaction_intent_key=interaction_intent_key(candidate),
                destination_context=after.platform_context,
                destination_screen_fingerprint=after.screen_fingerprint.value,
                outcome="observed",
            ),
            task_advanced=False,
        )
    assert tracker.workflow_route_is_exhausted(open_editor, first, "Add item A.")
    assert not tracker.workflow_route_is_exhausted(open_editor, first, "Add item B.")

    tracker.record_route(
        open_editor,
        first,
        second,
        ProgressRouteAttempt(
            step_index=5,
            work_item="Add item B.",
            source_context=first.platform_context,
            source_screen_fingerprint=first.screen_fingerprint.value,
            action=open_editor.action,
            target=open_editor.target_expression,
            interaction_intent_key=interaction_intent_key(open_editor),
            destination_context=second.platform_context,
            destination_screen_fingerprint=second.screen_fingerprint.value,
            outcome="observed",
        ),
        task_advanced=True,
    )

    assert tracker.workflow_route_is_exhausted(open_editor, first, "Add item A.")
    assert not tracker.workflow_route_is_exhausted(open_editor, first, "Add item B.")


def test_verifier_applies_route_exhaustion_to_the_exact_work_item(tmp_path: Path) -> None:
    first = _observation("screen-a", "0000000000000000").model_copy(
        update={"platform_context": "collection"}
    )
    second = _observation("screen-b", "1111111111111111").model_copy(
        update={"platform_context": "editor"}
    )
    open_editor = proposal(
        ActionKind.TAP,
        proposal_id="open-editor",
        source_observation_id=first.observation_id,
        target_expression="the add control",
    )
    close_editor = proposal(
        ActionKind.BACK,
        proposal_id="close-editor",
        source_observation_id=second.observation_id,
    )
    ledger = RunProgressLedger(
        tmp_path,
        screen_comparison=ScreenComparisonPolicy(),
        identifier_factory=iter(f"event-{index}" for index in range(10)).__next__,
    )
    ledger.start(("Add item A.", "Add item B."), first, None)
    for index, (candidate, before, after) in enumerate(
        (
            (open_editor, first, second),
            (close_editor, second, first),
            (open_editor, first, second),
            (close_editor, second, first),
        ),
        start=1,
    ):
        ledger.record_route(
            index,
            candidate,
            "Add item A.",
            before,
            after,
            _outcome(
                candidate,
                after.observation_id,
                ActionOutcomeDisposition.OBSERVED,
                frozenset({ActionOutcomeSignal.CONTEXT_CHANGED}),
            ),
        )

    refusal = ledger.record_verification(
        PreDispatchVerification(
            progress_assessed=True,
            proposal_work_item="Add item A.",
        ),
        first,
        open_editor,
    )
    allowed = ledger.record_verification(
        PreDispatchVerification(
            progress_assessed=True,
            proposal_work_item="Add item B.",
        ),
        first,
        open_editor,
    )

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.REPEATED_NO_PROGRESS
    assert allowed is None

def test_pixel_change_without_task_advancement_does_not_clear_screen_exhaustion() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
    screen = _observation("screen", "0000000000000000")
    changed = _observation("changed", "1111111111111111")
    tracker.start(screen)
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"no-progress-{index}",
            target_expression=f"control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_outcome(
            candidate,
            screen,
            screen,
            _outcome(
                candidate,
                screen.observation_id,
                ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
                frozenset(),
            ),
        )
    cosmetic = proposal(
        ActionKind.TAP,
        proposal_id="cosmetic-change",
        target_expression="a cosmetic toggle",
        source_observation_id=screen.observation_id,
    )
    tracker.record_outcome(
        cosmetic,
        screen,
        changed,
        _outcome(
            cosmetic,
            changed.observation_id,
            ActionOutcomeDisposition.OBSERVED,
            frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        ),
        task_advanced=False,
    )
    fresh = proposal(
        ActionKind.TAP,
        proposal_id="fresh-after-cosmetic-change",
        target_expression="another control",
        source_observation_id=screen.observation_id,
    )

    assert tracker.refusal_for(fresh, screen) is not None


def _outcome(
    candidate: ActionProposal,
    settled_observation_id: str,
    disposition: ActionOutcomeDisposition,
    signals: frozenset[ActionOutcomeSignal],
) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=f"outcome-{candidate.proposal_id}",
        authorization_id=f"authorization-{candidate.proposal_id}",
        receipt_id=f"receipt-{candidate.proposal_id}",
        action=candidate.action,
        source_observation_id=candidate.source_observation_id,
        settled_observation_id=settled_observation_id,
        disposition=disposition,
        signals=signals,
        reason="The runtime recorded the bounded outcome.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
