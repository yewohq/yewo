"""Screen ownership and rendering tests for bounded planner history."""

import json

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.agent.planning.history import (
    RECENT_PLANNING_HISTORY_LIMIT,
    planning_history_context,
    planning_history_summary,
    trusted_history_content,
    EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT,
    EARLIER_HISTORY_SUMMARY_LIMIT,
    PlanningHistoryEntry,
    build_planning_history_entry,
)
from yewo.agent.planning.records import MAXIMUM_PLANNING_HISTORY_ENTRIES
from yewo.artifacts import ArtifactRetention

from agent.actuation.support import proposal
from agent.planning.support import planning_request
from agent.runtime.test_agent import _observation


def _refused_history(target_expression: str, index: int) -> PlanningHistoryEntry:
    action = ActionProposal(
        proposal_id=f"proposal-{index}",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression=target_expression,
        expected_outcome="The target opens.",
        reason="Inspect this target.",
    )
    return PlanningHistoryEntry(
        proposal=action,
        refusal=ActionRefusal(
            proposal_id=action.proposal_id,
            code=ActionRefusalCode.ACTION_NOT_ALLOWED,
            reason="This action is unavailable.",
        ),
    )


def _outcome(
    *,
    source_observation_id: str,
    settled_observation_id: str,
    outcome_id: str = "outcome-1",
) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=outcome_id,
        authorization_id=f"authorization-{outcome_id}",
        receipt_id=f"receipt-{outcome_id}",
        action=ActionKind.TAP,
        source_observation_id=source_observation_id,
        settled_observation_id=settled_observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The screen changed.",
        retention=ArtifactRetention.RUN,
    )


def test_recent_history_renders_one_ordered_screen_transition() -> None:
    before = _observation("obs_1", "0000000000000000")
    after = _observation("obs_2", "1111111111111111")
    action = proposal(
        ActionKind.TAP,
        source_observation_id=before.observation_id,
        target_expression="the visible Settings row",
        reason="The row is the visible route to settings.",
    )
    entry = build_planning_history_entry(
        action,
        _outcome(
            source_observation_id=before.observation_id,
            settled_observation_id=after.observation_id,
        ),
        None,
        before,
        after,
    )

    context = planning_history_context(entry)

    assert context["source_screen"] == {
        "platform_context": before.platform_context,
        "keyboard_visible": False,
    }
    assert context["destination_screen"] == {
        "platform_context": after.platform_context,
        "keyboard_visible": False,
    }
    assert "planner_decision_summary" not in context
    feedback = context["feedback"]
    assert isinstance(feedback, dict)
    assert "reason" not in feedback
    assert not [key for key, value in context.items() if value is None]


def test_older_history_does_not_merge_the_same_action_across_source_screens() -> None:
    first = _observation("obs_1", "0000000000000000")
    second = _observation("obs_2", "1111111111111111")
    destination = _observation("obs_3", "2222222222222222")
    entries: list[PlanningHistoryEntry] = []
    for index, source in enumerate((first, second), start=1):
        action = proposal(
            ActionKind.TAP,
            proposal_id=f"proposal-{index}",
            source_observation_id=source.observation_id,
            target_expression="the Settings row",
        )
        entries.append(
            build_planning_history_entry(
                action,
                _outcome(
                    source_observation_id=source.observation_id,
                    settled_observation_id=destination.observation_id,
                    outcome_id=f"outcome-{index}",
                ),
                None,
                source,
                destination,
            )
        )

    summary = planning_history_summary(tuple(entries))

    summary_entries = summary["entries"]
    assert len(summary_entries) == 2
    assert [item["count"] for item in summary_entries] == [1, 1]
    source_screens = [item["source_screen"] for item in summary_entries]
    assert all(isinstance(screen, dict) for screen in source_screens)
    assert [
        screen["screen_fingerprint"] for screen in source_screens if isinstance(screen, dict)
    ] == [
        "0000000000000000",
        "1111111111111111",
    ]
    assert summary["omitted_groups"] == 0


def test_older_history_summary_reports_item_limit_omissions() -> None:
    history = tuple(
        _refused_history(f"target {index}", index)
        for index in range(EARLIER_HISTORY_SUMMARY_LIMIT + 5)
    )

    summary = planning_history_summary(history)
    summary_entries = summary["entries"]

    assert len(summary_entries) == EARLIER_HISTORY_SUMMARY_LIMIT
    assert summary_entries[0]["target_expression"] == "target 5"
    assert summary_entries[-1]["target_expression"] == "target 24"
    assert summary["omitted_groups"] == 5


def test_older_history_summary_reports_character_limit_omissions() -> None:
    history = tuple(_refused_history(f"target {index} " + "x" * 900, index) for index in range(20))

    summary = planning_history_summary(history)
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary_entries = summary["entries"]

    assert len(encoded) <= EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT
    assert len(summary_entries) < EARLIER_HISTORY_SUMMARY_LIMIT
    assert summary["omitted_groups"] > 0


def test_trusted_history_keeps_recent_steps_and_summarises_the_rest() -> None:
    candidate = proposal(
        ActionKind.TAP,
        proposal_id="proposal-refused",
        target_expression="the Save button",
        source_observation_id="obs_0",
    )
    entry = PlanningHistoryEntry(
        proposal=candidate,
        refusal=ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.KEYBOARD_RULE,
            reason="Dismiss the visible keyboard before pointer actions.",
        ),
    )

    blocks = trusted_history_content((entry,) * 110)

    # One summary block, then one byte-stable block per recent step. Without the
    # window the planner prompt grows by one block for every step of the run.
    assert len(blocks) == RECENT_PLANNING_HISTORY_LIMIT + 1
    assert blocks[0].text.startswith("Earlier history summary:")
    assert all(item.text.startswith("Trusted history event:") for item in blocks[1:])


def test_maximum_planning_history_validates_and_compacts() -> None:
    history = tuple(
        _refused_history(f"target {index}", index)
        for index in range(MAXIMUM_PLANNING_HISTORY_ENTRIES)
    )

    request = planning_request(history=history)
    blocks = trusted_history_content(request.history)

    assert len(request.history) == MAXIMUM_PLANNING_HISTORY_ENTRIES
    assert len(blocks) == RECENT_PLANNING_HISTORY_LIMIT + 1


def test_trusted_history_omits_the_summary_for_a_short_run() -> None:
    candidate = proposal(ActionKind.TAP, proposal_id="p1", source_observation_id="obs_0")
    entry = PlanningHistoryEntry(
        proposal=candidate,
        refusal=ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.KEYBOARD_RULE,
            reason="Dismiss the visible keyboard before pointer actions.",
        ),
    )

    blocks = trusted_history_content((entry,) * 3)

    assert len(blocks) == 3
    assert all(item.text.startswith("Trusted history event:") for item in blocks)


def test_history_reports_an_exhausted_scroll_surface_to_the_planner() -> None:
    before = _observation("obs_1", "0000000000000000")
    after = _observation("obs_2", "0000000000000000")
    action = proposal(
        ActionKind.SCROLL,
        source_observation_id=before.observation_id,
        reason="More restaurants should be below the visible cards.",
    )
    outcome = _outcome(
        source_observation_id=before.observation_id,
        settled_observation_id=after.observation_id,
    ).model_copy(update={"action": ActionKind.SCROLL, "scroll_exhausted": True})

    entry = build_planning_history_entry(action, outcome, None, before, after)
    feedback = planning_history_context(entry)["feedback"]

    assert isinstance(feedback, dict)
    assert feedback["scroll_exhausted"] is True


def test_history_omits_scroll_evidence_the_platform_did_not_supply() -> None:
    before = _observation("obs_1", "0000000000000000")
    after = _observation("obs_2", "1111111111111111")
    action = proposal(ActionKind.TAP, source_observation_id=before.observation_id)
    outcome = _outcome(
        source_observation_id=before.observation_id,
        settled_observation_id=after.observation_id,
    )

    entry = build_planning_history_entry(action, outcome, None, before, after)
    feedback = planning_history_context(entry)["feedback"]

    assert isinstance(feedback, dict)
    assert "scroll_exhausted" not in feedback
