"""Deterministic builders for proposal verification tests."""

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
)
from yewo.agent.planning.history import PlanningHistoryEntry, planning_screen_reference
from yewo.agent.planning.records import PlanningRequest
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.artifacts import ArtifactRetention

from agent.discovery.support import observation


def save_proposal() -> ActionProposal:
    return ActionProposal(
        proposal_id="proposal_1",
        source_observation_id="obs_1",
        action=ActionKind.TAP,
        target_expression="the visible Save button",
        expected_outcome="The editor saves and closes.",
        reason="Save the current editor.",
    )


def discovery_request() -> PlanningRequest:
    source = observation("dropdown_closed", "1111111111111111")
    destination = observation("dropdown_open", "2222222222222222")
    dropdown_proposal = ActionProposal(
        proposal_id="dropdown_proposal",
        source_observation_id=source.observation_id,
        action=ActionKind.TAP,
        target_expression="the Category dropdown",
        expected_outcome="The available record types become visible.",
        reason="Inspect the dropdown choices.",
    )
    dropdown_outcome = ActionOutcome(
        outcome_id="dropdown_outcome",
        authorization_id="dropdown_authorization",
        receipt_id="dropdown_receipt",
        action=ActionKind.TAP,
        source_observation_id=source.observation_id,
        settled_observation_id=destination.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The dropdown opened.",
        retention=ArtifactRetention.RUN,
    )
    return PlanningRequest(
        observation=observation("obs_1", "1111111111111111"),
        objective="Discover how to create a record.",
        open_work=("Cover every relevant visible field.",),
        completion_conditions=("The saved record can be reopened.",),
        available_actions=frozenset({ActionKind.TAP}),
        history=(
            PlanningHistoryEntry(
                proposal=dropdown_proposal,
                source_screen=planning_screen_reference(source),
                destination_screen=planning_screen_reference(destination),
                outcome=dropdown_outcome,
            ),
        ),
    )


def form_screen_analysis(request: PlanningRequest) -> CurrentScreenAnalysis:
    observed = request.observation
    return CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=observed.observation_id,
        screenshot_sha256=observed.screenshot.sha256,
        presentation=observed.presentation_key,
        screen_summary="Visible workflow form.",
        elements=(
            ScreenElement(
                element_ref="section-heading",
                name="Section heading",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.TEXT,
                    semantic=ScreenSemanticRole.LABEL,
                ),
                confidence=ScreenElementConfidence.HIGH,
            ),
            ScreenElement(
                element_ref="primary-field",
                name="Primary field",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.INPUT_SURFACE,
                    semantic=ScreenSemanticRole.TEXT_FIELD,
                ),
                confidence=ScreenElementConfidence.HIGH,
            ),
        ),
    )
