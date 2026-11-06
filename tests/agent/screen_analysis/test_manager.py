"""Bounded semantic-analysis cache behavior."""

from observations.factories import build_observation

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    CurrentScreenAnalysisManager,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.observations import Observation


class _ToggleAnalyzer:
    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        return CurrentScreenAnalysis(
            analysis_id="analysis-1",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.redacted_screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible settings screen.",
            elements=(
                ScreenElement(
                    element_ref="toggle-1",
                    name="Example toggle",
                    kind=ScreenElementKind(
                        visual=ScreenVisualKind.CONTROL_SURFACE,
                        semantic=ScreenSemanticRole.SWITCH,
                    ),
                    state="checked",
                    confidence=ScreenElementConfidence.HIGH,
                    evidence="A switch is visible.",
                ),
            ),
        )


def test_reused_structure_does_not_invent_toggle_state() -> None:
    manager = CurrentScreenAnalysisManager(
        _ToggleAnalyzer(),
        identifier_factory=lambda: "analysis-2",
    )
    before = build_observation()
    analysis = manager.analyze(before)
    assert analysis is not None
    after = before.model_copy(
        update={
            "observation_id": "observation-2",
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )
    proposal = ActionProposal(
        proposal_id="proposal-1",
        source_observation_id=before.observation_id,
        action=ActionKind.TAP,
        target_element_ref="toggle-1",
        expected_outcome="The toggle changes state.",
        reason="Change the visible toggle.",
    )
    outcome = ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.TAP,
        source_observation_id=before.observation_id,
        settled_observation_id=after.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The screen changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )

    manager.observe_outcome(proposal, before, after, outcome)
    rebound = manager.analyze(after)

    assert manager.required_model_requests(after) == 0
    assert rebound is not None
    assert rebound.elements[0].state is None


def test_request_reasons_follow_the_variant_cache_bound() -> None:
    manager = CurrentScreenAnalysisManager(_ToggleAnalyzer(), maximum_variants=1)
    first = build_observation()
    second = first.model_copy(
        update={
            "observation_id": "observation-2",
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )

    manager.invalidate(first, "first_reason")
    manager.invalidate(second, "second_reason")

    assert manager.request_reason(first) == "structural_variant_cache_miss"
    assert manager.request_reason(second) == "second_reason"
