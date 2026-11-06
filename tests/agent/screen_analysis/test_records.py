"""Offline contracts for observation-bound current-screen analysis."""

import json

import pytest
from pydantic import ValidationError

from observations.factories import build_observation
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementAffordance,
    ScreenElementKind,
    element_affordance,
    element_may_be_targeted,
    ScreenElementConfidence,
    ScreenSemanticRole,
    ScreenVisualKind,
    screen_analysis_context,
    screen_role_affordance,
    screen_element_catalog,
    resolve_element_expression,
)


def _kind(visual: ScreenVisualKind, semantic: ScreenSemanticRole) -> ScreenElementKind:
    return ScreenElementKind(visual=visual, semantic=semantic)


def _analysis(**updates: object) -> CurrentScreenAnalysis:
    observation = build_observation()
    values: dict[str, object] = {
        "analysis_id": "analysis-1",
        "observation_id": observation.observation_id,
        "screen_fingerprint": observation.screen_fingerprint.value,
        "screenshot_sha256": observation.redacted_screenshot.sha256,
        "platform_context": observation.platform_context,
        "keyboard_visible": observation.keyboard_visible,
        "orientation": observation.orientation,
        "elements": (
            ScreenElement(
                element_ref="primary-field",
                name="Primary field",
                kind=_kind(ScreenVisualKind.INPUT_SURFACE, ScreenSemanticRole.TEXT_FIELD),
                state="empty",
                confidence=ScreenElementConfidence.HIGH,
                evidence="An underlined input is visible.",
                visual_cues=("below Section heading", "above Secondary field"),
            ),
            ScreenElement(
                element_ref="section-heading",
                name="Section heading",
                kind=_kind(ScreenVisualKind.TEXT, ScreenSemanticRole.LABEL),
                confidence=ScreenElementConfidence.MEDIUM,
                evidence="Visible text has no input affordance.",
            ),
        ),
    }
    values.update(updates)
    return CurrentScreenAnalysis.model_validate(values)


def test_analysis_matches_only_its_exact_observation() -> None:
    observation = build_observation()
    analysis = _analysis()

    assert analysis.matches(observation)
    assert not analysis.matches(observation.model_copy(update={"observation_id": "obs-2"}))


def test_element_references_are_unique_per_analysis() -> None:
    analysis = _analysis()

    with pytest.raises(ValidationError, match="element reference must be unique"):
        _analysis(
            elements=(
                analysis.elements[0],
                analysis.elements[1].model_copy(update={"element_ref": "primary-field"}),
            )
        )


def test_visual_cues_and_grounding_expression_are_bounded() -> None:
    element = _analysis().elements[0]
    normalized = ScreenElement.model_validate(
        {**element.model_dump(), "visual_cues": ("x" * 300,)}
    )
    crowded = normalized.model_copy(
        update={"visual_cues": tuple("x" * 200 for _ in range(20))}
    )

    expression = resolve_element_expression(
        _analysis(elements=(crowded,)),
        crowded.element_ref,
    )

    assert normalized.visual_cues == ("x" * 200,)
    assert expression is not None
    assert len(expression) == 1_000


def test_compact_context_derives_affordance_and_omits_evidence_prose() -> None:
    context = screen_analysis_context(_analysis())

    assert context["analysis_id"] == "analysis-1"
    elements = context["elements"]
    assert isinstance(elements, tuple)
    primary_field, section_heading = elements
    assert isinstance(primary_field, dict) and isinstance(section_heading, dict)
    assert primary_field["kind"] == {
        "visual": "input_surface",
        "semantic": "text_field",
    }
    assert primary_field["affordance"] == "interactive"
    assert section_heading["affordance"] == "non_interactive"
    assert "evidence" not in primary_field
    assert "matches_proposal_target" not in primary_field


def test_unknown_kind_never_implies_an_interactive_affordance() -> None:
    assert (
        screen_role_affordance(ScreenSemanticRole.UNKNOWN)
        is ScreenElementAffordance.CONTEXT_DEPENDENT
    )


def test_input_surface_requires_unique_visible_surface() -> None:
    catalog = screen_element_catalog()

    assert catalog.role_definition(ScreenSemanticRole.TEXT_FIELD).visible_requirements
    input_surface = catalog.visual_definition(ScreenVisualKind.INPUT_SURFACE)
    assert input_surface.requires_unique_surface
    assert "uniquely associated" in " ".join(input_surface.visible_requirements)


def test_grounding_expression_omits_coordinates() -> None:
    expression = resolve_element_expression(_analysis(), "primary-field")

    assert expression is not None
    assert "0.4" not in expression


def test_noninteractive_element_reference_cannot_reach_grounding() -> None:
    with pytest.raises(ValueError, match="does not identify an interactive element"):
        resolve_element_expression(_analysis(), "section-heading")


def test_the_shared_inventory_carries_no_coordinates() -> None:
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id="observation-1",
        screen_fingerprint="1111111111111111",
        screenshot_sha256="a" * 64,
        platform_context=".Editor",
        keyboard_visible=False,
        orientation=None,
        elements=(
            ScreenElement(
                element_ref="element-1",
                name="Numeric input",
                kind=_kind(ScreenVisualKind.INPUT_SURFACE, ScreenSemanticRole.TEXT_FIELD),
                state="10",
                confidence=ScreenElementConfidence.HIGH,
                evidence="An underlined field holds a number.",
            ),
        ),
    )

    context = screen_analysis_context(analysis)

    assert "region" not in json.dumps(context)
    elements = context["elements"]
    assert isinstance(elements, tuple)
    assert set(elements[0]) == {
        "element_ref",
        "name",
        "kind",
        "affordance",
        "state",
        "confidence",
    }


def _element(
    semantic: ScreenSemanticRole,
    *,
    visual: ScreenVisualKind = ScreenVisualKind.TEXT,
    **updates: object,
) -> ScreenElement:
    values: dict[str, object] = {
        "element_ref": "element-1",
        "name": "Destination",
        "kind": _kind(visual, semantic),
        "confidence": ScreenElementConfidence.MEDIUM,
        "evidence": "Dark text on a white drawer surface.",
    }
    values.update(updates)
    return ScreenElement.model_validate(values)


def test_an_analyzer_hedge_toward_interaction_keeps_the_element_targetable() -> None:
    hedged = _element(
        ScreenSemanticRole.TEXT,
        role_alternatives=(
            ScreenSemanticRole.NAVIGATION_ITEM,
            ScreenSemanticRole.LIST_ITEM,
        ),
    )

    assert element_affordance(hedged) is ScreenElementAffordance.CONTEXT_DEPENDENT
    assert element_may_be_targeted(hedged)


def test_a_committed_label_stays_blocked_so_text_never_becomes_a_field() -> None:
    committed = _element(ScreenSemanticRole.LABEL, name="Section label:")

    assert element_affordance(committed) is ScreenElementAffordance.NON_INTERACTIVE
    assert not element_may_be_targeted(committed)


def test_a_hedge_toward_other_inert_kinds_does_not_unlock_targeting() -> None:
    inert = _element(
        ScreenSemanticRole.TEXT,
        role_alternatives=(ScreenSemanticRole.HEADING,),
    )

    assert not element_may_be_targeted(inert)


def test_an_interactive_role_with_an_inert_alternative_is_context_dependent() -> None:
    uncertain = _element(
        ScreenSemanticRole.TEXT_FIELD,
        visual=ScreenVisualKind.INPUT_SURFACE,
        role_alternatives=(ScreenSemanticRole.HEADING,),
    )

    assert element_affordance(uncertain) is ScreenElementAffordance.CONTEXT_DEPENDENT
    assert element_may_be_targeted(uncertain)


def test_navigation_text_is_described_to_grounding_by_its_visible_surface() -> None:
    element = _element(ScreenSemanticRole.NAVIGATION_ITEM)

    expression = resolve_element_expression(
        _analysis(elements=(element,)),
        "element-1",
    )

    assert expression is not None
    assert expression.startswith("the visible text 'Destination' in its navigation row")
    assert "navigation item named" not in expression


def test_semantic_role_rejects_an_impossible_visual_surface() -> None:
    with pytest.raises(ValidationError, match="not an allowed visual kind"):
        _element(ScreenSemanticRole.TEXT_FIELD, visual=ScreenVisualKind.ICON)
