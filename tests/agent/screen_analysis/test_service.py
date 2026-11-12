"""Offline tests for model-backed analysis and bounded screen reuse."""

import json

import pytest

from observations.factories import build_observation
from agent.planning.support import StubPlannerModel
from yewo.agent.screen_analysis import (
    ModelScreenAnalyzer,
    ScreenAnalysisError,
    ScreenElementConfidence,
    ScreenSemanticRole,
    ScreenWorkflowRole,
)
from yewo.models import ModelRole, TextModelContent
from yewo.agent.screen_analysis.prompt import screen_analyzer_system_prompt


def _output() -> dict[str, object]:
    return {
        "screen_summary": "Visible application form.",
        "elements": [
            {
                "analysis_key": "primary-field",
                "name": "Primary field",
                "kind": {"visual": "input_surface", "semantic": "text_field"},
                "state": "empty",
                "confidence": "high",
                "role_alternatives": [],
                "input_surface_key": "primary-input",
                "parent_key": None,
                "position_in_parent": None,
                "visual_cues": ["below Section heading", "above Secondary field"],
                "region_hint": {"left": 0.4, "top": 0.4, "right": 0.6, "bottom": 0.6},
            }
        ],
    }


def test_analyzer_prompt_keeps_visible_only_and_role_alternative_guards() -> None:
    prompt = screen_analyzer_system_prompt()

    assert "Classify only visible application elements" in prompt
    assert "Do not infer hidden, covered, off-screen, or purely behavioral properties" in prompt
    assert "Give role_alternatives only when that evidence supports each alternative" in prompt


def test_model_analyzer_uses_catalog_prompt_and_assigns_runtime_identity() -> None:
    model = StubPlannerModel(_output())
    identifiers = iter(("request-1", "analysis-1"))
    analyzer = ModelScreenAnalyzer(
        model,
        "screen-analyzer-primary",
        identifier_factory=lambda: next(identifiers),
    )

    analysis = analyzer.analyze(build_observation())

    assert analysis.analysis_id == "analysis-1"
    assert analysis.elements[0].element_ref == "element-1"
    assert analysis.elements[0].kind.visual.value == "input_surface"
    assert analysis.elements[0].confidence is ScreenElementConfidence.HIGH
    assert analysis.elements[0].region_hint is not None
    assert model.requests[0].role is ModelRole.SCREEN_ANALYZER
    prompt = model.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert "visible_requirements" in prompt.text
    assert '"name":"switch"' in prompt.text
    assert '"visual_kinds"' in prompt.text
    contract = model.requests[0].response_contract
    assert contract is not None
    definitions = contract.json_schema["$defs"]
    assert isinstance(definitions, dict)
    element_schema = definitions["ScreenElementHypothesis"]
    assert isinstance(element_schema, dict)
    properties = element_schema["properties"]
    assert isinstance(properties, dict)
    assert "evidence" not in properties
    context = model.requests[0].messages[1].content[1]
    assert isinstance(context, TextModelContent)
    assert json.loads(context.text.split("\n", 1)[1]) == {"platform": "android"}


def test_model_analyzer_preserves_language_independent_workflow_role() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list)
    elements[0].update(
        {
            "name": "Guardar",
            "kind": {"visual": "text_and_icon", "semantic": "button"},
            "input_surface_key": None,
            "workflow_role": "advance_stage",
        }
    )
    analyzer = ModelScreenAnalyzer(
        StubPlannerModel(output),
        "screen-analyzer-primary",
        identifier_factory=iter(("request-1", "analysis-1")).__next__,
    )

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[0].name == "Guardar"
    assert analysis.elements[0].workflow_role is ScreenWorkflowRole.ADVANCE_STAGE


def test_model_analyzer_rejects_element_text_above_the_record_limit() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0]["name"] = "n" * 300
    elements[0]["state"] = "s" * 300

    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    with pytest.raises(ScreenAnalysisError, match="invalid hypothesis"):
        analyzer.analyze(build_observation())


def test_model_analyzer_rejects_an_element_without_screenshot_geometry() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0].pop("region_hint")
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    with pytest.raises(ScreenAnalysisError, match="invalid hypothesis"):
        analyzer.analyze(build_observation())


def test_model_analyzer_maps_analysis_parent_keys_to_runtime_references() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list)
    elements.insert(
        0,
        {
            "analysis_key": "form",
            "name": "Form",
            "kind": {"visual": "container", "semantic": "content_region"},
            "state": None,
            "confidence": "high",
            "role_alternatives": [],
            "input_surface_key": None,
            "parent_key": None,
            "position_in_parent": None,
            "visual_cues": ["large bordered region"],
            "region_hint": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9},
        },
    )
    field = elements[1]
    assert isinstance(field, dict)
    field["parent_key"] = "form"
    field["position_in_parent"] = 1
    identifiers = iter(("request-1", "analysis-1"))
    analyzer = ModelScreenAnalyzer(
        StubPlannerModel(output),
        "screen-analyzer-primary",
        identifier_factory=lambda: next(identifiers),
    )

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[1].parent_ref == "element-1"
    assert analysis.elements[1].position_in_parent == 1


def test_model_analyzer_rejects_two_elements_claiming_one_input_surface() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    duplicate = dict(elements[0])
    duplicate["analysis_key"] = "secondary-field"
    duplicate["name"] = "Secondary field"
    elements.append(duplicate)
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    with pytest.raises(ScreenAnalysisError, match="invalid hypothesis"):
        analyzer.analyze(build_observation())


def test_model_analyzer_derives_a_missing_input_surface_key() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0]["input_surface_key"] = None

    analysis = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary").analyze(
        build_observation()
    )

    assert analysis.elements[0].input_surface_key == "primary-field"


def test_model_analyzer_discards_an_input_key_from_a_non_input_visual() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0].update(
        {
            "kind": {"visual": "container", "semantic": "content_region"},
            "input_surface_key": "incorrect-container-key",
        }
    )

    analysis = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary").analyze(
        build_observation()
    )

    assert analysis.elements[0].input_surface_key is None


def test_model_analyzer_safely_downgrades_uncertain_role_without_alternatives() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0]["confidence"] = "medium"
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[0].kind.semantic.value == "unknown"
    assert analysis.elements[0].role_alternatives == (ScreenSemanticRole.TEXT_FIELD,)


def test_model_analyzer_downgrades_a_contextual_row_without_a_parent() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0].update(
        {
            "name": "Destination",
            "kind": {"visual": "text", "semantic": "navigation_item"},
            "input_surface_key": None,
        }
    )
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[0].kind.semantic is ScreenSemanticRole.UNKNOWN
    assert analysis.elements[0].role_alternatives == (ScreenSemanticRole.NAVIGATION_ITEM,)
    assert analysis.elements[0].parent_ref is None


def test_model_analyzer_downgrades_a_contextual_row_with_an_absent_parent() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    elements[0].update(
        {
            "name": "Destination",
            "kind": {"visual": "text", "semantic": "navigation_item"},
            "input_surface_key": None,
            "parent_key": "unreported-navigation-container",
            "position_in_parent": 2,
        }
    )
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[0].kind.semantic is ScreenSemanticRole.UNKNOWN
    assert analysis.elements[0].role_alternatives == (ScreenSemanticRole.NAVIGATION_ITEM,)
    assert analysis.elements[0].parent_ref is None
    assert analysis.elements[0].position_in_parent is None


def test_model_analyzer_drops_a_duplicate_optional_sibling_position() -> None:
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    first_field = elements[0]
    first_field["parent_key"] = "form"
    first_field["position_in_parent"] = 1
    second_field = dict(first_field)
    second_field.update(
        {
            "analysis_key": "secondary-field",
            "name": "Secondary field",
            "input_surface_key": "secondary-input",
        }
    )
    elements[:0] = [
        {
            "analysis_key": "form",
            "name": "Form",
            "kind": {"visual": "container", "semantic": "content_region"},
            "state": None,
            "confidence": "high",
            "role_alternatives": [],
            "input_surface_key": None,
            "parent_key": None,
            "position_in_parent": None,
            "visual_cues": ["large bordered region"],
            "region_hint": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9},
        }
    ]
    elements.append(second_field)
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    analysis = analyzer.analyze(build_observation())

    assert analysis.elements[1].position_in_parent == 1
    assert analysis.elements[2].position_in_parent is None


def test_a_child_listed_before_its_parent_keeps_the_whole_screen() -> None:
    # Nothing asks the model to list parents first. Resolving a parent reference against the
    # elements kept so far dropped the reference while keeping the position that requires one,
    # which refused the record and discarded every element on the screen.
    output = _output()
    elements = output["elements"]
    assert isinstance(elements, list) and isinstance(elements[0], dict)
    child = elements[0]
    child["parent_key"] = "form"
    child["position_in_parent"] = 1
    elements.append(
        {
            "analysis_key": "form",
            "name": "Form",
            "kind": {"visual": "container", "semantic": "content_region"},
            "state": None,
            "confidence": "high",
            "role_alternatives": [],
            "input_surface_key": None,
            "parent_key": None,
            "position_in_parent": None,
            "visual_cues": ["large bordered region"],
            "region_hint": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9},
        }
    )
    analyzer = ModelScreenAnalyzer(StubPlannerModel(output), "screen-analyzer-primary")

    analysis = analyzer.analyze(build_observation())

    assert len(analysis.elements) == len(elements)
    assert analysis.elements[0].position_in_parent == 1
    assert analysis.elements[0].parent_ref == analysis.elements[-1].element_ref
