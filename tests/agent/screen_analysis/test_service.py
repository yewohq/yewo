"""Offline tests for model-backed analysis and bounded screen reuse."""

import json

import pytest

from observations.factories import build_observation
from agent.planning.support import StubPlannerModel
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    CurrentScreenAnalysisManager,
    ModelScreenAnalyzer,
    ScreenAnalysisError,
    ScreenElementConfidence,
    ScreenSemanticRole,
    ScreenWorkflowRole,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    LiteralText,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import ModelRole, TextModelContent
from yewo.models import ModelInputError
from yewo.observations import Observation


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
    assert analysis.elements[0].evidence == "Visible input surface 'Primary field'."
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


class _CountingAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        self.calls += 1
        return CurrentScreenAnalysis(
            analysis_id=f"analysis-{self.calls}",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.redacted_screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
            elements=(),
        )


def test_manager_reuses_same_variant_and_reanalyzes_keyboard_variant() -> None:
    source = _CountingAnalyzer()
    manager = CurrentScreenAnalysisManager(
        source,
        maximum_variants=2,
        identifier_factory=lambda: "analysis-rebound",
    )
    first = build_observation()
    same = first.model_copy(
        update={
            "observation_id": "obs-2",
            "redacted_screenshot": first.redacted_screenshot.model_copy(
                update={"sha256": "b" * 64}
            ),
        }
    )
    keyboard = first.model_copy(update={"observation_id": "obs-3", "keyboard_visible": True})

    assert manager.required_model_requests(first) == 1
    first_analysis = manager.analyze(first)
    assert first_analysis is not None
    assert manager.required_model_requests(same) == 0
    same_analysis = manager.analyze(same)
    assert same_analysis is not None and same_analysis.observation_id == "obs-2"
    assert same_analysis.analysis_id == "analysis-rebound"
    assert manager.required_model_requests(keyboard) == 1
    manager.analyze(keyboard)
    assert source.calls == 2


class _FailingAnalyzer(_CountingAnalyzer):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self._failures = failures

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        self.calls += 1
        if self.calls <= self._failures:
            raise ScreenAnalysisError("Temporary analyzer failure.")
        return CurrentScreenAnalysis(
            analysis_id=f"analysis-{self.calls}",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.redacted_screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
            elements=(),
        )


class _InvalidInputAnalyzer(_CountingAnalyzer):
    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        del observation
        self.calls += 1
        raise ModelInputError("The analyzer request exceeds its input contract.")


def test_manager_can_be_configured_for_a_second_source_attempt() -> None:
    source = _FailingAnalyzer(failures=1)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 1
    assert manager.analyze(observation) is not None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 2


def test_manager_contains_model_input_failure_from_optional_analysis() -> None:
    source = _InvalidInputAnalyzer()
    manager = CurrentScreenAnalysisManager(source)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 1


def test_manager_stops_after_bounded_failures_for_one_variant() -> None:
    source = _FailingAnalyzer(failures=3)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 2


def test_manager_default_caches_failure_after_sources_bounded_repair() -> None:
    source = _FailingAnalyzer(failures=2)
    manager = CurrentScreenAnalysisManager(source)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert manager.analyze(observation) is None
    assert source.calls == 1


def test_manager_reanalyzes_after_verified_text_entry() -> None:
    model = StubPlannerModel(_output())
    identifiers = iter(("request-1", "analysis-1", "element-1", "analysis-rebound"))
    manager = CurrentScreenAnalysisManager(
        ModelScreenAnalyzer(
            model,
            "screen-analyzer-primary",
            identifier_factory=identifiers.__next__,
        ),
        identifier_factory=identifiers.__next__,
    )
    before = build_observation()
    analysis = manager.analyze(before)
    assert analysis is not None
    after = before.model_copy(
        update={
            "observation_id": "observation-after-type",
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
            "redacted_screenshot": before.redacted_screenshot.model_copy(
                update={"sha256": "b" * 64}
            ),
        }
    )
    candidate = ActionProposal(
        proposal_id="proposal-type",
        source_observation_id=before.observation_id,
        action=ActionKind.TYPE,
        target_element_ref=analysis.elements[0].element_ref,
        text=LiteralText(value="new value"),
        expected_outcome="Primary field contains the entered text.",
        reason="Enter the required value.",
    )
    manager.observe_outcome(candidate, before, after, _observed_outcome(candidate, after))

    assert manager.required_model_requests(after) == 1
    assert manager.request_reason(after) == "structural_change_after_type"
    assert len(model.requests) == 1


def test_manager_reanalyzes_after_keyboard_layout_or_scroll_changes() -> None:
    source = _CountingAnalyzer()
    manager = CurrentScreenAnalysisManager(source)
    before = build_observation()
    assert manager.analyze(before) is not None
    keyboard = before.model_copy(
        update={
            "observation_id": "observation-keyboard",
            "keyboard_visible": not before.keyboard_visible,
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )
    focus = ActionProposal(
        proposal_id="proposal-focus",
        source_observation_id=before.observation_id,
        action=ActionKind.FOCUS,
        target_expression="the visible field",
        expected_outcome="The field gains focus.",
        reason="Prepare the field for entry.",
    )
    manager.observe_outcome(focus, before, keyboard, _observed_outcome(focus, keyboard))

    assert manager.required_model_requests(keyboard) == 1
    assert manager.request_reason(keyboard) == "keyboard_layout_changed"


def test_failed_analysis_budget_does_not_poison_a_new_dynamic_state_variant() -> None:
    source = _FailingAnalyzer(failures=1)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    before = build_observation()
    assert manager.analyze(before) is None
    after = before.model_copy(
        update={
            "observation_id": "observation-after-type",
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )
    candidate = ActionProposal(
        proposal_id="proposal-type",
        source_observation_id=before.observation_id,
        action=ActionKind.TYPE,
        target_expression="the focused field",
        text=LiteralText(value="value"),
        expected_outcome="The field contains the entered text.",
        reason="Enter the value.",
    )
    manager.observe_outcome(candidate, before, after, _observed_outcome(candidate, after))

    assert manager.required_model_requests(after) == 1
    assert manager.analyze(after) is not None
    assert manager.required_model_requests(after) == 0
    assert source.calls == 2


def _observed_outcome(
    proposal: ActionProposal,
    after: Observation,
) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=f"outcome-{proposal.proposal_id}",
        authorization_id=f"authorization-{proposal.proposal_id}",
        receipt_id=f"receipt-{proposal.proposal_id}",
        action=proposal.action,
        source_observation_id=proposal.source_observation_id,
        settled_observation_id=after.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The settled observation contains action-outcome evidence.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
