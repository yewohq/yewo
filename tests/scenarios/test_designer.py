"""Tests for model-backed semantic test design."""

from itertools import count
import json

from pydantic import ValidationError
import pytest

from yewo.models import ImageModelContent, ModelCallDisposition, ModelRole, TextModelContent
from yewo.project_config import CoverageCategory, ResolvedValue
from yewo.scenarios import (
    CategoryApplicability,
    TestDesignerError as DesignerError,
    TestDesignerOutput as DesignerOutput,
    TestDesignerService as DesignerService,
    ScenarioSurfaceKind,
    evaluate_testing_categories,
)

from agent.planning.support import StubPlannerModel
from agent.replay.support import evidence as discovery_evidence
from scenarios.test_bridge import _bridge, _journey, _procedure, _result


def _identifiers():
    values = count(1)
    return lambda: f"request_{next(values)}"


def _ready_inputs():
    procedure = _procedure()
    suite = _bridge(procedure=procedure, result=_result(procedure))
    return suite, procedure, discovery_evidence()


def _output(check: str, surface_id: str) -> dict[str, object]:
    return {
        "check_interpretations": [
            {
                "source_check_index": 0,
                "source_check": check,
                "category": "lifecycle",
                "scenario_type": "interruption",
                "execution_change": "lifecycle_interruption",
            }
        ],
        "category_interpretations": [
            {
                "category": "navigation",
                "surface_ids": [surface_id],
                "variations": [
                    {
                        "scenario_type": "happy_path",
                        "execution_change": "accepted_route",
                    },
                    {
                        "scenario_type": "recovery",
                        "execution_change": "recovery_attempt",
                    },
                ],
                "reason": "The observed step moves from one application state to another.",
            }
        ],
    }


def test_designer_translates_checks_and_semantic_surface_meaning() -> None:
    check = "The user's work must remain after the app restarts."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel(_output(check, suite.surfaces[0].surface_id))

    design = DesignerService(
        model,
        "test-designer",
        identifier_factory=_identifiers(),
    ).design(journey, suite, procedure, evidence)

    assert model.requests[0].role is ModelRole.TEST_DESIGNER
    assert model.requests[0].response_contract is not None
    assert model.requests[0].response_contract.name == "test-designer-output-v3"
    assert model.response_models == [DesignerOutput]
    assert model.validation_contexts[0] is not None
    assert design.concerns[0].source_check == check
    assert design.concerns[0].category is CoverageCategory.LIFECYCLE
    assert design.concerns[0].execution_change.value == "lifecycle_interruption"
    assert design.category_interpretations[0].category is CoverageCategory.NAVIGATION
    assert [
        item.execution_change.value for item in design.category_interpretations[0].variations
    ] == ["accepted_route", "recovery_attempt"]
    registry = evaluate_testing_categories(
        journey,
        suite,
        concerns=design.concerns,
        interpretations=design.category_interpretations,
    )
    navigation = next(
        item for item in registry.assessments if item.category is CoverageCategory.NAVIGATION
    )
    assert navigation.applicability is CategoryApplicability.APPLICABLE
    assert navigation.interpretation_ids == (design.category_interpretations[0].interpretation_id,)


def test_designer_rejects_consistency_check_without_evidence_surfaces() -> None:
    check = "The summary amount must match the entered amount."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel(
        {
            "check_interpretations": [
                {
                    "source_check_index": 0,
                    "source_check": check,
                    "category": "data_consistency",
                    "scenario_type": "happy_path",
                    "execution_change": "accepted_route",
                }
            ],
            "category_interpretations": [],
        }
    )

    with pytest.raises(DesignerError, match="invalid evidence mapping"):
        DesignerService(model, "test-designer").design(
            journey,
            suite,
            procedure,
            evidence,
        )


def test_designer_receives_two_redacted_discovery_screens() -> None:
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel({"check_interpretations": [], "category_interpretations": []})

    DesignerService(model, "test-designer").design(_journey(), suite, procedure, evidence)

    images = tuple(
        content
        for content in model.requests[0].messages[1].content
        if isinstance(content, ImageModelContent)
    )
    assert len(images) == 2


def test_designer_can_match_a_presentation_surface_to_a_screenshot_state() -> None:
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel({"check_interpretations": [], "category_interpretations": []})

    DesignerService(model, "test-designer").design(_journey(), suite, procedure, evidence)

    presentation = next(
        item for item in suite.surfaces if item.kind is ScenarioSurfaceKind.PRESENTATION
    )
    content = model.requests[0].messages[1].content
    context_content = content[0]
    assert isinstance(context_content, TextModelContent)
    context = json.loads(context_content.text)
    projected = next(
        item for item in context["surfaces"] if item["surface_id"] == presentation.surface_id
    )
    assert projected["state_ids"] == list(presentation.state_ids)
    assert any(
        isinstance(item, TextModelContent)
        and any(state_id in item.text for state_id in presentation.state_ids)
        for item in content[1:]
    )


def test_designer_sends_declared_value_metadata_without_its_value() -> None:
    secret = "a-secret-that-must-not-reach-test-designer"
    journey = _journey().model_copy(
        update={
            "declared_values": {
                "password": ResolvedValue(
                    name="password",
                    value=secret,
                    secret=True,
                )
            }
        }
    )
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel({"check_interpretations": [], "category_interpretations": []})

    DesignerService(model, "test-designer").design(journey, suite, procedure, evidence)

    serialized = model.requests[0].model_dump_json()
    assert "password" in serialized
    assert secret not in serialized


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda output: output["check_interpretations"][0].update(
                {"source_check": "A changed requirement."}
            ),
            "invalid evidence mapping",
        ),
        (
            lambda output: output["check_interpretations"][0].update(
                {"expected_outcome": "An invented product rule."}
            ),
            "invalid evidence mapping",
        ),
        (
            lambda output: output["check_interpretations"][0].pop("execution_change"),
            "invalid evidence mapping",
        ),
        (
            lambda output: output["category_interpretations"][0].update(
                {"surface_ids": ["invented_surface"]}
            ),
            "invalid evidence mapping",
        ),
        (
            lambda output: output.update({"check_interpretations": []}),
            "invalid evidence mapping",
        ),
    ],
)
def test_designer_rejects_untrusted_mappings(mutate, message: str) -> None:
    check = "The user's work must remain after the app restarts."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite, procedure, evidence = _ready_inputs()
    output = _output(check, suite.surfaces[0].surface_id)
    mutate(output)

    with pytest.raises(DesignerError, match=message):
        DesignerService(StubPlannerModel(output), "test-designer").design(
            journey, suite, procedure, evidence
        )


def test_designer_rejects_an_app_control_as_native_permission_evidence() -> None:
    suite, procedure, evidence = _ready_inputs()
    output = {
        "check_interpretations": [],
        "category_interpretations": [
            {
                "category": "permission",
                "surface_ids": [suite.surfaces[0].surface_id],
                "variations": [
                    {
                        "scenario_type": "negative",
                        "execution_change": "permission_decision",
                    }
                ],
                "reason": "The app offers an access choice.",
            }
        ],
    }

    with pytest.raises(DesignerError, match="invalid evidence mapping"):
        DesignerService(StubPlannerModel(output), "test-designer").design(
            _journey(), suite, procedure, evidence
        )


def test_permission_mapping_validation_can_correct_mixed_surface_kinds() -> None:
    suite, procedure, evidence = _ready_inputs()
    operation = suite.surfaces[0]
    system_interaction = suite.surfaces[1].model_copy(
        update={"kind": ScenarioSurfaceKind.SYSTEM_INTERACTION}
    )
    suite = suite.model_copy(
        update={"surfaces": (operation, system_interaction, *suite.surfaces[2:])}
    )
    output = {
        "check_interpretations": [],
        "category_interpretations": [
            {
                "category": "permission",
                "surface_ids": [operation.surface_id, system_interaction.surface_id],
                "variations": [
                    {
                        "scenario_type": "negative",
                        "execution_change": "permission_decision",
                    }
                ],
                "reason": "The native system prompt offers an access choice.",
            }
        ],
    }
    model = StubPlannerModel(output)

    with pytest.raises(DesignerError, match="invalid evidence mapping"):
        DesignerService(model, "test-designer").design(_journey(), suite, procedure, evidence)

    validation_context = model.validation_contexts[0]
    assert validation_context is not None
    with pytest.raises(ValidationError, match="native system-interaction"):
        DesignerOutput.model_validate(output, context=validation_context)

    permission = output["category_interpretations"][0]
    assert isinstance(permission, dict)
    permission["surface_ids"] = [system_interaction.surface_id]
    corrected = DesignerOutput.model_validate(output, context=validation_context)
    assert corrected.category_interpretations[0].surface_ids == (system_interaction.surface_id,)


def test_designer_reports_a_failed_model_call() -> None:
    suite, procedure, evidence = _ready_inputs()
    model = StubPlannerModel(None, disposition=ModelCallDisposition.PROVIDER_ERROR)

    with pytest.raises(DesignerError, match="configured planner call failed"):
        DesignerService(model, "test-designer").design(_journey(), suite, procedure, evidence)


def test_design_identity_is_stable_across_equivalent_calls() -> None:
    suite, procedure, evidence = _ready_inputs()
    output: dict[str, object] = {
        "check_interpretations": [],
        "category_interpretations": [],
    }

    first = DesignerService(StubPlannerModel(output), "test-designer").design(
        _journey(), suite, procedure, evidence
    )
    second = DesignerService(StubPlannerModel(output), "test-designer").design(
        _journey(), suite, procedure, evidence
    )

    assert first.design_id == second.design_id
