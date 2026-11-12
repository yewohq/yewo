"""What the planner refuses: unusable responses, undeclared values, mismatched evidence."""

import pytest
from pydantic import ValidationError

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
)
from yewo.agent.planning.prompt import planner_action_catalog
from yewo.agent.planning.planner import (
    PlanningHistoryEntry,
    TaskPlanningError,
)
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.artifacts import ArtifactRetention
from yewo.models import (
    ModelCallDisposition,
)
from agent.planning.support import (
    StubPlannerModel,
    planner as _planner,
    planning_request as _planning_request,
    proposal_output as _proposal_output,
)


def test_provider_failure_is_distinct_from_cannot_progress() -> None:
    failed_model = StubPlannerModel(
        None,
        disposition=ModelCallDisposition.PROVIDER_ERROR,
    )

    with pytest.raises(TaskPlanningError) as failure:
        _planner(failed_model).plan(_planning_request())

    assert failure.value.disposition is ModelCallDisposition.PROVIDER_ERROR

    stopped_model = StubPlannerModel(_proposal_output(ActionKind.CANNOT_PROGRESS))
    proposal = _planner(stopped_model).plan(_planning_request())
    assert proposal.action is ActionKind.CANNOT_PROGRESS


def test_rejects_an_undeclared_value_name() -> None:
    output = {
        "action": "type",
        "text": {"source": "declared_value", "value_name": "unknown_password"},
        "expected_outcome": "The password field contains the declared value.",
        "reason": "The password field must be completed.",
    }

    with pytest.raises(TaskPlanningError, match="unavailable declared value"):
        _planner(StubPlannerModel(output)).plan(_planning_request())


@pytest.mark.parametrize(
    "output",
    [
        _proposal_output(ActionKind.WAIT),
        {
            "action": "tap",
        },
    ],
)
def test_rejects_a_response_that_is_not_usable_for_the_request(
    output: dict[str, object],
) -> None:
    model = StubPlannerModel(output)
    request = _planning_request(available_actions=frozenset({ActionKind.TAP}))

    with pytest.raises(TaskPlanningError):
        _planner(model).plan(request)


@pytest.mark.parametrize("extra_field", ["workflow_mode", "declared_values"])
def test_request_rejects_workflow_switches_and_value_bodies(extra_field: str) -> None:
    values = _planning_request().model_dump()
    values[extra_field] = "not allowed"

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate(values)


def test_history_rejects_an_outcome_from_another_observation() -> None:
    proposal = ActionProposal.model_validate(
        {
            "proposal_id": "proposal-1",
            "source_observation_id": "obs_1",
            **_proposal_output(),
        }
    )
    outcome = ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.TAP,
        source_observation_id="obs_other",
        settled_observation_id="obs_2",
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The screen changed.",
        retention=ArtifactRetention.RUN,
    )

    with pytest.raises(ValidationError, match="proposal observation"):
        PlanningHistoryEntry(proposal=proposal, outcome=outcome)


def test_action_catalog_covers_every_runtime_action_from_its_definition() -> None:
    # The catalogue is derived, so a new action reaches the planner without a prompt edit.
    catalog = {entry["action"]: entry for entry in planner_action_catalog(ActionKind)}

    assert set(catalog) == {action.value for action in ActionKind}
    assert catalog["clear"]["optional_fields"] == []
    assert catalog["clear"]["requires_visible_keyboard"] is True
    assert catalog["long_press"]["required_fields"] == [
        "duration_seconds",
        "expected_outcome",
        "reason",
    ]
    # A region target and a destination target both bar the recovery protocols.
    assert catalog["scroll"]["supports_grounding_correction"] is False
    assert catalog["drag"]["supports_interaction_refinement"] is False
    assert catalog["tap"]["supports_interaction_refinement"] is True
