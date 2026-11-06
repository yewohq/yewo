"""Grounding-correction contract: retried with guidance, not a run-ending error."""

from typing import Any

import pytest

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
)
from yewo.agent.planning.history import correctable_grounding_proposals
from yewo.agent.planning.planner import (
    GroundedTargetPoint,
    PlanningHistoryEntry,
    TaskPlanningError,
    _grounding_correction_error,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.grounding import PixelPoint

from agent.planning.support import (
    history_contexts as _history_contexts,
    StubPlannerModel,
    planner as _planner,
    planning_request as _planning_request,
    proposal_output as _proposal_output,
)


def test_planner_receives_target_point_and_accepts_one_refined_correction() -> None:
    failed_proposal = ActionProposal(
        proposal_id="proposal-failed",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the Edit Record item",
        expected_outcome="The record editor opens.",
        reason="The record must be edited.",
    )
    failed_entry = PlanningHistoryEntry(
        proposal=failed_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-failed",
            authorization_id="authorization-failed",
            receipt_id="receipt-failed",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=GroundedTargetPoint(
            point=PixelPoint(x=546, y=1240),
            screenshot_width_px=1080,
            screenshot_height_px=2400,
        ),
    )
    model = StubPlannerModel(
        _proposal_output(
            grounding_correction_for_proposal_id="proposal-failed",
            target_expression=(
                "the Edit Record row directly below Records near the top of "
                "the open navigation drawer, not the blank lower drawer area"
            ),
            reason=(
                "The previous point at (546,1240) was below the visible rows; refine "
                "the target toward the upper drawer row."
            ),
        )
    )

    proposal = _planner(model).plan(_planning_request(history=(failed_entry,)))

    assert proposal.grounding_correction_for_proposal_id == "proposal-failed"
    assert failed_entry.grounded_target_point is not None
    assert failed_entry.grounded_target_point.correction().minimum_distance_px == 22
    history = _history_contexts(model.requests[0])
    assert history[0]["grounded_target_point"] == {
        "point": {"x": 546, "y": 1240},
        "screenshot_height_px": 2400,
        "screenshot_width_px": 1080,
    }


def test_planner_rejects_a_correction_of_a_correction() -> None:
    corrected_proposal = ActionProposal(
        proposal_id="proposal-correction",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        grounding_correction_for_proposal_id="proposal-original",
        target_expression="the upper Edit Record row",
        expected_outcome="The record editor opens.",
        reason="The prior point was below the visible row.",
    )
    corrected_entry = PlanningHistoryEntry(
        proposal=corrected_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-correction",
            authorization_id="authorization-correction",
            receipt_id="receipt-correction",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=GroundedTargetPoint(
            point=PixelPoint(x=150, y=675),
            screenshot_width_px=1080,
            screenshot_height_px=2400,
        ),
    )
    model = StubPlannerModel(
        _proposal_output(
            grounding_correction_for_proposal_id="proposal-correction",
            target_expression="the Edit Record text row at the top of the drawer",
        )
    )

    with pytest.raises(TaskPlanningError, match="cannot correct another grounding correction"):
        _planner(model).plan(_planning_request(history=(corrected_entry,)))




def _entry(
    proposal_id: str,
    *,
    grounded: bool,
    corrects: str | None = None,
    target: str = "the Save button",
) -> PlanningHistoryEntry:
    proposal = ActionProposal(
        proposal_id=proposal_id,
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression=target,
        expected_outcome="The form saves.",
        reason="The visible Save button commits the form.",
        grounding_correction_for_proposal_id=corrects,
    )
    return PlanningHistoryEntry(
        proposal=proposal,
        outcome=ActionOutcome(
            outcome_id=f"outcome-{proposal_id}",
            authorization_id=f"authorization-{proposal_id}",
            receipt_id=f"receipt-{proposal_id}",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=(
            GroundedTargetPoint(
                point=PixelPoint(x=546, y=1240),
                screenshot_width_px=1080,
                screenshot_height_px=2400,
            )
            if grounded
            else None
        ),
    )


def _correction_payload(failed_proposal_id: str) -> dict[str, object]:
    return _proposal_output(
        grounding_correction_for_proposal_id=failed_proposal_id,
        target_expression="the Save button in the lower toolbar, not the top app bar",
        reason="The previous point at (546,1240) missed the control; refine the target.",
    )


def _durable_correction_payload(failed_proposal_id: str) -> dict[str, object]:
    return {
        "proposal_id": "proposal-correction",
        "source_observation_id": "obs_1",
        **_correction_payload(failed_proposal_id),
    }


def test_correcting_a_successful_target_names_the_field_to_use_instead() -> None:
    succeeded = _entry("proposal-worked", grounded=False)

    with pytest.raises(ValueError) as rejection:
        ActionProposal.model_validate(
            _durable_correction_payload("proposal-worked"),
            context={
                "correctable_grounding_proposals": correctable_grounding_proposals((succeeded,)),
            },
        )

    # The run that found this set the field when a tap grounded correctly but had the
    # wrong effect, so the message has to name the field that expresses that.
    assert "interaction_refinement_for_proposal_id" in str(rejection.value)


def test_a_valid_correction_passes_the_same_validator() -> None:
    failed = _entry("proposal-failed", grounded=True)

    proposal = ActionProposal.model_validate(
        _durable_correction_payload("proposal-failed"),
        context={"correctable_grounding_proposals": correctable_grounding_proposals((failed,))},
    )

    assert proposal.grounding_correction_for_proposal_id == "proposal-failed"


def test_the_planner_hands_the_correctable_map_to_the_retrying_validator() -> None:
    failed = _entry("proposal-failed", grounded=True)
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(_planning_request(history=(failed,)))

    context = model.validation_contexts[0]
    assert context is not None
    assert context["correctable_grounding_proposals"] == {
        "proposal-failed": ("tap", "the Save button")
    }


@pytest.mark.parametrize(
    ("label", "history_builder"),
    (
        ("correctable", lambda: (_entry("subject", grounded=True),)),
        ("target_succeeded", lambda: (_entry("subject", grounded=False),)),
        (
            "already_corrected",
            lambda: (
                _entry("subject", grounded=True),
                _entry("later", grounded=True, corrects="subject", target="a refined target"),
            ),
        ),
        (
            "is_itself_a_correction",
            lambda: (
                _entry("earlier", grounded=True),
                _entry("subject", grounded=True, corrects="earlier", target="a refined target"),
            ),
        ),
    ),
)
def test_the_retried_rule_and_the_backstop_agree_on_eligibility(
    label: str,
    history_builder: Any,
) -> None:
    history = history_builder()
    correction = ActionProposal.model_validate(_durable_correction_payload("subject"))

    eligible = "subject" in correctable_grounding_proposals(history)
    backstop_accepts = _grounding_correction_error(correction, history) is None

    # One owner of eligibility. If these drift, the backstop would end a run for a
    # correction the retried validator already allowed.
    assert eligible is backstop_accepts, label
