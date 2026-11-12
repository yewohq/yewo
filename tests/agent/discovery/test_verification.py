"""Independent discovery goal-verification tests."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic import BaseModel

from yewo.agent.discovery import DiscoveryObjective
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
)
from yewo.agent.planning.goal_verifier import (
    GoalEvidenceConflict,
    GoalVerificationAssessment,
    GoalVerificationActionEvidence,
    GoalVerificationMilestone,
    ModelGoalVerifier,
    format_goal_evidence_conflicts,
)
from yewo.agent.planning.goal_evidence import AdaptiveGoalEvidencePacker
from yewo.artifacts import ArtifactRetention
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelImageConstraints,
    TextModelContent,
)
from yewo.observations import ScreenshotStore

from observations.factories import build_observation
from observations.factories import build_screenshot


class StubAuditModel:
    def __init__(self, output: dict[str, object] | None) -> None:
        self.output = output
        self.request: ModelRequest | None = None

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        del response_model, validation_context
        self.request = request
        disposition = (
            ModelCallDisposition.COMPLETED
            if self.output is not None
            else ModelCallDisposition.PROVIDER_ERROR
        )
        return ModelResponse(
            request_id=request.request_id,
            disposition=disposition,
            text=json.dumps(self.output) if self.output is not None else None,
            structured_output=self.output,
            reason=None if self.output is not None else "The audit provider failed.",
            calls=(
                ModelCallRecord(
                    request_id=request.request_id,
                    profile_id=request.profile_id,
                    role=request.role,
                    requested_provider="offline",
                    requested_model="deterministic-auditor",
                    attempt_number=1,
                    started_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
                    disposition=disposition,
                    usage=ModelCallUsage(),
                ),
            ),
            retention=request.retention,
        )


def _objective() -> DiscoveryObjective:
    return DiscoveryObjective(
        objective_id="record_form",
        statement="The record form is reachable.",
        completion_conditions=("The record form heading is visible.",),
    )


def test_model_goal_verifier_uses_a_separate_auditor_call() -> None:
    observation = build_observation(observation_id="observation_final")
    output = GoalVerificationAssessment(
        satisfied=True,
        reason="The record form heading is visible.",
    ).model_dump(mode="json")
    model = StubAuditModel(output)
    identifiers = iter(("verification_1", "request_1"))

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=lambda: next(identifiers),
    ).verify(
        _objective().objective_id,
        _objective().statement,
        _objective().completion_conditions,
        observation,
    )

    assert verification.satisfied
    assert verification.verification_id == "verification_1"
    assert verification.objective_id == "record_form"
    assert verification.final_observation_id == observation.observation_id
    assert verification.model_request_id == "request_1"
    assert verification.model_call_count == 1
    assert model.request is not None
    assert model.request.role is ModelRole.GOAL_VERIFIER


def test_model_goal_verifier_preserves_objective_only_discovery() -> None:
    observation = build_observation(observation_id="observation_final")
    model = StubAuditModel(
        GoalVerificationAssessment(
            satisfied=True,
            reason="The visible final state supports the objective.",
        ).model_dump(mode="json")
    )

    ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=iter(("verification_1", "request_1")).__next__,
    ).verify("objective_1", "Create a record.", (), observation)

    assert model.request is not None
    content = model.request.messages[1].content[0]
    assert isinstance(content, TextModelContent)
    context = json.loads(content.text.split("\n", 1)[1])
    assert context["completion_conditions"] == []


def test_objective_only_discovery_accepts_transition_evidence() -> None:
    source = build_observation(observation_id="observation_source")
    final = build_observation(observation_id="observation_final")
    model = StubAuditModel(
        GoalVerificationAssessment(
            satisfied=True,
            reason="The transition and final screen support the objective.",
        ).model_dump(mode="json")
    )

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=iter(("verification_1", "request_1")).__next__,
    ).verify(
        "objective_1",
        "Open the saved record.",
        (),
        final,
        milestone_evidence=(
            GoalVerificationMilestone(
                source_observation=source,
                observation=final,
                transition_evidence="The recorded tap opened the saved record.",
            ),
        ),
    )

    assert verification.satisfied
    assert model.request is not None
    content = model.request.messages[1].content[0]
    assert isinstance(content, TextModelContent)
    context = json.loads(content.text.split("\n", 1)[1])
    assert context["milestones"][0]["completion_condition_index"] is None
    assert context["milestones"][0]["completion_condition"] is None


def test_auditor_response_contract_excludes_runtime_owned_fields() -> None:
    properties = GoalVerificationAssessment.model_json_schema()["properties"]

    assert set(properties) == {
        "satisfied",
        "reason",
        "observed_app_state",
        "evidence_conflicts",
    }


def test_goal_verification_rejects_success_with_conflicting_evidence() -> None:
    with pytest.raises(ValueError, match="Contradictory goal evidence"):
        GoalVerificationAssessment(
            satisfied=True,
            reason="The final screen shows a total.",
            evidence_conflicts=(
                GoalEvidenceConflict(
                    subject="amount charged",
                    earlier_observation="Payment submission showed 26.00.",
                    later_observation="Order confirmation showed 31.00.",
                    reason="The same transaction reports two charged amounts.",
                ),
            ),
        )


def test_model_goal_verifier_preserves_evidence_conflicts() -> None:
    observation = build_observation(observation_id="observation_final")
    model = StubAuditModel(
        GoalVerificationAssessment(
            satisfied=False,
            reason="The charged amount is contradictory.",
            evidence_conflicts=(
                GoalEvidenceConflict(
                    subject="amount charged",
                    earlier_observation="Payment submission showed 26.00.",
                    later_observation="Order confirmation showed 31.00.",
                    reason="The same transaction reports two charged amounts.",
                ),
            ),
        ).model_dump(mode="json")
    )

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=iter(("verification_1", "request_1")).__next__,
    ).verify(
        _objective().objective_id,
        _objective().statement,
        _objective().completion_conditions,
        observation,
    )

    assert not verification.satisfied
    assert verification.evidence_conflicts[0].subject == "amount charged"
    assert format_goal_evidence_conflicts(verification.evidence_conflicts) == (
        "Discovery found 1 material evidence conflict(s). amount charged: "
        "Payment submission showed 26.00. Conflicting evidence: "
        "Order confirmation showed 31.00. The same transaction reports two charged amounts."
    )


def test_model_goal_verifier_fails_closed_when_the_call_fails() -> None:
    observation = build_observation(observation_id="observation_final")
    model = StubAuditModel(None)
    identifiers = iter(("verification_1", "request_1"))

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=lambda: next(identifiers),
    ).verify(
        _objective().objective_id,
        _objective().statement,
        _objective().completion_conditions,
        observation,
    )

    assert not verification.satisfied
    assert verification.reason == "The audit provider failed."


def test_model_goal_verifier_audits_bounded_milestones_and_final_image() -> None:
    final_observation = build_observation(
        observation_id="observation_final",
        screenshot=build_screenshot(artifact_id="shot_final"),
    )
    milestone_observation = build_observation(
        observation_id="observation_milestone",
        screenshot=build_screenshot(artifact_id="shot_milestone"),
    )
    source_observation = build_observation(
        observation_id="observation_source",
        screenshot=build_screenshot(artifact_id="shot_source"),
    )
    output = GoalVerificationAssessment(
        satisfied=True,
        reason="Both mapped conditions are visible in the milestone.",
    ).model_dump(mode="json")
    model = StubAuditModel(output)
    identifiers = iter(("verification_1", "request_1"))

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=lambda: next(identifiers),
    ).verify(
        "record_form",
        "Complete the record form.",
        ("The heading is visible.", "The saved value is visible."),
        final_observation,
        milestone_evidence=(
            GoalVerificationMilestone(
                completion_condition_index=0,
                source_observation=source_observation,
                observation=milestone_observation,
                transition_evidence=(
                    "This observation followed an observed Login tap and screen change."
                ),
                supporting_actions=(
                    GoalVerificationActionEvidence(
                        step_index=1,
                        procedure_index=0,
                        action=ActionKind.TAP,
                        target_expression="the record form button",
                        outcome_disposition=ActionOutcomeDisposition.OBSERVED,
                        outcome_signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
                    ),
                ),
            ),
            GoalVerificationMilestone(
                completion_condition_index=1,
                observation=milestone_observation,
            ),
        ),
    )

    assert verification.satisfied
    assert model.request is not None
    image_parts = tuple(
        part for part in model.request.messages[1].content if isinstance(part, ImageModelContent)
    )
    assert tuple(part.screenshot.artifact_id for part in image_parts) == (
        "shot_source",
        "shot_milestone",
        "shot_final",
    )
    text_part = model.request.messages[1].content[0]
    assert isinstance(text_part, TextModelContent)
    assert "observed Login tap and screen change" in text_part.text
    assert '"supporting_actions"' in text_part.text
    assert '"source_observation"' in text_part.text
    assert "the record form button" in text_part.text


def test_four_milestones_are_packed_into_one_auditor_image(tmp_path: Path) -> None:
    store = ScreenshotStore(
        tmp_path,
        identifier_factory=iter(f"shot_{index}" for index in range(9)).__next__,
    )

    def observation(index: int):
        output = BytesIO()
        Image.new("RGB", (200, 300), color=(index * 20, 40, 80)).save(
            output,
            format="PNG",
        )
        screenshot = store.persist(
            output.getvalue(),
            retention=ArtifactRetention.RUN,
        )
        return build_observation(
            observation_id=f"observation_{index}",
            screenshot=screenshot,
        )

    observations = tuple(observation(index) for index in range(9))
    final_observation = observations[-1]
    conditions = tuple(f"Condition {index} is visibly satisfied." for index in range(4))
    output = GoalVerificationAssessment(
        satisfied=True,
        reason="The packed milestone chain supports all four conditions.",
    ).model_dump(mode="json")
    model = StubAuditModel(output)
    identifiers = iter(("verification_1", "request_1"))
    milestones = tuple(
        GoalVerificationMilestone(
            completion_condition_index=index,
            source_observation=observations[index * 2],
            observation=observations[index * 2 + 1],
            transition_evidence=f"Observed transition {index}.",
            supporting_actions=(
                GoalVerificationActionEvidence(
                    step_index=index + 1,
                    procedure_index=index,
                    action=ActionKind.TAP,
                    target_expression=f"control {index}",
                    outcome_disposition=ActionOutcomeDisposition.OBSERVED,
                    outcome_signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
                ),
            ),
        )
        for index in range(4)
    )

    verification = ModelGoalVerifier(
        model,
        "auditor-primary",
        evidence_packer=AdaptiveGoalEvidencePacker(
            tmp_path,
            ModelImageConstraints(
                max_images=1,
                max_image_bytes=4_000_000,
                max_width_px=1_024,
                max_height_px=2_048,
            ),
        ),
        identifier_factory=lambda: next(identifiers),
    ).verify(
        "sample_workflow",
        "Create, save, and reopen a record.",
        conditions,
        final_observation,
        milestone_evidence=milestones,
    )

    assert verification.satisfied
    assert model.request is not None
    image_parts = tuple(
        part for part in model.request.messages[1].content if isinstance(part, ImageModelContent)
    )
    assert len(image_parts) == 1
    text_part = model.request.messages[1].content[0]
    assert isinstance(text_part, TextModelContent)
    context = json.loads(text_part.text.split("\n", 1)[1])
    assert tuple(item["completion_condition_index"] for item in context["milestones"]) == (
        0,
        1,
        2,
        3,
    )
    assert all(item["supporting_actions"] for item in context["milestones"])
    assert len(context["evidence_images"]) == 1
    assert len(context["evidence_images"][0]["observation_ids"]) == 9


def _verifier(model: StubAuditModel) -> ModelGoalVerifier:
    return ModelGoalVerifier(
        model,
        "auditor-primary",
        identifier_factory=iter(("verification_1", "request_1")).__next__,
    )


def test_several_milestones_can_evidence_one_completion_condition() -> None:
    source = build_observation(observation_id="observation_source")
    middle = build_observation(observation_id="observation_middle")
    final = build_observation(observation_id="observation_final")
    model = StubAuditModel(
        GoalVerificationAssessment(
            satisfied=True,
            reason="Both captured transitions establish the condition.",
        ).model_dump(mode="json")
    )

    verification = _verifier(model).verify(
        "record_form",
        "Create the record.",
        ("The saved record is visible.",),
        final,
        milestone_evidence=(
            GoalVerificationMilestone(
                completion_condition_index=0,
                source_observation=source,
                observation=middle,
                transition_evidence="The recorded tap opened the record.",
            ),
            GoalVerificationMilestone(
                completion_condition_index=0,
                source_observation=middle,
                observation=final,
                transition_evidence="The recorded save returned to the collection.",
            ),
        ),
    )

    assert verification.satisfied
    assert model.request is not None
    content = model.request.messages[1].content[0]
    assert isinstance(content, TextModelContent)
    milestones = json.loads(content.text.split("\n", 1)[1])["milestones"]
    assert [item["completion_condition_index"] for item in milestones] == [0, 0]
    assert {item["completion_condition"] for item in milestones} == {"The saved record is visible."}


def test_milestone_index_must_name_a_supplied_condition() -> None:
    final = build_observation(observation_id="observation_final")

    with pytest.raises(ValueError, match="must name a condition"):
        _verifier(StubAuditModel(None)).verify(
            "record_form",
            "Create the record.",
            ("The saved record is visible.",),
            final,
            milestone_evidence=(
                GoalVerificationMilestone(completion_condition_index=1, observation=final),
            ),
        )


def test_objective_milestone_requires_an_objective_without_conditions() -> None:
    final = build_observation(observation_id="observation_final")

    with pytest.raises(ValueError, match="objective without conditions"):
        _verifier(StubAuditModel(None)).verify(
            "record_form",
            "Create the record.",
            ("The saved record is visible.",),
            final,
            milestone_evidence=(GoalVerificationMilestone(observation=final),),
        )
