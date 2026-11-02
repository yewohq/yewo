"""Bounded model recovery decisions after established application failures."""

import json
from pathlib import Path
from typing import Any

from yewo.agent.action_diagnostics import ActionApplicationFailure
from yewo.agent.actuation import (
    ActionKind,
    ActionRefusal,
    ActionRefusalCode,
    LiteralText,
)
from yewo.agent.planner import PlanningHistoryEntry, PlanningRequest, TaskProgressContext
from yewo.agent.app_memory import ScreenComparisonPolicy
from yewo.agent.recovery_advisor import (
    ModelRecoveryAdvisor,
    RecoveryAssessment,
    RecoveryDisposition,
)
from yewo.agent.recovery_runtime import RecoveryRunState
from yewo.agent.runtime_records import AgentStep, AgentTerminalReason
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelInputError,
    ModelRole,
    TextModelContent,
)
from yewo.observations import Observation

from agent.actuation.support import proposal
from agent.discovery.support import agent_result, observation
from agent.planner_support import StubPlannerModel


def _planning_request(before: Observation) -> PlanningRequest:
    previous = proposal(
        ActionKind.TYPE,
        proposal_id="proposal_previous",
        source_observation_id=before.observation_id,
        text=LiteralText(value="do-not-send"),
    )
    return PlanningRequest(
        observation=before,
        objective="Save a training schedule.",
        known_facts=("The editor was reached in this run.",),
        open_work=("Complete the schedule form.", "Save the schedule."),
        completion_conditions=("The saved schedule is visible in the schedule list.",),
        available_actions=frozenset({ActionKind.TAP}),
        history=(
            PlanningHistoryEntry(
                proposal=previous,
                refusal=ActionRefusal(
                    proposal_id=previous.proposal_id,
                    code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                    reason="The transition was exhausted.",
                ),
            ),
        ),
        progress_context=TaskProgressContext(
            current_progress=("The schedule editor was opened.",),
            known_procedures=("The Add control opens the editor.",),
            remaining_work=("Complete the schedule form.", "Save the schedule."),
            exhausted_branches=("The overflow menu was irrelevant.",),
            replay_target_transition_id="transition_1",
        ),
        discovery_mode=True,
    )


def _failure() -> ActionApplicationFailure:
    return ActionApplicationFailure(
        classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
        failure_kinds=(ApplicationFailureKind.CRASH,),
        summary="The application crashed after submission.",
        recovery_succeeded=True,
    )


def _failed_step(
    before: Observation,
    recovered: Observation,
    *,
    literal: str = "failed-secret",
) -> AgentStep:
    settled = observation("obs_settled", "2222222222222222", offset_seconds=1)
    result = agent_result(
        before,
        settled,
        terminal=AgentTerminalReason.APPLICATION_EXITED,
        action=ActionKind.TYPE,
    )
    values: dict[str, Any] = result.steps[0].model_dump(mode="python")
    values["proposal"]["text"]["value"] = literal
    values["proposal"]["reason"] = f"Type {literal} into the field."
    values["authorization"]["text"]["value"] = literal
    values["authorization"]["expected_outcome"] = f"The field contains {literal}."
    values["outcome"]["signals"] = ["app_exited"]
    values["application_failure"] = _failure().model_dump(mode="python")
    values["recovery_observation_id"] = recovered.observation_id
    return AgentStep.model_validate(values)


def test_advisor_persists_validated_continue_decision(tmp_path: Path) -> None:
    before = observation("obs_before", "1111111111111111")
    recovered = observation("obs_recovered", "0000000000000000", offset_seconds=1)
    failed_step = _failed_step(before, recovered)
    model = StubPlannerModel(
        {
            "assessment_id": "assessment_1",
            "failed_proposal_id": failed_step.proposal.proposal_id,
            "pre_failure_observation_id": before.observation_id,
            "recovery_observation_id": recovered.observation_id,
            "disposition": "continue",
            "reason": "The fresh screen allows one bounded reconstruction.",
            "remaining_work": ["Complete every visible required field before submission."],
            "planner_guidance": ["The previous submission did not prove persistence."],
        }
    )
    identifiers = iter(("assessment_1", "request_1"))
    advisor = ModelRecoveryAdvisor(
        model,
        "recovery-primary",
        tmp_path,
        identifier_factory=lambda: next(identifiers),
    )
    planning_request = _planning_request(before)
    planning_request = planning_request.model_copy(
        update={"history": planning_request.history * 16}
    )

    assessment = advisor.assess(planning_request, failed_step, recovered)

    assert assessment.disposition is RecoveryDisposition.CONTINUE
    assert assessment.model_request_id == "request_1"
    assert assessment.model_call_count == 1
    request = model.requests[0]
    assert request.response_contract is not None
    assert request.role is ModelRole.RECOVERY_ADVISOR
    assert request.response_contract.name == "recovery-assessment-v1"
    assert request.max_completion_tokens == 8_192
    system = request.messages[0].content[0]
    assert isinstance(system, TextModelContent)
    text, before_image, label, recovered_image = request.messages[1].content
    assert isinstance(text, TextModelContent)
    assert isinstance(before_image, ImageModelContent)
    assert isinstance(label, TextModelContent)
    assert isinstance(recovered_image, ImageModelContent)
    context_json = text.text.split("Context JSON follows:\n", 1)[1].split(
        "\nPre-failure screenshot:", 1
    )[0]
    context = json.loads(context_json)
    assert context["discovery_mode"] is True
    assert context["progress_context"]["current_progress"] == [
        "The schedule editor was opened."
    ]
    assert context["progress_context"]["replay_target_transition_id"] == "transition_1"
    assert context["recent_history"][0]["action"] == "type"
    assert len(context["recent_history"]) == 14
    assert context["earlier_history_summary"]["entries"][0]["count"] == 2
    assert "text" not in context["recent_history"][0]
    assert context["failed_step"]["application_failure"]["recovery_succeeded"]
    assert "do-not-send" not in text.text
    assert "failed-secret" not in text.text
    assert context["failed_step"]["proposal"]["reason"] == (
        "Type [typed value] into the field."
    )
    stored = tmp_path / "assets/agent/recovery-assessments.jsonl"
    assert json.loads(stored.read_text().strip())["assessment_id"] == "assessment_1"


def test_short_literal_redaction_does_not_corrupt_structured_evidence(tmp_path: Path) -> None:
    before = observation("obs_before", "1111111111111111")
    recovered = observation("obs_recovered", "0000000000000000", offset_seconds=1)
    failed_step = _failed_step(before, recovered, literal="3")
    model = StubPlannerModel(None, disposition=ModelCallDisposition.PROVIDER_ERROR)
    identifiers = iter(("assessment_1", "request_1"))
    advisor = ModelRecoveryAdvisor(
        model,
        "recovery-primary",
        tmp_path,
        identifier_factory=lambda: next(identifiers),
    )

    advisor.assess(_planning_request(before), failed_step, recovered)

    content = model.requests[0].messages[1].content[0]
    assert isinstance(content, TextModelContent)
    context_json = content.text.split("Context JSON follows:\n", 1)[1].split(
        "\nPre-failure screenshot:", 1
    )[0]
    context = json.loads(context_json)
    failed_context = context["failed_step"]
    assert failed_step.receipt is not None
    assert failed_context["receipt"]["receipt_id"] == failed_step.receipt.receipt_id
    assert failed_context["step_index"] == failed_step.step_index
    assert failed_context["proposal"]["reason"] == "Type [typed value] into the field."


def test_advisor_fails_closed_on_unusable_response(tmp_path: Path) -> None:
    before = observation("obs_before", "1111111111111111")
    recovered = observation("obs_recovered", "0000000000000000", offset_seconds=1)
    failed_step = _failed_step(before, recovered)
    model = StubPlannerModel(None, disposition=ModelCallDisposition.PROVIDER_ERROR)
    identifiers = iter(("assessment_1", "request_1"))
    advisor = ModelRecoveryAdvisor(
        model,
        "recovery-primary",
        tmp_path,
        identifier_factory=lambda: next(identifiers),
    )

    assessment = advisor.assess(_planning_request(before), failed_step, recovered)

    assert assessment.disposition is RecoveryDisposition.STOP
    assert "configured planner call failed" in assessment.reason


def test_advisor_transport_failure_does_not_end_the_run(tmp_path: Path) -> None:
    before = observation("obs_before", "1111111111111111")
    recovered = observation("obs_recovered", "0000000000000000", offset_seconds=1)
    failed_step = _failed_step(before, recovered)

    class UnavailableAdvisor:
        def assess(
            self,
            planning_request: PlanningRequest,
            failed_step: AgentStep,
            recovery_observation: Observation,
        ) -> RecoveryAssessment:
            raise ModelInputError("The model request exceeds the profile image-count limit.")

    state = RecoveryRunState(UnavailableAdvisor(), ScreenComparisonPolicy())

    assessment = state.assess(_planning_request(before), failed_step, recovered, 4, 10.0)

    assert assessment is None
    assert state.assessment is None
    assert state.is_active is False
    assert state.credit(4, 11.0).steps == 0
