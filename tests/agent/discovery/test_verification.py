"""Independent discovery goal-verification tests."""

from datetime import datetime, timezone
import json

from pydantic import BaseModel

from yewo.agent.discovery import DiscoveryObjective
from yewo.agent.goal_verifier import GoalVerification, ModelGoalVerifier
from yewo.models import (
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
    ModelRole,
)

from observations.factories import build_observation


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
                    role=ModelRole.AUDITOR,
                    requested_provider="offline",
                    requested_model="deterministic-auditor",
                    attempt_number=1,
                    started_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
                    disposition=disposition,
                    usage=ModelCallUsage(),
                ),
            ),
            sensitivity=request.sensitivity,
            retention=request.retention,
        )


def _objective() -> DiscoveryObjective:
    return DiscoveryObjective(
        objective_id="account_form",
        statement="The account form is reachable.",
        completion_conditions=("The account form heading is visible.",),
    )


def test_model_goal_verifier_uses_a_separate_auditor_call() -> None:
    observation = build_observation(observation_id="observation_final")
    output = GoalVerification(
        verification_id="verification_1",
        objective_id="account_form",
        final_observation_id=observation.observation_id,
        satisfied=True,
        reason="The account form heading is visible.",
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
    assert verification.model_request_id == "request_1"
    assert verification.model_call_count == 1
    assert model.request is not None
    assert model.request.role is ModelRole.AUDITOR


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
