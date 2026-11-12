"""Shared ownership rules for structured model outputs."""

import pytest
from pydantic import BaseModel

from yewo.agent.actuation import ActionProposalDecision
from yewo.agent.actuation.proposal_verifier import ActionReadinessDecision
from yewo.agent.planning.goal_verifier import GoalVerificationAssessment
from yewo.agent.recovery.advisor import RecoveryDecision
from yewo.agent.replay.procedure.objective_matching import ObjectiveMatchOutput
from yewo.agent.replay.reconciliation.records import ReconcilerSelection
from yewo.agent.screen_analysis.service import ScreenAnalysisHypothesis


RUNTIME_OWNED_OUTPUT_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "model_request_id",
        "model_call_count",
        "proposal_id",
        "source_observation_id",
        "verification_id",
        "assessment_id",
        "objective_id",
        "failed_proposal_id",
        "pre_failure_observation_id",
        "recovery_observation_id",
        "final_observation_id",
    }
)


@pytest.mark.parametrize(
    "response_model",
    (
        ActionProposalDecision,
        ActionReadinessDecision,
        GoalVerificationAssessment,
        RecoveryDecision,
        ObjectiveMatchOutput,
        ReconcilerSelection,
        ScreenAnalysisHypothesis,
    ),
)
def test_structured_model_outputs_exclude_runtime_identity_and_accounting(
    response_model: type[BaseModel],
) -> None:
    properties = set(response_model.model_json_schema().get("properties", {}))

    assert properties.isdisjoint(RUNTIME_OWNED_OUTPUT_FIELDS)
