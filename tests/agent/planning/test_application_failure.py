"""Planner context tests for action-scoped application failures."""

from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
)
from yewo.agent.planning.planner import PlanningHistoryEntry
from yewo.artifacts import ArtifactRetention
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)

from agent.planning.support import (
    StubPlannerModel,
    history_contexts,
    planner,
    planning_request,
    proposal_output,
)


def test_planner_receives_concise_recovered_application_failure() -> None:
    proposal = ActionProposal(
        proposal_id="proposal-crashed",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the Save button",
        expected_outcome="The form saves.",
        reason="The completed form must be saved.",
    )
    entry = PlanningHistoryEntry(
        proposal=proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-1",
            authorization_id="authorization-1",
            receipt_id="receipt-1",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.APP_EXITED}),
            reason="The application crashed after Save.",
            retention=ArtifactRetention.RUN,
        ),
        application_failure=ActionApplicationFailure(
            classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
            failure_kinds=(ApplicationFailureKind.CRASH,),
            summary="The application crashed after Save.",
            source_artifact_ids=("diagnostic_1",),
            recovery_succeeded=True,
        ),
    )
    model = StubPlannerModel(proposal_output())

    planner(model).plan(planning_request(history=(entry,)))

    history = history_contexts(model.requests[0])
    assert history[0]["application_failure"] == {
        "classification": "application_crash",
        "summary": "The application crashed after Save.",
        "recovery_succeeded": True,
    }
