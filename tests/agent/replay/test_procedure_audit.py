"""Final audit evidence for compiled-procedure replay."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import AgentModePolicy
from yewo.agent.actuation import ActionKind
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay.procedure.audit import audit_completed_replay
from yewo.agent.replay.procedure.records import MilestoneEvidence
from yewo.models import ModelClient
from yewo.runs import RunExecutionContext

from agent.actuation.support import limits
from agent.discovery.support import agent_result
from agent.replay.support import evidence as _evidence, observations as _observations


def test_completed_procedure_auditor_receives_transition_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    source_observation = _observations()[2]
    live_observation = _observations()[-1]
    milestone = MilestoneEvidence(
        completion_condition_index=0,
        procedure_index=2,
        source_step=3,
        observation_id=live_observation.observation_id,
        screenshot_artifact_id=live_observation.redacted_screenshot.artifact_id,
        screen_fingerprint=live_observation.screen_fingerprint,
    )
    received_milestones = ()

    class StubVerifier:
        def verify(self, objective_id, objective, conditions, final, *, milestone_evidence):
            nonlocal received_milestones
            del objective, conditions
            received_milestones = milestone_evidence
            return GoalVerification(
                verification_id="verification_1",
                objective_id=objective_id,
                final_observation_id=final.observation_id,
                satisfied=True,
                reason="The live milestone proves the completion condition.",
            )

    monkeypatch.setattr(
        "yewo.agent.replay.procedure.audit.ObservationStore.load_selected",
        lambda unused, observation_ids: (source_observation, live_observation),
    )
    monkeypatch.setattr(
        "yewo.agent.replay.procedure.audit.ModelGoalVerifier",
        lambda *args, **kwargs: StubVerifier(),
    )
    context = cast(
        RunExecutionContext,
        SimpleNamespace(workspace=SimpleNamespace(root=tmp_path)),
    )
    transition_step = (
        agent_result(source_observation, live_observation, action=ActionKind.TAP)
        .steps[0]
        .model_copy(
            update={
                "decision": PlannerDecisionMetadata(
                    source=PlannerDecisionSource.COMPILED_PROCEDURE,
                    reused_flow_id="procedure_1",
                    reused_procedure_step_ids=("source_step_3",),
                    reused_segment_index=2,
                )
            }
        )
    )

    verification = audit_completed_replay(
        context,
        evidence.task,
        (milestone,),
        live_observation.observation_id,
        0,
        0,
        AgentModePolicy(
            action_limits=limits(),
            maximum_steps=5,
            maximum_grounding_requests=5,
            maximum_model_requests=10,
        ),
        cast(ModelClient, object()),
        "auditor-primary",
        (transition_step,),
    )

    assert verification.satisfied
    assert len(received_milestones) == 1
    assert received_milestones[0].source_observation == source_observation
    assert received_milestones[0].observation == live_observation
    assert len(received_milestones[0].supporting_actions) == 1
    assert received_milestones[0].supporting_actions[0].action is ActionKind.TAP
    assert received_milestones[0].supporting_actions[0].procedure_index == 2
    assert "recorded tap action" in received_milestones[0].transition_evidence
    assert "pixels_changed" in received_milestones[0].transition_evidence
