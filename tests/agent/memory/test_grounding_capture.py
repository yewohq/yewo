"""Procedure grounding capture tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.memory.grounding_capture import build_grounding_assessments
from yewo.agent.replay.procedure.records import (
    ProcedureGroundingDecision,
    ProcedureGroundingDecisionCode,
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.reconciliation.records import (
    ProcedureEffect,
    ProcedureExpectation,
    ReplayProcedure,
    ReplayProcedureStep,
)
from yewo.grounding import GroundedPoint, GroundingResolutionSource, PixelPoint

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


def test_completed_procedure_grounding_reinforces_exact_memory_location(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    evidence = agent_result(before, after)
    source_step = evidence.steps[0]
    transition = memory.transitions[0]
    assert source_step.target_grounding is not None
    assert transition.target_grounding is not None
    procedure = ReplayProcedure(
        procedure_id="procedure_1",
        request_id="request_1",
        evidence_id="evidence_1",
        source_run_id="run_1",
        application=memory.compatibility.application,
        source_step_ids=(source_step.step_id,),
        steps=(
            ReplayProcedureStep(
                procedure_index=0,
                source_step=1,
                source_step_id=source_step.step_id,
                source_state_id="state_1",
                proposal=source_step.proposal,
                target_grounding=source_step.target_grounding,
                procedure_effect=ProcedureEffect.NAVIGATION,
                expects=ProcedureExpectation(),
            ),
        ),
    )
    decision = ProcedureGroundingDecision(
        grounding_id="live_grounding_1",
        source_step_id=source_step.step_id,
        procedure_index=0,
        expected_state_id="state_1",
        observation_id="live_obs_1",
        screenshot_sha256="0" * 64,
        decision=ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_REUSED,
        stored_location=source_step.target_grounding.result.location,
        resolved_location=source_step.target_grounding.result.location,
        resolution_source=GroundingResolutionSource.PROCEDURE,
    )
    replay = ProcedureReplayResult(
        procedure_id=procedure.procedure_id,
        evidence_id=procedure.evidence_id,
        source_run_id=procedure.source_run_id,
        setup_id="attempt_1",
        application=procedure.application,
        terminal=ProcedureReplayTerminal.BUDGET_EXHAUSTED,
        procedure_step_count=1,
        completed_step_count=1,
        grounding_decisions=(decision,),
        reason="The test stops after the completed step.",
        agent_result=evidence,
    )

    assessments = build_grounding_assessments(
        memory, "run_2", replay, procedure, datetime(2026, 11, 3, tzinfo=timezone.utc)
    )

    assert len(assessments) == 1
    assert assessments[0].status.value == "active"
    assert assessments[0].successful_use_count == 1
    assert assessments[0].transition_id == transition.transition_id

    stored_location = decision.stored_location
    assert isinstance(stored_location, GroundedPoint)
    corrected_location = stored_location.model_copy(
        update={
            "pixel": PixelPoint(
                x=stored_location.pixel.x + 1,
                y=stored_location.pixel.y,
            )
        }
    )
    corrected = decision.model_copy(
        update={
            "decision": ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_RELOCATED,
            "resolved_location": corrected_location,
            "resolution_source": GroundingResolutionSource.PROCEDURE,
        }
    )
    invalidated = build_grounding_assessments(
        memory,
        "run_3",
        replay.model_copy(update={"grounding_decisions": (corrected,)}),
        procedure,
        datetime(2026, 11, 3, tzinfo=timezone.utc),
    )

    assert invalidated[0].status.value == "invalidated"
    assert invalidated[0].successful_use_count == 0
    assert invalidated[0].invalidation_reason is not None
