"""Capture grounding reinforcement and invalidation from procedure replay."""

from datetime import datetime

from yewo.agent.memory.knowledge import (
    GroundingAssessment,
    GroundingAssessmentStatus,
    GroundingRole,
    KnowledgeEvidenceReference,
)
from yewo.agent.memory.records import AppMemory, derived_memory_identifier
from yewo.agent.replay.procedure.records import (
    ProcedureGroundingDecision,
    ProcedureGroundingDecisionCode,
    ProcedureReplayResult,
)
from yewo.agent.replay.reconciliation.records import ReplayProcedure, ReplayProcedureStep


def build_grounding_assessments(
    memory: AppMemory,
    current_run_id: str,
    replay: ProcedureReplayResult,
    procedure: ReplayProcedure,
    assessed_at: datetime,
) -> tuple[GroundingAssessment, ...]:
    """Translate completed procedure decisions into exact grounding evidence."""
    if replay.procedure_id != procedure.procedure_id:
        raise ValueError("The replay result does not match the retained procedure.")
    procedure_steps_by_index = {step.procedure_index: step for step in procedure.steps}
    transitions_by_source_step_id = {
        transition.source_step_id: transition for transition in memory.transitions
    }
    replay_step_ids_by_procedure_index = {
        step.decision.reused_segment_index: step.step_id
        for step in replay.agent_result.steps
        if step.decision.reused_segment_index is not None
    }
    assessments: list[GroundingAssessment] = []
    for decision in replay.grounding_decisions:
        if (
            decision.procedure_index is None
            or decision.procedure_index >= replay.completed_step_count
        ):
            continue
        procedure_step = procedure_steps_by_index.get(decision.procedure_index)
        if procedure_step is None or decision.source_step_id != procedure_step.source_step_id:
            continue
        transition = transitions_by_source_step_id.get(procedure_step.source_step_id)
        if transition is None:
            continue
        role = _grounding_role(procedure_step, decision)
        if role is None:
            continue
        grounding = (
            transition.target_grounding
            if role is GroundingRole.TARGET
            else transition.destination_grounding
        )
        if grounding is None:
            continue
        status = _grounding_status(decision)
        if status is None:
            continue
        source_observation_id = grounding.source_observation_id or transition.source_observation_id
        source = KnowledgeEvidenceReference(
            run_id=current_run_id,
            step_id=replay_step_ids_by_procedure_index.get(decision.procedure_index),
            observation_id=decision.observation_id,
            procedure_attempt_id=replay.setup_id,
        )
        assessment_id = derived_memory_identifier(
            "grounding-assessment",
            transition.transition_id,
            role.value,
            source_observation_id,
        )
        is_active = status is GroundingAssessmentStatus.ACTIVE
        assessments.append(
            GroundingAssessment(
                assessment_id=assessment_id,
                transition_id=transition.transition_id,
                role=role,
                source_observation_id=source_observation_id,
                status=status,
                successful_use_count=int(is_active),
                last_success_at=assessed_at if is_active else None,
                invalidation_reason=(
                    None if is_active else "A completed replay used a corrected grounding location."
                ),
                sources=(source,),
            )
        )
    return tuple(assessments)


def _grounding_role(
    procedure_step: ReplayProcedureStep,
    decision: ProcedureGroundingDecision,
) -> GroundingRole | None:
    candidates = tuple(
        role
        for role, evidence in (
            (GroundingRole.TARGET, procedure_step.target_grounding),
            (GroundingRole.DESTINATION, procedure_step.destination_grounding),
        )
        if evidence is not None and evidence.result.location == decision.stored_location
    )
    return candidates[0] if len(candidates) == 1 else None


def _grounding_status(
    decision: ProcedureGroundingDecision,
) -> GroundingAssessmentStatus | None:
    if decision.decision is ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_REUSED:
        return GroundingAssessmentStatus.ACTIVE
    if decision.decision is ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_RELOCATED:
        return (
            GroundingAssessmentStatus.ACTIVE
            if decision.resolved_location == decision.stored_location
            else GroundingAssessmentStatus.INVALIDATED
        )
    return None
