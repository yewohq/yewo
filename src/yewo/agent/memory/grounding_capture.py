"""Capture grounding reinforcement and invalidation from procedure replay."""

from datetime import datetime

from yewo.agent.memory.knowledge import (
    GroundingAssessment,
    GroundingAssessmentStatus,
    GroundingRole,
    KnowledgeEvidenceReference,
)
from yewo.agent.memory.records import AcceptedTransition, AppMemory, derived_memory_identifier
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
        assessment = _assessment_for_decision(
            decision=decision,
            completed_step_count=replay.completed_step_count,
            procedure_steps_by_index=procedure_steps_by_index,
            transitions_by_source_step_id=transitions_by_source_step_id,
            replay_step_ids_by_procedure_index=replay_step_ids_by_procedure_index,
            current_run_id=current_run_id,
            procedure_attempt_id=replay.setup_id,
            assessed_at=assessed_at,
        )
        if assessment is not None:
            assessments.append(assessment)
    return tuple(assessments)


def _assessment_for_decision(
    *,
    decision: ProcedureGroundingDecision,
    completed_step_count: int,
    procedure_steps_by_index: dict[int, ReplayProcedureStep],
    transitions_by_source_step_id: dict[str, AcceptedTransition],
    replay_step_ids_by_procedure_index: dict[int, str],
    current_run_id: str,
    procedure_attempt_id: str,
    assessed_at: datetime,
) -> GroundingAssessment | None:
    procedure_index = decision.procedure_index
    if procedure_index is None or procedure_index >= completed_step_count:
        return None
    procedure_step = procedure_steps_by_index.get(procedure_index)
    if procedure_step is None or decision.source_step_id != procedure_step.source_step_id:
        return None
    transition = transitions_by_source_step_id.get(procedure_step.source_step_id)
    if transition is None:
        return None
    role = _grounding_role(procedure_step, decision)
    status = _grounding_status(decision)
    if role is None or status is None:
        return None
    grounding = (
        transition.target_grounding
        if role is GroundingRole.TARGET
        else transition.destination_grounding
    )
    if grounding is None:
        return None
    source_observation_id = grounding.source_observation_id or transition.source_observation_id
    is_active = status is GroundingAssessmentStatus.ACTIVE
    return GroundingAssessment(
        assessment_id=derived_memory_identifier(
            "grounding-assessment",
            transition.transition_id,
            role.value,
            source_observation_id,
        ),
        transition_id=transition.transition_id,
        role=role,
        source_observation_id=source_observation_id,
        status=status,
        successful_use_count=int(is_active),
        last_success_at=assessed_at if is_active else None,
        invalidation_reason=(
            None if is_active else "A completed replay used a corrected grounding location."
        ),
        sources=(
            KnowledgeEvidenceReference(
                run_id=current_run_id,
                step_id=replay_step_ids_by_procedure_index.get(procedure_index),
                observation_id=decision.observation_id,
                procedure_attempt_id=procedure_attempt_id,
            ),
        ),
    )


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
