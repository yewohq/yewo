"""Audit completed procedure replay from current-run milestone evidence."""

from yewo.agent.planning.goal_evidence import AdaptiveGoalEvidencePacker
from yewo.agent.planning.goal_verifier import (
    GoalVerification,
    GoalVerificationActionEvidence,
    GoalVerificationMilestone,
    ModelGoalVerifier,
    build_goal_transition_evidence,
)
from yewo.agent.replay.procedure.records import MilestoneEvidence
from yewo.agent.replay.reconciliation.records import EvidenceTask
from yewo.agent.runtime.records import AgentModePolicy, AgentStep
from yewo.agent.runtime.verification import goal_verification_budget_reason
from yewo.artifacts import new_record_identifier
from yewo.models import ModelClient, ModelImageConstraints, ModelInputError
from yewo.observations import (
    Observation,
    ObservationCaptureError,
    ObservationStore,
)
from yewo.runs import RunExecutionContext


def audit_completed_replay(
    context: RunExecutionContext,
    audit_task: EvidenceTask,
    milestones: tuple[MilestoneEvidence, ...],
    final_observation_id: str,
    used_model_requests: int,
    elapsed_seconds: float,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    auditor_profile_id: str,
    agent_steps: tuple[AgentStep, ...] = (),
    auditor_image_constraints: ModelImageConstraints | None = None,
) -> GoalVerification:
    """Verify every completion condition from current-run replay evidence."""
    budget_reason = goal_verification_budget_reason(
        mode_policy,
        used_model_requests,
        elapsed_seconds,
    )
    if budget_reason is not None:
        return _failed_replay_verification(
            audit_task.objective_id,
            final_observation_id,
            budget_reason,
        )
    expected_indexes = set(range(len(audit_task.completion_conditions)))
    actual_indexes = {item.completion_condition_index for item in milestones}
    steps_by_procedure_index = {
        step.decision.reused_segment_index: step
        for step in agent_steps
        if step.decision.reused_segment_index is not None
    }
    observation_ids = {item.observation_id for item in milestones}
    observation_ids.update(
        step.source_observation_id
        for item in milestones
        if (step := steps_by_procedure_index.get(item.procedure_index)) is not None
    )
    observation_ids.add(final_observation_id)
    try:
        observations = ObservationStore(context.workspace.root).load_selected(observation_ids)
    except ObservationCaptureError:
        return _failed_replay_verification(
            audit_task.objective_id,
            final_observation_id,
            "Current-run milestone observations are unavailable for goal verification.",
        )
    observations_by_id = {item.observation_id: item for item in observations}
    evidence_reason = _milestone_evidence_reason(
        expected_indexes,
        actual_indexes,
        milestones,
        observations_by_id,
        final_observation_id,
    )
    if evidence_reason is not None:
        return _failed_replay_verification(
            audit_task.objective_id,
            final_observation_id,
            evidence_reason,
        )
    final_observation = observations_by_id[final_observation_id]
    ordered_procedure_steps = tuple(
        sorted(
            steps_by_procedure_index.values(),
            key=lambda step: step.decision.reused_segment_index or 0,
        )
    )
    milestone_evidence = tuple(
        GoalVerificationMilestone(
            completion_condition_index=item.completion_condition_index,
            source_observation=_source_observation(
                item,
                steps_by_procedure_index,
                observations_by_id,
            ),
            observation=observations_by_id[item.observation_id],
            transition_evidence=_transition_evidence(
                steps_by_procedure_index.get(item.procedure_index)
            ),
            supporting_actions=tuple(
                _action_evidence(step)
                for step in ordered_procedure_steps
                if (step.decision.reused_segment_index or 0) <= item.procedure_index
            ),
        )
        for item in milestones
    )
    try:
        evidence_packer = (
            AdaptiveGoalEvidencePacker(
                context.workspace.root,
                auditor_image_constraints,
            )
            if auditor_image_constraints is not None
            else None
        )
        return ModelGoalVerifier(
            model_client,
            auditor_profile_id,
            evidence_packer=evidence_packer,
        ).verify(
            audit_task.objective_id,
            audit_task.objective,
            audit_task.completion_conditions,
            final_observation,
            milestone_evidence=milestone_evidence,
        )
    except ModelInputError:
        return _failed_replay_verification(
            audit_task.objective_id,
            final_observation_id,
            "The auditor profile cannot accept the bounded milestone evidence.",
        )


def _transition_evidence(step: AgentStep | None) -> str | None:
    if step is None or step.outcome is None:
        return None
    return build_goal_transition_evidence(
        step.proposal.action,
        step.proposal.target_expression,
        step.outcome.signals,
    )


def _source_observation(
    milestone: MilestoneEvidence,
    steps_by_procedure_index: dict[int, AgentStep],
    observations_by_id: dict[str, Observation],
) -> Observation | None:
    step = steps_by_procedure_index.get(milestone.procedure_index)
    if step is None:
        return None
    return observations_by_id.get(step.source_observation_id)


def _action_evidence(step: AgentStep) -> GoalVerificationActionEvidence:
    outcome = step.outcome
    return GoalVerificationActionEvidence(
        step_index=step.step_index,
        procedure_index=step.decision.reused_segment_index,
        action=step.proposal.action,
        target_expression=step.proposal.target_expression,
        outcome_disposition=(outcome.disposition if outcome is not None else None),
        outcome_signals=(outcome.signals if outcome is not None else frozenset()),
        text_delivery=(outcome.text_delivery if outcome is not None else None),
    )


def _milestone_evidence_reason(
    expected_indexes: set[int],
    actual_indexes: set[int],
    milestones: tuple[MilestoneEvidence, ...],
    observations_by_id: dict[str, Observation],
    final_observation_id: str,
) -> str | None:
    if actual_indexes != expected_indexes:
        return "Current-run milestone evidence does not cover every completion condition."
    if final_observation_id not in observations_by_id:
        return "The final replay observation is unavailable for goal verification."
    for milestone in milestones:
        observation = observations_by_id.get(milestone.observation_id)
        if observation is None:
            return "A current-run milestone observation is unavailable."
        if (
            observation.redacted_screenshot.artifact_id != milestone.screenshot_artifact_id
            or observation.screen_fingerprint != milestone.screen_fingerprint
        ):
            return "A current-run milestone does not match its persisted observation."
    return None


def _failed_replay_verification(
    objective_id: str,
    final_observation_id: str,
    reason: str,
) -> GoalVerification:
    return GoalVerification(
        verification_id=new_record_identifier(),
        objective_id=objective_id,
        final_observation_id=final_observation_id,
        satisfied=False,
        reason=reason,
    )
