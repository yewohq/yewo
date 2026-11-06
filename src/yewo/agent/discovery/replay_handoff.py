"""Shared compiled-procedure replay evidence passed into bounded discovery."""

from collections.abc import Mapping

from yewo.agent.planning.goal_evidence import GoalEvidencePacker
from yewo.agent.planning.goal_verifier import (
    GoalVerification,
    GoalVerificationActionEvidence,
    GoalVerificationMilestone,
    GoalVerifier,
    ModelGoalVerifier,
    build_goal_transition_evidence,
)
from yewo.agent.memory.records import derived_memory_identifier
from yewo.agent.discovery.coverage import budget_coverage
from yewo.agent.discovery.records import DiscoveryObjective, DiscoveryRequest, DiscoveryResult
from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentTerminalReason,
    remaining_mode_policy,
)
from yewo.agent.runtime.verification import goal_verification_budget_reason
from yewo.models import ModelClient
from yewo.observations import Observation


def result_observation_ids(result: AgentResult) -> set[str]:
    """Return every observation referenced by one agent result."""
    identifiers = {result.final_observation_id}
    for step in result.steps:
        identifiers.add(step.source_observation_id)
        identifiers.update(step.settlement_observation_ids)
        if step.settled_observation_id is not None:
            identifiers.add(step.settled_observation_id)
        if step.recovery_observation_id is not None:
            identifiers.add(step.recovery_observation_id)
    return identifiers


def replay_observation_ids(
    result: AgentResult,
    procedure_replay: ProcedureReplayResult | None,
) -> set[str]:
    """Return repair observations plus completed procedure milestones."""
    identifiers = result_observation_ids(result)
    if procedure_replay is not None:
        identifiers.update(result_observation_ids(procedure_replay.agent_result))
    return identifiers


def replay_goal_milestones(
    procedure_replay: ProcedureReplayResult | None,
    observations_by_id: Mapping[str, Observation],
) -> tuple[GoalVerificationMilestone, ...]:
    """Build final-audit evidence from completed procedure milestones."""
    if procedure_replay is None:
        return ()
    steps_by_procedure_index = {
        step.decision.reused_segment_index: step
        for step in procedure_replay.agent_result.steps
        if step.decision.reused_segment_index is not None
    }
    ordered_procedure_steps = tuple(
        sorted(
            steps_by_procedure_index.values(),
            key=lambda step: step.decision.reused_segment_index or 0,
        )
    )
    return tuple(
        GoalVerificationMilestone(
            completion_condition_index=item.completion_condition_index,
            source_observation=_replay_source_observation(
                steps_by_procedure_index.get(item.procedure_index),
                observations_by_id,
            ),
            observation=observations_by_id[item.observation_id],
            transition_evidence=_replay_transition_evidence(
                steps_by_procedure_index.get(item.procedure_index)
            ),
            supporting_actions=tuple(
                _replay_action_evidence(step)
                for step in ordered_procedure_steps
                if (step.decision.reused_segment_index or 0) <= item.procedure_index
            ),
        )
        for item in procedure_replay.milestones
    )


def _replay_transition_evidence(step: AgentStep | None) -> str | None:
    if step is None or step.outcome is None:
        return None
    return build_goal_transition_evidence(
        step.proposal.action,
        step.proposal.target_expression,
        step.outcome.signals,
    )


def _replay_source_observation(
    step: AgentStep | None,
    observations_by_id: Mapping[str, Observation],
) -> Observation | None:
    if step is None:
        return None
    return observations_by_id.get(step.source_observation_id)


def _replay_action_evidence(step: AgentStep) -> GoalVerificationActionEvidence:
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


def verify_goal_after_replay(
    verifier: GoalVerifier,
    objective: DiscoveryObjective,
    final_observation: Observation,
    procedure_replay: ProcedureReplayResult | None,
    observations_by_id: Mapping[str, Observation],
    discovery_milestones: tuple[GoalVerificationMilestone, ...] = (),
) -> GoalVerification:
    """Audit repair output with any completed procedure milestones."""
    milestones = (
        replay_goal_milestones(procedure_replay, observations_by_id) or discovery_milestones
    )
    if not objective.completion_conditions:
        milestones = tuple(
            item.model_copy(update={"completion_condition_index": None}) for item in milestones
        )
    return verifier.verify(
        objective.objective_id,
        objective.statement,
        objective.completion_conditions,
        final_observation,
        milestone_evidence=milestones,
    )


def verify_discovery_goal(
    objective: DiscoveryObjective,
    agent_result: AgentResult,
    observations_by_id: Mapping[str, Observation],
    policy: AgentModePolicy,
    used_model_requests: int,
    elapsed_seconds: float,
    procedure_replay: ProcedureReplayResult | None,
    model_client: ModelClient,
    auditor_profile_id: str,
    verifier: GoalVerifier | None,
    discovery_milestones: tuple[GoalVerificationMilestone, ...] = (),
    evidence_packer: GoalEvidencePacker | None = None,
) -> GoalVerification | None:
    """Audit a completed repair with retained procedure evidence."""
    if agent_result.terminal_reason is not AgentTerminalReason.DONE_REQUESTED:
        return None
    final_observation = observations_by_id[agent_result.final_observation_id]
    budget_reason = goal_verification_budget_reason(policy, used_model_requests, elapsed_seconds)
    if budget_reason is not None:
        return GoalVerification(
            verification_id=derived_memory_identifier(
                "verification-budget",
                objective.objective_id,
                final_observation.observation_id,
            ),
            objective_id=objective.objective_id,
            final_observation_id=final_observation.observation_id,
            satisfied=False,
            reason=budget_reason,
        )
    active_verifier = verifier or ModelGoalVerifier(
        model_client,
        auditor_profile_id,
        evidence_packer=evidence_packer,
    )
    return verify_goal_after_replay(
        active_verifier,
        objective,
        final_observation,
        procedure_replay,
        observations_by_id,
        discovery_milestones,
    )


def exhausted_replay_budget_result(
    request: DiscoveryRequest,
    procedure_replay: ProcedureReplayResult | None,
    memory_id: str | None,
) -> DiscoveryResult | None:
    """Build the shared-budget conclusion for compiled-procedure replay."""
    if procedure_replay is None:
        return None
    agent_result = procedure_replay.agent_result.model_copy(
        update={
            "terminal_reason": AgentTerminalReason.BUDGET_EXHAUSTED,
            "reason": "Replay consumed the shared discovery budget.",
        }
    )
    return DiscoveryResult(
        request=request,
        terminal_reason=agent_result.terminal_reason,
        coverage=budget_coverage(request),
        agent_result=agent_result,
        procedure_replay=procedure_replay,
        memory_id=memory_id,
    )


def remaining_policy_after_replay(
    policy: AgentModePolicy,
    procedure_replay: ProcedureReplayResult | None,
    elapsed_seconds: float,
) -> AgentModePolicy | None:
    """Return the shared budget after compiled-procedure replay."""
    if procedure_replay is not None:
        remaining = remaining_mode_policy(
            policy,
            procedure_replay.agent_result.usage,
            elapsed_seconds,
        )
        if remaining is None or remaining.maximum_model_requests is None:
            return remaining
        audit_calls = (
            procedure_replay.verification.model_call_count
            if procedure_replay.verification is not None
            else 0
        )
        maximum_model_requests = remaining.maximum_model_requests - audit_calls
        return (
            remaining.model_copy(update={"maximum_model_requests": maximum_model_requests})
            if maximum_model_requests > 0
            else None
        )
    return policy


def replay_model_request_count(procedure_replay: ProcedureReplayResult | None) -> int:
    """Return model requests consumed by compiled-procedure replay."""
    if procedure_replay is None:
        return 0
    return (
        procedure_replay.agent_result.usage.planning_requests
        + procedure_replay.agent_result.usage.grounding_requests
        + (
            procedure_replay.verification.model_call_count
            if procedure_replay.verification is not None
            else 0
        )
    )
