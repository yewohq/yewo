"""Compile current-run evidence early enough to audit its milestones."""

from dataclasses import dataclass
from pathlib import Path

from yewo.agent.discovery.coverage import budget_coverage
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.memory.records import AppMemory, MemoryCompatibility, ScreenComparisonPolicy
from yewo.agent.planning.goal_verifier import (
    GoalVerification,
    GoalVerificationActionEvidence,
    GoalVerificationMilestone,
    build_goal_transition_evidence,
)
from yewo.agent.replay.procedure_store import CompiledProcedureStore
from yewo.agent.replay.reconciliation_records import (
    EvidenceDocument,
    EvidenceStep,
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
)
from yewo.agent.replay.preparation import compile_completed_run
from yewo.agent.replay.reconciliation_store import (
    EVIDENCE_DOCUMENT_PATH,
    REPLAY_PROCEDURE_PATH,
)
from yewo.models import ModelClient
from yewo.observations import Observation
from yewo.agent.runtime.records import AgentResult, AgentTerminalReason


@dataclass(frozen=True)
class DiscoveryCompilationMilestones:
    """One provisional compilation and its audit evidence links."""

    report: ProcedureCompilationReport
    evidence: EvidenceDocument | None = None
    procedure: ReplayProcedure | None = None

    def observation_ids(self) -> set[str]:
        """Return observations captured by compiled milestone steps."""
        return {
            observation_id
            for _, _, step in self._milestone_steps()
            for observation_id in (step.source_observation_id, step.destination_observation_id)
            if observation_id is not None
        }

    def goal_milestones(
        self,
        observations_by_id: dict[str, Observation],
    ) -> tuple[GoalVerificationMilestone, ...]:
        """Convert compiler mappings into final-auditor evidence."""
        if self.evidence is None or self.procedure is None:
            return ()
        evidence_by_step = {step.step: step for step in self.evidence.steps}
        return tuple(
            GoalVerificationMilestone(
                completion_condition_index=condition_index,
                source_observation=observations_by_id[source.source_observation_id],
                observation=observations_by_id[source.destination_observation_id],
                transition_evidence=build_goal_transition_evidence(
                    source.action,
                    source.proposal.target_expression,
                    source.signals,
                ),
                supporting_actions=tuple(
                    _compiled_action_evidence(
                        evidence_by_step[procedure_step.source_step],
                        procedure_step.procedure_index,
                    )
                    for procedure_step in self.procedure.steps
                    if procedure_step.procedure_index <= procedure_index
                ),
            )
            for condition_index, procedure_index, source in self._milestone_steps()
            if source.destination_observation_id is not None
        )

    def _milestone_steps(self) -> tuple[tuple[int, int, EvidenceStep], ...]:
        if self.evidence is None or self.procedure is None:
            return ()
        evidence_by_step = {step.step: step for step in self.evidence.steps}
        captures: list[tuple[int, int, EvidenceStep]] = []
        for procedure_step in self.procedure.steps:
            evidence_step = evidence_by_step[procedure_step.source_step]
            if evidence_step.destination_observation_id is None:
                continue
            captures.extend(
                (condition_index, procedure_step.procedure_index, evidence_step)
                for condition_index in procedure_step.captures_completion_conditions
            )
        return tuple(sorted(captures))


def _compiled_action_evidence(
    step: EvidenceStep,
    procedure_index: int,
) -> GoalVerificationActionEvidence:
    return GoalVerificationActionEvidence(
        step_index=step.recorded_step_index,
        procedure_index=procedure_index,
        action=step.action,
        target_expression=step.proposal.target_expression,
        outcome_disposition=step.outcome_disposition,
        outcome_signals=step.signals,
        text_delivery=step.text_delivery,
    )


def compile_discovery_milestones(
    *,
    root: Path,
    run_id: str,
    result: DiscoveryResult,
    model_client: ModelClient,
    model_profile_id: str,
    comparison: ScreenComparisonPolicy,
    memory: AppMemory | None = None,
) -> DiscoveryCompilationMilestones:
    """Compile without durable promotion and expose mapped milestone evidence."""
    report = compile_completed_run(
        root=root,
        run_id=run_id,
        result=result,
        model_client=model_client,
        model_profile_id=model_profile_id,
        comparison=comparison,
        memory=memory,
    )
    if report.status is not ProcedureCompilationStatus.COMPILED:
        return DiscoveryCompilationMilestones(report=report)
    evidence = EvidenceDocument.model_validate_json((root / EVIDENCE_DOCUMENT_PATH).read_bytes())
    procedure = ReplayProcedure.model_validate_json((root / REPLAY_PROCEDURE_PATH).read_bytes())
    return DiscoveryCompilationMilestones(
        report=report,
        evidence=evidence,
        procedure=procedure,
    )


def compile_completed_discovery(
    has_procedure_replay: bool,
    root: Path,
    run_id: str,
    request: DiscoveryRequest,
    agent_result: AgentResult,
    model_client: ModelClient,
    model_profile_id: str,
    comparison: ScreenComparisonPolicy,
    memory: AppMemory | None = None,
) -> DiscoveryCompilationMilestones | None:
    """Compile one newly discovered completed path before its final audit."""
    if (
        has_procedure_replay
        or agent_result.terminal_reason is not AgentTerminalReason.DONE_REQUESTED
    ):
        return None
    provisional_result = DiscoveryResult(
        request=request,
        terminal_reason=agent_result.terminal_reason,
        coverage=budget_coverage(request),
        agent_result=agent_result,
    )
    return compile_discovery_milestones(
        root=root,
        run_id=run_id,
        result=provisional_result,
        model_client=model_client,
        model_profile_id=model_profile_id,
        comparison=comparison,
        memory=memory,
    )


def save_verified_compilation(
    compilation: DiscoveryCompilationMilestones | None,
    verification: GoalVerification | None,
    store: CompiledProcedureStore | None,
    compatibility: MemoryCompatibility,
) -> None:
    """Promote a compiled procedure only after independent goal verification."""
    if (
        compilation is None
        or verification is None
        or not verification.satisfied
        or compilation.evidence is None
        or compilation.procedure is None
        or store is None
    ):
        return
    store.save(compatibility, compilation.evidence, compilation.procedure)


def verified_compiled_procedure(
    compilation: DiscoveryCompilationMilestones | None,
    verification: GoalVerification | None,
) -> ReplayProcedure | None:
    """Return the procedure that passed independent final verification."""
    if compilation is None or verification is None or not verification.satisfied:
        return None
    return compilation.procedure
