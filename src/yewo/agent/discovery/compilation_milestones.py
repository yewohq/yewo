"""Compile current-run evidence early enough to audit its milestones."""

from dataclasses import dataclass
from pathlib import Path

from yewo.agent.discovery.coverage import budget_coverage
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.memory.records import AppMemory, MemoryCompatibility
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.goal_verifier import (
    GoalVerification,
    GoalVerificationActionEvidence,
    GoalVerificationMilestone,
    build_goal_transition_evidence,
)
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    EvidenceStep,
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
    ReconcilerResponse,
)
from yewo.agent.replay.reconciliation.compilation import compile_completed_discovery_run
from yewo.agent.replay.reconciliation.store import (
    EVIDENCE_DOCUMENT_PATH,
    RECONCILER_RESPONSE_PATH,
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
    response: ReconcilerResponse | None = None
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
        if self.evidence is None:
            return ()
        evidence_by_step = {step.step: step for step in self.evidence.steps}
        return tuple(
            GoalVerificationMilestone(
                completion_condition_index=condition_index,
                source_observation=observations_by_id[source.source_observation_id],
                observation=observations_by_id[source.destination_observation_id],
                transition_evidence=build_goal_transition_evidence(
                    source.proposal.action,
                    source.proposal.target_expression,
                    source.signals,
                ),
                supporting_actions=self._supporting_actions(
                    procedure_index,
                    source,
                    evidence_by_step,
                ),
            )
            for condition_index, procedure_index, source in self._milestone_steps()
            if source.destination_observation_id is not None
        )

    def _milestone_steps(self) -> tuple[tuple[int, int | None, EvidenceStep], ...]:
        if self.evidence is None:
            return ()
        evidence_by_step = {step.step: step for step in self.evidence.steps}
        if self.procedure is None:
            return self._response_milestone_steps(evidence_by_step)
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

    def _response_milestone_steps(
        self,
        evidence_by_step: dict[int, EvidenceStep],
    ) -> tuple[tuple[int, int | None, EvidenceStep], ...]:
        evidence = self.evidence
        if self.response is None or evidence is None:
            return ()
        allowed_indices = set(range(len(evidence.task.completion_conditions)))
        procedure_indices = {
            source_step: procedure_index
            for procedure_index, source_step in enumerate(self.response.procedure)
            if source_step in evidence_by_step
        }
        return tuple(
            sorted(
                (
                    (
                        condition_index,
                        procedure_indices.get(source_step),
                        evidence_by_step[source_step],
                    )
                    for condition_index, source_step in self.response.milestones.items()
                    if condition_index in allowed_indices
                    and source_step in evidence_by_step
                    and evidence_by_step[source_step].destination_observation_id is not None
                ),
                key=lambda item: (item[0], item[2].step),
            ),
        )

    def _supporting_actions(
        self,
        maximum_index: int | None,
        milestone_source: EvidenceStep,
        evidence_by_step: dict[int, EvidenceStep],
    ) -> tuple[GoalVerificationActionEvidence, ...]:
        if self.procedure is not None:
            if maximum_index is None:
                raise AssertionError("A compiled milestone requires a procedure index.")
            return tuple(
                _compiled_action_evidence(
                    evidence_by_step[step.source_step],
                    step.procedure_index,
                )
                for step in self.procedure.steps
                if step.procedure_index <= maximum_index
            )
        if self.evidence is None or self.response is None:
            return ()
        selected_actions = tuple(
            (index, evidence_by_step[source_step])
            for index, source_step in enumerate(self.response.procedure)
            if source_step in evidence_by_step and source_step <= milestone_source.step
        )
        milestone_action = (
            () if milestone_source.step in self.response.procedure else ((None, milestone_source),)
        )
        return tuple(
            _compiled_action_evidence(step, index)
            for index, step in sorted(
                (*selected_actions, *milestone_action),
                key=lambda item: item[1].step,
            )
        )


def _compiled_action_evidence(
    step: EvidenceStep,
    procedure_index: int | None,
) -> GoalVerificationActionEvidence:
    return GoalVerificationActionEvidence(
        step_index=step.step,
        procedure_index=procedure_index,
        action=step.proposal.action,
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
    report = compile_completed_discovery_run(
        root=root,
        run_id=run_id,
        discovery_result=result,
        model_client=model_client,
        model_profile_id=model_profile_id,
        comparison=comparison,
        memory=memory,
    )
    evidence_path = root / EVIDENCE_DOCUMENT_PATH
    response_path = root / RECONCILER_RESPONSE_PATH
    evidence = (
        EvidenceDocument.model_validate_json(evidence_path.read_bytes())
        if evidence_path.is_file()
        else None
    )
    response = (
        ReconcilerResponse.model_validate_json(response_path.read_bytes())
        if response_path.is_file()
        else None
    )
    procedure_path = root / REPLAY_PROCEDURE_PATH
    procedure = (
        ReplayProcedure.model_validate_json(procedure_path.read_bytes())
        if report.status is ProcedureCompilationStatus.COMPILED and procedure_path.is_file()
        else None
    )
    return DiscoveryCompilationMilestones(
        report=report,
        evidence=evidence,
        response=response,
        procedure=procedure,
    )


def compile_completed_discovery(
    procedure_replay: ProcedureReplayResult | None,
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
    if agent_result.terminal_reason is not AgentTerminalReason.DONE_REQUESTED:
        return None
    provisional_result = DiscoveryResult(
        request=request,
        terminal_reason=agent_result.terminal_reason,
        coverage=budget_coverage(request),
        agent_result=agent_result,
        procedure_replay=procedure_replay,
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
    source_run_root: Path,
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
    store.save(
        compatibility,
        compilation.evidence,
        compilation.procedure,
        source_run_root=source_run_root,
    )


def verified_compiled_procedure(
    compilation: DiscoveryCompilationMilestones | None,
    verification: GoalVerification | None,
) -> ReplayProcedure | None:
    """Return the procedure that passed independent final verification."""
    if compilation is None or verification is None or not verification.satisfied:
        return None
    return compilation.procedure
