"""Execute one compiler-approved procedure without planner repair."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from yewo.agent.actuation import DeclaredValueReference
from yewo.agent.composition import build_device_agent
from yewo.agent.memory.grounding import AppMemoryGrounder
from yewo.agent.memory.records import (
    AppMemory,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.memory.screens import AppMemoryActionSource
from yewo.agent.planning.planner import TaskPlanner
from yewo.agent.planning.goal_verifier import (
    GoalVerification,
    persist_goal_verification,
)
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.replay.procedure.audit import audit_completed_replay
from yewo.agent.replay.procedure.budget import procedure_budget_refusal_reason
from yewo.agent.replay.procedure.grounding import ProcedureGrounder
from yewo.agent.replay.procedure.records import (
    ProcedureDivergence,
    ProcedureRepairEvidence,
    ProcedureReplayResult,
    ProcedureReplaySetup,
    ProcedureReplayTerminal,
    ProcedureResetMethod,
)
from yewo.agent.replay.procedure.settlement import ProcedureSettlementPolicySource
from yewo.agent.replay.procedure.repair import BoundedProcedureRepairActionSource
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.procedure.store import ProcedureReplayStore
from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    EvidenceTask,
    ReplayProcedure,
)
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentTask,
    AgentTerminalReason,
)
from yewo.agent.screen_analysis import ScreenAnalysisFailureLedger
from yewo.applications import ApplicationInstallationChange
from yewo.artifacts import EvidenceApplicationIdentity, EvidenceDeviceIdentity
from yewo.grounding import GroundingProfile, GroundingService, VisualGrounder
from yewo.models import ModelClient, ModelImageConstraints
from yewo.observations import (
    ObservationStore,
    ScreenshotStore,
)
from yewo.runs import InterruptionSignal, RunExecutionContext


def execute_compiled_procedure(
    context: RunExecutionContext,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
    setup: ProcedureReplaySetup,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    interruption: InterruptionSignal,
    *,
    planner_profile_id: str,
    recovery_profile_id: str | None = None,
    auditor_profile_id: str,
    proposal_verifier_profile_id: str | None = None,
    outcome_verifier_profile_id: str | None = None,
    screen_analyzer_profile_id: str | None = None,
    screen_analysis_failures: ScreenAnalysisFailureLedger | None = None,
    auditor_image_constraints: ModelImageConstraints | None = None,
    memory: AppMemory | None = None,
    declared_value_resolver: Callable[[str], str] | None = None,
    verification_task: EvidenceTask | None = None,
    initial_model_requests: int = 0,
) -> ProcedureReplayResult:
    """Run a compiled procedure and audit current-run milestones once."""
    started_at = monotonic()
    _validate_execution(context, procedure, evidence, setup, mode_policy, memory)
    audit_task = verification_task or evidence.task
    if len(audit_task.completion_conditions) != len(evidence.task.completion_conditions):
        raise ValueError("A replay objective match must preserve the completion-condition mapping.")
    required_values = tuple(
        dict.fromkeys(
            step.proposal.text.value_name
            for step in procedure.steps
            if isinstance(step.proposal.text, DeclaredValueReference)
        )
    )
    if required_values and declared_value_resolver is None:
        raise ValueError("This procedure requires declared-value resolution.")

    screenshot_store = ScreenshotStore(context.workspace.root)
    procedure_source = CompiledProcedureActionSource(
        procedure,
        evidence,
        screenshot_loader=screenshot_store.load,
    )
    repair_source = TaskPlanner(model_client, planner_profile_id)
    if memory is not None:
        repair_source = AppMemoryActionSource(
            repair_source,
            memory,
        )
    settlement_policy_source = ProcedureSettlementPolicySource(procedure, memory)
    source = BoundedProcedureRepairActionSource(
        procedure_source,
        repair_source,
        settlement_policy_source=settlement_policy_source,
    )
    fallback: VisualGrounder = GroundingService(model_client, grounding_profile)
    if memory is not None:
        fallback = AppMemoryGrounder(fallback, memory, grounding_profile)
    grounder = ProcedureGrounder(
        fallback,
        source,
        screenshot_store.load,
    )
    comparison = ScreenComparisonPolicy(
        maximum_phash_distance=evidence.state_grouping_maximum_phash_distance,
        maximum_replay_phash_distance=evidence.state_grouping_maximum_phash_distance,
    )
    execution_started_at = monotonic()
    agent_result = build_device_agent(
        context,
        model_client,
        grounding_profile,
        None,
        declared_value_resolver=declared_value_resolver,
        proposal_source=source,
        recovery_profile_id=recovery_profile_id,
        recover_application=recovery_profile_id is not None,
        proposal_verifier_profile_id=proposal_verifier_profile_id,
        outcome_verifier_profile_id=outcome_verifier_profile_id,
        screen_analyzer_profile_id=screen_analyzer_profile_id,
        screen_analysis_failures=screen_analysis_failures,
        screen_comparison=comparison,
        planner_decision_source=PlannerDecisionSource.COMPILED_PROCEDURE,
        reused_flow_id=procedure.procedure_id,
        visual_grounder=grounder,
        initial_planning_requests=initial_model_requests,
    ).run(
        AgentTask(
            objective=f"Replay compiled prefix: {evidence.task.objective}",
            known_facts=("Use only compiler-approved procedure actions.",),
            open_work=evidence.task.open_work,
            completion_conditions=("Every compiled expectation passes.",),
            declared_value_names=required_values,
            targeted_type_available=any(step.proposal.text is not None for step in procedure.steps),
            discovery_mode=True,
        ),
        mode_policy,
        interruption,
    )
    procedure_execution_ms = (monotonic() - execution_started_at) * 1_000
    divergences = source.divergences
    divergence = divergences[-1] if divergences else None
    if agent_result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS and divergence is None:
        divergence = _runtime_divergence(context, procedure, source, agent_result.reason)
        divergences = (divergence,)
    repairs = source.repairs
    repair = source.repair_evidence
    audit_started_at = monotonic()
    verification = (
        audit_completed_replay(
            context,
            audit_task,
            source.milestones,
            agent_result.final_observation_id,
            agent_result.usage.planning_requests + agent_result.usage.grounding_requests,
            monotonic() - started_at,
            mode_policy,
            model_client,
            auditor_profile_id,
            agent_result.steps,
            auditor_image_constraints,
        )
        if _can_audit_replay(agent_result.terminal_reason, source, divergence, repair)
        else None
    )
    final_audit_ms = (monotonic() - audit_started_at) * 1_000
    terminal = _procedure_terminal(
        agent_result.terminal_reason,
        source,
        divergence,
        repair,
        verification,
    )
    result = ProcedureReplayResult(
        procedure_id=procedure.procedure_id,
        evidence_id=evidence.evidence_id,
        source_run_id=evidence.run.run_id,
        visual_policy_version=procedure.visual_policy_version,
        setup_id=setup.setup_id,
        application=procedure.application,
        terminal=terminal,
        procedure_step_count=len(procedure.steps),
        completed_step_count=source.completed_step_count,
        failed_procedure_index=(divergence.procedure_index if divergence is not None else None),
        milestones=source.milestones,
        grounding_decisions=grounder.decisions,
        divergences=divergences,
        repairs=repairs,
        environment_events=source.environment_events,
        verification=verification,
        reason=verification.reason if verification is not None else agent_result.reason,
        procedure_execution_ms=procedure_execution_ms,
        final_audit_ms=final_audit_ms,
        total_elapsed_ms=(monotonic() - started_at) * 1_000,
        agent_result=agent_result,
    )
    ProcedureReplayStore(context.workspace.root).persist(result)
    if verification is not None:
        persist_goal_verification(context.workspace.root, verification)
    return result


def _validate_execution(
    context: RunExecutionContext,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
    setup: ProcedureReplaySetup,
    mode_policy: AgentModePolicy,
    memory: AppMemory | None,
) -> None:
    if procedure.evidence_id != evidence.evidence_id:
        raise ValueError("The procedure does not match the evidence document.")
    if setup.application != procedure.application:
        raise ValueError("The replay setup does not match the procedure application.")
    if setup.setup_contract_sha256 != procedure.setup_contract.content_sha256():
        raise ValueError("The replay setup does not match the procedure setup contract.")
    if setup.method.value != procedure.setup_contract.reset_method:
        raise ValueError("The replay reset method does not match the procedure setup contract.")
    current_application = _application_identity(context)
    if current_application != procedure.application:
        raise ValueError("The active application build does not match the procedure.")
    if setup.method is ProcedureResetMethod.FRESH_INSTALL and (
        context.session.installation_change
        not in {
            ApplicationInstallationChange.INSTALLED,
            ApplicationInstallationChange.REINSTALLED,
        }
    ):
        raise ValueError("Fresh-install replay requires an observed installation change.")
    device = context.session.device.descriptor
    current_device = EvidenceDeviceIdentity(
        target_id=device.target_id,
        platform=device.platform,
        kind=device.kind,
        operating_system_version=device.operating_system_version,
        model=device.device_model,
        architecture=device.architecture,
    )
    if current_device != procedure.setup_contract.device:
        raise ValueError("The active device does not match the procedure setup contract.")
    budget_refusal = procedure_budget_refusal_reason(mode_policy, procedure)
    if budget_refusal is not None:
        raise ValueError(budget_refusal)
    actions = {step.proposal.action for step in procedure.steps}
    available_budget = (
        mode_policy.action_limits.allowed_actions & mode_policy.action_limits.supported_actions
    )
    if not actions <= available_budget:
        raise ValueError("The active mode does not allow every compiled procedure action.")
    if memory is not None and memory.compatibility.application != procedure.application:
        raise ValueError("Grounding memory does not match the procedure application.")


def _application_identity(context: RunExecutionContext) -> EvidenceApplicationIdentity:
    application = context.session.application
    return EvidenceApplicationIdentity(
        platform=application.platform,
        application_id=application.application_id,
        version=application.version,
        build=application.build,
        artifact_sha256=application.artifact_sha256,
    )


def _runtime_divergence(
    context: RunExecutionContext,
    procedure: ReplayProcedure,
    source: CompiledProcedureActionSource | BoundedProcedureRepairActionSource,
    reason: str,
) -> ProcedureDivergence:
    observations = ObservationStore(context.workspace.root).load_all()
    if not observations:
        raise ValueError("Procedure divergence requires a persisted live observation.")
    step = procedure.steps[source.next_procedure_index]
    actual = observations[-1]
    return ProcedureDivergence(
        procedure_id=procedure.procedure_id,
        procedure_index=step.procedure_index,
        source_step=step.source_step,
        guard="runtime",
        expected_state_id=step.source_state_id,
        actual_observation_id=actual.observation_id,
        actual_fingerprint=actual.screen_fingerprint,
        reason=reason,
    )


def _procedure_terminal(
    terminal: AgentTerminalReason,
    source: CompiledProcedureActionSource | BoundedProcedureRepairActionSource,
    divergence: ProcedureDivergence | None,
    repair: ProcedureRepairEvidence | None,
    verification: GoalVerification | None,
) -> ProcedureReplayTerminal:
    direct_terminal = {
        AgentTerminalReason.APPLICATION_EXITED: ProcedureReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.APPLICATION_INACTIVE: ProcedureReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.DEVICE_REFUSED: ProcedureReplayTerminal.DEVICE_REFUSED,
        AgentTerminalReason.BUDGET_EXHAUSTED: ProcedureReplayTerminal.BUDGET_EXHAUSTED,
    }.get(terminal)
    if direct_terminal is not None:
        return direct_terminal
    if terminal is AgentTerminalReason.DONE_REQUESTED:
        return _completed_replay_terminal(
            source,
            divergence,
            repair,
            verification,
        )
    if divergence is not None:
        return ProcedureReplayTerminal.DIVERGED
    if terminal is AgentTerminalReason.CANNOT_PROGRESS:
        raise ValueError("Procedure divergence requires guard evidence.")
    raise ValueError(f"Unsupported procedure terminal: {terminal.value}.")


def _completed_replay_terminal(
    source: CompiledProcedureActionSource | BoundedProcedureRepairActionSource,
    divergence: ProcedureDivergence | None,
    repair: ProcedureRepairEvidence | None,
    verification: GoalVerification | None,
) -> ProcedureReplayTerminal:
    if divergence is not None and (repair is None or not repair.succeeded):
        return ProcedureReplayTerminal.DIVERGED
    if not source.replay_complete:
        raise ValueError("Replay completion was requested before all expectations passed.")
    if verification is None:
        raise ValueError("Replay completion requires final goal verification.")
    if verification.satisfied:
        return ProcedureReplayTerminal.COMPLETED
    return ProcedureReplayTerminal.REPLAY_COMPLETE


def _can_audit_replay(
    terminal: AgentTerminalReason,
    source: CompiledProcedureActionSource | BoundedProcedureRepairActionSource,
    divergence: ProcedureDivergence | None,
    repair: ProcedureRepairEvidence | None,
) -> bool:
    """Audit only a complete clean replay or a deterministically rejoined replay."""
    return (
        terminal is AgentTerminalReason.DONE_REQUESTED
        and source.replay_complete
        and (divergence is None or (repair is not None and repair.succeeded))
    )
