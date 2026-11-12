"""Select compatible fresh-install procedure replay before discovery."""

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.memory.records import AppMemory, MemoryCompatibility
from yewo.agent.discovery.narration import (
    narrate_compiled_procedure_selection,
    narrate_no_compiled_procedure,
)
from yewo.agent.discovery.coverage import procedure_replay_coverage
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.discovery.preparation import DiscoveryPreparation
from yewo.agent.replay.procedure.budget import procedure_budget_refusal_reason
from yewo.agent.replay.procedure.execution import execute_compiled_procedure
from yewo.agent.replay.procedure.action_risks import compiled_procedure_action_risk_reasons
from yewo.agent.replay.procedure.records import (
    ProcedureReplayResult,
    ProcedureReplaySetup,
    ProcedureResetMethod,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.procedure.store import (
    CompiledProcedureSchemaError,
    CompiledProcedureStore,
)
from yewo.agent.replay.reconciliation.records import (
    CompilerReason,
    EvidenceTask,
    ReplayProcedure,
)
from yewo.agent.runtime.records import AgentModePolicy
from yewo.agent.screen_analysis import ScreenAnalysisFailureLedger
from yewo.applications import ApplicationInstallationChange
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient, ModelImageConstraints
from yewo.run_narration import current_narrator
from yewo.runs import InterruptionSignal, RunExecutionContext


@dataclass(frozen=True)
class ProcedureReplayEntry:
    """Compiled-procedure attempt selected at discovery entry."""

    replay: ProcedureReplayResult | None
    conclusion: DiscoveryResult | None


def prepare_procedure_replay_entry(
    context: RunExecutionContext,
    compatibility: MemoryCompatibility,
    request: DiscoveryRequest,
    policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    auditor_profile_id: str,
    planner_profile_id: str,
    recovery_profile_id: str,
    proposal_verifier_profile_id: str,
    outcome_verifier_profile_id: str,
    interruption: InterruptionSignal,
    procedure_store: CompiledProcedureStore | None,
    auditor_image_constraints: ModelImageConstraints,
    memory: AppMemory | None,
    declared_value_resolver: Callable[[str], str] | None,
    preparation: DiscoveryPreparation | None = None,
    screen_analyzer_profile_id: str | None = None,
    screen_analysis_failures: ScreenAnalysisFailureLedger | None = None,
) -> ProcedureReplayEntry:
    """Attempt one compatible compiled procedure before bounded discovery."""
    objective = request.objectives[0]
    replay = replay_compiled_procedure_if_available(
        context,
        compatibility,
        objective.objective_id,
        objective.statement,
        policy,
        model_client,
        grounding_profile,
        auditor_profile_id,
        interruption,
        procedure_store,
        planner_profile_id=planner_profile_id,
        recovery_profile_id=recovery_profile_id,
        proposal_verifier_profile_id=proposal_verifier_profile_id,
        outcome_verifier_profile_id=outcome_verifier_profile_id,
        screen_analyzer_profile_id=screen_analyzer_profile_id,
        screen_analysis_failures=screen_analysis_failures,
        auditor_image_constraints=auditor_image_constraints,
        memory=memory,
        declared_value_resolver=declared_value_resolver,
        preparation=preparation,
        verification_task=(
            EvidenceTask(
                objective_id=objective.objective_id,
                objective=objective.statement,
                completion_conditions=objective.completion_conditions,
            )
            if objective.completion_conditions
            else None
        ),
    )
    if replay is None:
        narrate_no_compiled_procedure()
    conclusion = procedure_terminal_discovery_result(
        request,
        replay,
        memory.memory_id if memory is not None else None,
    )
    return ProcedureReplayEntry(replay=replay, conclusion=conclusion)


def replay_compiled_procedure_if_available(
    context: RunExecutionContext,
    compatibility: MemoryCompatibility,
    objective_id: str,
    objective: str,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    auditor_profile_id: str,
    interruption: InterruptionSignal,
    procedure_store: CompiledProcedureStore | None,
    *,
    planner_profile_id: str | None = None,
    recovery_profile_id: str | None = None,
    proposal_verifier_profile_id: str | None = None,
    outcome_verifier_profile_id: str | None = None,
    screen_analyzer_profile_id: str | None = None,
    screen_analysis_failures: ScreenAnalysisFailureLedger | None = None,
    auditor_image_constraints: ModelImageConstraints | None = None,
    memory: AppMemory | None,
    declared_value_resolver: Callable[[str], str] | None,
    preparation: DiscoveryPreparation | None = None,
    verification_task: EvidenceTask | None = None,
) -> ProcedureReplayResult | None:
    """Replay a prepared procedure only after an observed fresh installation."""
    if procedure_store is None or context.session.installation_change not in {
        ApplicationInstallationChange.INSTALLED,
        ApplicationInstallationChange.REINSTALLED,
    }:
        return None
    selection = preparation.procedure_selection if preparation is not None else None
    stored_objective_id = selection.stored_objective_id if selection is not None else objective_id
    stored_objective = selection.stored_objective if selection is not None else objective
    try:
        bundle = procedure_store.load(
            compatibility,
            stored_objective_id,
            stored_objective,
        )
    except CompiledProcedureSchemaError:
        return None
    if bundle is None:
        return None
    if selection is not None and bundle.procedure.procedure_id != selection.procedure_id:
        return None
    if (
        selection is None
        and verification_task is not None
        and bundle.evidence.task.completion_conditions != verification_task.completion_conditions
    ):
        return None
    action_risk_reasons = compiled_procedure_action_risk_reasons(
        bundle.evidence,
        bundle.procedure,
        memory,
    )
    if action_risk_reasons:
        _narrate_action_risk_refusal(bundle.procedure, action_risk_reasons)
        return None
    if not _policy_can_execute(mode_policy, bundle.procedure):
        return None
    if planner_profile_id is None:
        raise ValueError("Compiled procedure repair requires a planner profile.")
    procedure_store.stage_screenshots(bundle, context.workspace.root)
    setup = ProcedureReplaySetup(
        setup_id=new_record_identifier(),
        method=ProcedureResetMethod.FRESH_INSTALL,
        application=bundle.procedure.application,
        setup_contract_sha256=bundle.procedure.setup_contract.content_sha256(),
    )
    narrate_compiled_procedure_selection(
        bundle.procedure.procedure_id,
        len(bundle.procedure.steps),
    )
    return execute_compiled_procedure(
        context,
        bundle.procedure,
        bundle.evidence,
        setup,
        mode_policy,
        model_client,
        grounding_profile,
        interruption,
        planner_profile_id=planner_profile_id,
        recovery_profile_id=recovery_profile_id,
        auditor_profile_id=auditor_profile_id,
        proposal_verifier_profile_id=proposal_verifier_profile_id,
        outcome_verifier_profile_id=outcome_verifier_profile_id,
        screen_analyzer_profile_id=screen_analyzer_profile_id,
        screen_analysis_failures=screen_analysis_failures,
        auditor_image_constraints=auditor_image_constraints,
        memory=memory,
        declared_value_resolver=declared_value_resolver,
        verification_task=verification_task,
        initial_model_requests=(
            selection.preparation_model_requests if selection is not None else 0
        ),
    )


def _policy_can_execute(
    policy: AgentModePolicy,
    procedure: ReplayProcedure,
) -> bool:
    if procedure_budget_refusal_reason(policy, procedure) is not None:
        return False
    available_actions = (
        policy.action_limits.allowed_actions & policy.action_limits.supported_actions
    )
    return all(step.proposal.action in available_actions for step in procedure.steps)


def _narrate_action_risk_refusal(
    procedure: ReplayProcedure,
    reasons: tuple[CompilerReason, ...],
) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    action_risk_ids = tuple(
        reason.action_risk_id for reason in reasons if reason.action_risk_id is not None
    )
    narrator.record(
        "procedure-action-risk-gate",
        "A compiled procedure was not replayed because it matches known application action risks.",
    )
    narrator.trace(
        "procedure-action-risk-gate",
        f"procedure={procedure.procedure_id} action_risk_ids={','.join(action_risk_ids)}",
    )


def procedure_terminal_discovery_result(
    request: DiscoveryRequest,
    replay: ProcedureReplayResult | None,
    memory_id: str | None,
) -> DiscoveryResult | None:
    """Return a discovery conclusion when procedure replay must stop the run."""
    if replay is None or (
        replay.terminal
        in {ProcedureReplayTerminal.REPLAY_COMPLETE, ProcedureReplayTerminal.DIVERGED}
    ):
        return None
    return DiscoveryResult(
        request=request,
        terminal_reason=replay.agent_result.terminal_reason,
        coverage=procedure_replay_coverage(request, replay),
        agent_result=replay.agent_result,
        procedure_replay=replay,
        verification=(
            replay.verification if replay.terminal is ProcedureReplayTerminal.COMPLETED else None
        ),
        compiled_procedure_id=(
            replay.procedure_id if replay.terminal is ProcedureReplayTerminal.COMPLETED else None
        ),
        compiled_procedure_step_count=(
            replay.procedure_step_count
            if replay.terminal is ProcedureReplayTerminal.COMPLETED
            else None
        ),
        memory_id=memory_id,
    )
