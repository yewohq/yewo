"""Production composition for one Device Agent invocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from yewo.agent.actuation import ActionExecutor, ActionOutcomeVerifier
from yewo.agent.action_diagnostics import (
    ActionApplicationFailure,
    action_application_failure,
)
from yewo.agent.action_verifier import ModelActionVerifier
from yewo.agent.application_state import ApplicationRecovery, restart_and_rebaseline_application
from yewo.agent.app_memory import AppMemory, ScreenComparisonPolicy
from yewo.agent.grounding_memory import AppMemoryGrounder
from yewo.agent.memory_reconciliation import MemoryReconcilerActionSource
from yewo.agent.planner import ActionProposalSource, TaskPlanner, TaskProgressContext
from yewo.agent.planning_evidence import PlannerDecisionSource
from yewo.agent.replay_rejoin import ReplayRejoinActionSource, ReplayRejoinPlan
from yewo.agent.recovery_advisor import ModelRecoveryAdvisor
from yewo.agent.runtime import DeviceAgent
from yewo.agent.runtime_records import AgentStepStore
from yewo.agent.screen_memory import AppMemoryActionSource
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity, new_record_identifier
from yewo.devices.interaction import DeviceInteraction
from yewo.grounding import GroundingProfile, GroundingService
from yewo.models import (
    LiteLLMModelClient,
    ModelArtifactPolicy,
    ModelArtifactStore,
    ModelClient,
    ModelImageReader,
    ModelProfileRegistry,
)
from yewo.observations import (
    ObservationService,
    ObservationStore,
    RedactionRegion,
    RegionScreenshotRedactor,
    ScreenshotStore,
)
from yewo.runs import RunExecutionContext


def build_device_agent(
    context: RunExecutionContext,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    planner_profile_id: str | None,
    *,
    recovery_profile_id: str | None = None,
    action_verifier_profile_id: str | None = None,
    redaction_regions: Sequence[RedactionRegion] = (),
    declared_value_resolver: Callable[[str], str] | None = None,
    proposal_source: ActionProposalSource | None = None,
    planning_memory: AppMemory | None = None,
    planning_memory_objective_id: str | None = None,
    replay_rejoin_plans: Sequence[ReplayRejoinPlan] = (),
    grounding_memory: AppMemory | None = None,
    recover_application: bool = True,
    screen_comparison: ScreenComparisonPolicy | None = None,
    planner_decision_source: PlannerDecisionSource = PlannerDecisionSource.MODEL,
    reused_flow_id: str | None = None,
    progress_context: TaskProgressContext | None = None,
) -> DeviceAgent:
    """Connect the shared runtime to one coordinator-owned automation session."""
    if recover_application and recovery_profile_id is None:
        raise ValueError("Application recovery requires a recovery-advisor profile.")
    root = context.workspace.root
    interaction = DeviceInteraction.from_session(context.session)
    observations = ObservationService(
        interaction,
        context.session.device.descriptor,
        context.session.application,
        ScreenshotStore(root),
        ObservationStore(root),
        RegionScreenshotRedactor(redaction_regions),
    )
    if proposal_source is None:
        if planner_profile_id is None:
            raise ValueError("Model-backed agent composition requires a planner profile.")
        proposal_source = TaskPlanner(model_client, planner_profile_id)
    if planning_memory is not None:
        proposal_source = AppMemoryActionSource(
            proposal_source,
            planning_memory,
            grounding_profile,
        )
        if planning_memory_objective_id is not None and progress_context is None:
            if planner_profile_id is None:
                raise ValueError("Memory reconciliation requires a planner profile.")
            proposal_source = MemoryReconcilerActionSource(
                proposal_source,
                model_client,
                planner_profile_id,
                planning_memory,
                planning_memory_objective_id,
            )
    if replay_rejoin_plans:
        if planning_memory is None:
            raise ValueError("Replay rejoin requires app memory.")
        for replay_rejoin_plan in replay_rejoin_plans:
            proposal_source = ReplayRejoinActionSource(
                proposal_source,
                planning_memory,
                replay_rejoin_plan,
            )
    grounder = GroundingService(model_client, grounding_profile)
    if grounding_memory is not None:
        grounder = AppMemoryGrounder(grounder, grounding_memory, grounding_profile)

    def read_action_application_failure(action_id: str) -> ActionApplicationFailure | None:
        return action_application_failure(context.session.drain_action_logs(action_id))

    def restart_after_application_failure() -> bool:
        return restart_and_rebaseline_application(
            interaction.is_installed,
            interaction.relaunch,
            context.session.drain_recovery_logs,
            new_record_identifier,
        )

    recovery_advisor = (
        ModelRecoveryAdvisor(model_client, recovery_profile_id, root)
        if recover_application and recovery_profile_id is not None
        else None
    )

    return DeviceAgent(
        proposal_source,
        grounder,
        observations,
        ActionExecutor(
            interaction,
            declared_value_resolver=declared_value_resolver,
        ),
        ActionOutcomeVerifier(),
        AgentStepStore(root),
        application_recovery=ApplicationRecovery(
            interaction.recover_to_application if recover_application else None,
            restart_after_application_failure if recover_application else None,
            read_action_application_failure,
            observations,
        ),
        screen_comparison=screen_comparison or ScreenComparisonPolicy(),
        planner_decision_source=planner_decision_source,
        reused_flow_id=reused_flow_id,
        progress_context=progress_context,
        recovery_advisor=recovery_advisor,
        action_verifier=(
            ModelActionVerifier(model_client, action_verifier_profile_id)
            if action_verifier_profile_id is not None
            else None
        ),
    )


def build_model_client(
    root: Path,
    run_id: str,
    profiles: ModelProfileRegistry,
    scrub_text: Callable[[str], str],
    contains_declared_content: bool,
) -> LiteLLMModelClient:
    """Build one run-scoped model transport shared by agent workflows."""
    sensitivity = (
        ArtifactSensitivity.DECLARED_SENSITIVE
        if contains_declared_content
        else ArtifactSensitivity.DEVICE_CONTENT
    )
    return LiteLLMModelClient(
        profiles,
        ModelArtifactStore(
            root,
            ModelArtifactPolicy(
                sensitivity=sensitivity,
                retention=ArtifactRetention.RUN,
            ),
            scrub_text,
        ),
        ModelImageReader(root),
        scrub_text,
        cache_affinity=run_id,
    )
