"""Production composition for one Device Agent invocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from yewo.agent.actuation import ActionExecutor, ActionOutcomeVerifier
from yewo.agent.actuation.diagnostics import (
    ActionApplicationFailure,
    action_application_failure,
)
from yewo.agent.actuation.outcome_verifier import ModelOutcomeVerifier
from yewo.agent.actuation.proposal_verifier import ModelProposalVerifier
from yewo.agent.runtime.application_state import (
    ApplicationRecovery,
    restart_and_rebaseline_application,
)
from yewo.agent.memory.records import AppMemory
from yewo.observations.service import ObservationSource
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.memory.grounding import AppMemoryGrounder
from yewo.agent.planning.planner import (
    ActionProposalSource,
    TaskPlanner,
)
from yewo.agent.planning.records import (
    TaskProgressContext,
)
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.recovery.advisor import ModelRecoveryAdvisor
from yewo.agent.runtime.agent import DeviceAgent
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysisManager,
    ModelScreenAnalyzer,
    ScreenAnalysisFailureLedger,
    ScreenAnalysisStore,
)
from yewo.agent.runtime.records import AgentStepStore
from yewo.agent.memory.screens import AppMemoryActionSource
from yewo.artifacts import ArtifactRetention, new_record_identifier
from yewo.devices.interaction import DeviceInteraction
from yewo.grounding import GroundingProfile, GroundingService, VisualGrounder
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
    ScreenshotStore,
)
from yewo.runs import RunExecutionContext
from yewo.extensions import ExtensionRegistry


def build_device_agent(
    context: RunExecutionContext,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    planner_profile_id: str | None,
    *,
    recovery_profile_id: str | None = None,
    proposal_verifier_profile_id: str | None = None,
    outcome_verifier_profile_id: str | None = None,
    screen_analyzer_profile_id: str | None = None,
    declared_value_resolver: Callable[[str], str] | None = None,
    proposal_source: ActionProposalSource | None = None,
    planning_memory: AppMemory | None = None,
    grounding_memory: AppMemory | None = None,
    recover_application: bool = True,
    screen_comparison: ScreenComparisonPolicy | None = None,
    planner_decision_source: PlannerDecisionSource = PlannerDecisionSource.MODEL,
    reused_flow_id: str | None = None,
    progress_context: TaskProgressContext | None = None,
    initial_planning_requests: int = 0,
    visual_grounder: VisualGrounder | None = None,
    screen_analysis_failures: ScreenAnalysisFailureLedger | None = None,
) -> DeviceAgent:
    """Connect the shared runtime to one coordinator-owned automation session."""
    if recover_application and recovery_profile_id is None:
        raise ValueError("Application recovery requires a recovery-advisor profile.")
    root = context.workspace.root
    declared_values = context.declared_values
    extensions = context.extensions
    interaction = DeviceInteraction.from_session(context.session)
    observations = ObservationService(
        extensions.wrapped(ObservationSource, interaction),
        context.session.device.descriptor,
        context.session.application,
        ScreenshotStore(root),
        ObservationStore(root),
    )
    if proposal_source is None:
        if planner_profile_id is None:
            raise ValueError("Model-backed agent composition requires a planner profile.")
        proposal_source = TaskPlanner(model_client, planner_profile_id)
    if planning_memory is not None:
        proposal_source = AppMemoryActionSource(
            proposal_source,
            planning_memory,
        )
    grounder = visual_grounder or GroundingService(model_client, grounding_profile)
    if visual_grounder is not None and grounding_memory is not None:
        raise ValueError("Injected grounding cannot also request composition memory.")
    if visual_grounder is None and grounding_memory is not None:
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
            declared_values=declared_values,
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
        proposal_verifier=(
            ModelProposalVerifier(model_client, proposal_verifier_profile_id)
            if proposal_verifier_profile_id is not None
            else None
        ),
        predicted_outcome_verifier=(
            ModelOutcomeVerifier(model_client, outcome_verifier_profile_id, ScreenshotStore(root))
            if outcome_verifier_profile_id is not None
            else None
        ),
        screen_analysis=(
            CurrentScreenAnalysisManager(
                ModelScreenAnalyzer(model_client, screen_analyzer_profile_id),
                store=ScreenAnalysisStore(root),
                failure_ledger=screen_analysis_failures,
            )
            if screen_analyzer_profile_id is not None
            else None
        ),
        initial_planning_requests=initial_planning_requests,
    )


def build_model_client(
    root: Path,
    run_id: str,
    profiles: ModelProfileRegistry,
    declared_value_names: Sequence[str],
    extensions: ExtensionRegistry,
) -> ModelClient:
    """Build one run-scoped model transport shared by agent workflows."""
    client = LiteLLMModelClient(
        profiles,
        ModelArtifactStore(
            root,
            ModelArtifactPolicy(
                retention=ArtifactRetention.RUN,
            ),
        ),
        ModelImageReader(root),
        cache_affinity=run_id,
        ai_session_id=run_id,
    )
    return extensions.wrapped(ModelClient, client)
