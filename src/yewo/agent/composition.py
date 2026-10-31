"""Production composition for one Device Agent invocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from yewo.agent.actuation import ActionExecutor, ActionOutcomeVerifier
from yewo.agent.app_memory import AppMemory, ScreenComparisonPolicy
from yewo.agent.planner import ActionProposalSource, TaskPlanner
from yewo.agent.runtime import DeviceAgent
from yewo.agent.runtime_records import AgentStepStore
from yewo.agent.screen_memory import AppMemoryActionSource
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
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
    redaction_regions: Sequence[RedactionRegion] = (),
    declared_value_resolver: Callable[[str], str] | None = None,
    proposal_source: ActionProposalSource | None = None,
    planning_memory: AppMemory | None = None,
    recover_application: bool = False,
    screen_comparison: ScreenComparisonPolicy | None = None,
) -> DeviceAgent:
    """Connect the shared runtime to one coordinator-owned automation session."""
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
    return DeviceAgent(
        proposal_source,
        GroundingService(model_client, grounding_profile),
        observations,
        ActionExecutor(
            interaction,
            declared_value_resolver=declared_value_resolver,
        ),
        ActionOutcomeVerifier(),
        AgentStepStore(root),
        application_recovery=(interaction.recover_to_application if recover_application else None),
        screen_comparison=screen_comparison or ScreenComparisonPolicy(),
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
