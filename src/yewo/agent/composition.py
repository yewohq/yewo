"""Production composition for one Device Agent invocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from yewo.agent.actuation import ActionExecutor, ActionOutcomeVerifier
from yewo.agent.planner import TaskPlanner
from yewo.agent.runtime import DeviceAgent
from yewo.agent.runtime_records import AgentStepStore
from yewo.devices.interaction import DeviceInteraction
from yewo.grounding import GroundingProfile, GroundingService
from yewo.models import ModelClient
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
    planner_profile_id: str,
    *,
    redaction_regions: Sequence[RedactionRegion] = (),
    declared_value_resolver: Callable[[str], str] | None = None,
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
    return DeviceAgent(
        TaskPlanner(model_client, planner_profile_id),
        GroundingService(model_client, grounding_profile),
        observations,
        ActionExecutor(
            interaction,
            declared_value_resolver=declared_value_resolver,
        ),
        ActionOutcomeVerifier(),
        AgentStepStore(root),
    )
