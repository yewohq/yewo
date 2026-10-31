"""Normal-task workflow for the shared Device Agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from yewo.agent.composition import build_device_agent
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentTask,
    AgentTerminalReason,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.grounding import GroundingProfile
from yewo.models import (
    LiteLLMModelClient,
    ModelArtifactPolicy,
    ModelArtifactStore,
    ModelImageReader,
    ModelProfileRegistry,
)
from yewo.observations import RedactionRegion
from yewo.runs import (
    InterruptionSignal,
    RunExecutionContext,
    RunSessionPolicy,
    RunTerminalCategory,
    RunWorkflowOutcome,
)


class NormalAgentWorkflow:
    """Run one immediate task without discovery or scenario dependencies."""

    def __init__(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        model_profiles: ModelProfileRegistry,
        grounding_profile: GroundingProfile,
        planner_profile_id: str,
        scrub_text: Callable[[str], str],
        *,
        redaction_regions: Sequence[RedactionRegion] = (),
        declared_value_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._task = task
        self._mode_policy = mode_policy
        if task.declared_value_names and declared_value_resolver is None:
            raise ValueError("Declared values require a resolver.")
        self._model_profiles = model_profiles
        self._grounding_profile = grounding_profile
        self._planner_profile_id = planner_profile_id
        self._scrub_text = scrub_text
        self._redaction_regions = tuple(redaction_regions)
        self._declared_value_resolver = declared_value_resolver

    def session_policy(self) -> RunSessionPolicy:
        """Apply the same declared regions to recordings before session startup."""
        return RunSessionPolicy(recording_redaction_regions=self._redaction_regions)

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
    ) -> RunWorkflowOutcome:
        """Run the shared agent and translate its terminal for the coordinator."""
        model_client = _build_model_client(
            context.workspace.root,
            context.workspace.start.run_id,
            self._model_profiles,
            self._scrub_text,
            bool(self._redaction_regions),
        )
        result = build_device_agent(
            context,
            model_client,
            self._grounding_profile,
            self._planner_profile_id,
            redaction_regions=self._redaction_regions,
            declared_value_resolver=self._declared_value_resolver,
        ).run(self._task, self._mode_policy, interruption)
        return RunWorkflowOutcome(_classify_terminal(result.terminal_reason), result.reason)


def _build_model_client(
    root: Path,
    run_id: str,
    profiles: ModelProfileRegistry,
    scrub_text: Callable[[str], str],
    contains_declared_content: bool,
) -> LiteLLMModelClient:
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


def _classify_terminal(terminal: AgentTerminalReason) -> RunTerminalCategory:
    return {
        AgentTerminalReason.DONE_REQUESTED: RunTerminalCategory.COMPLETED,
        AgentTerminalReason.CANNOT_PROGRESS: RunTerminalCategory.INCOMPLETE,
        AgentTerminalReason.BUDGET_EXHAUSTED: RunTerminalCategory.BUDGET_EXHAUSTED,
        AgentTerminalReason.APPLICATION_EXITED: RunTerminalCategory.APPLICATION_FAILURE,
        AgentTerminalReason.APPLICATION_INACTIVE: RunTerminalCategory.INCOMPLETE,
        AgentTerminalReason.DEVICE_REFUSED: RunTerminalCategory.INCOMPLETE,
    }[terminal]
