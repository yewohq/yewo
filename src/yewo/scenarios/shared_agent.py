"""Execute scenario work through the existing Device Agent composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from yewo.agent.composition import build_device_agent
from yewo.agent.memory.records import AppMemory
from yewo.agent.runtime.records import AgentModePolicy, AgentResult, AgentTask
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.observations import RedactionRegion
from yewo.runs import InterruptionSignal, RunExecutionContext


class SharedDeviceAgentTaskExecutor:
    """Bind scenario context to the one production Device Agent loop."""

    def __init__(
        self,
        context: RunExecutionContext,
        model_client: ModelClient,
        grounding_profile: GroundingProfile,
        planner_profile_id: str,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
        *,
        recovery_profile_id: str | None = None,
        action_verifier_profile_id: str | None = None,
        screen_analyzer_profile_id: str | None = None,
        redaction_regions: Sequence[RedactionRegion] = (),
        declared_value_resolver: Callable[[str], str] | None = None,
        planning_memory: AppMemory | None = None,
        grounding_memory: AppMemory | None = None,
    ) -> None:
        self._context = context
        self._model_client = model_client
        self._grounding_profile = grounding_profile
        self._planner_profile_id = planner_profile_id
        self._mode_policy = mode_policy
        self._interruption = interruption
        self._recovery_profile_id = recovery_profile_id
        self._action_verifier_profile_id = action_verifier_profile_id
        self._screen_analyzer_profile_id = screen_analyzer_profile_id
        self._redaction_regions = tuple(redaction_regions)
        self._declared_value_resolver = declared_value_resolver
        self._planning_memory = planning_memory
        self._grounding_memory = grounding_memory

    def execute(self, task: AgentTask) -> AgentResult:
        """Run one task without adding a scenario-specific planning loop."""
        agent = build_device_agent(
            self._context,
            self._model_client,
            self._grounding_profile,
            self._planner_profile_id,
            recovery_profile_id=self._recovery_profile_id,
            action_verifier_profile_id=self._action_verifier_profile_id,
            screen_analyzer_profile_id=self._screen_analyzer_profile_id,
            redaction_regions=self._redaction_regions,
            declared_value_resolver=self._declared_value_resolver,
            planning_memory=self._planning_memory,
            grounding_memory=self._grounding_memory,
            recover_application=self._recovery_profile_id is not None,
        )
        return agent.run(task, self._mode_policy, self._interruption)
