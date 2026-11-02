"""Yewo's general mobile-task workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import time
from typing import TYPE_CHECKING

from yewo.agent.app_memory import derived_memory_identifier
from yewo.agent.composition import build_device_agent, build_model_client
from yewo.agent.goal_verifier import (
    GoalVerification,
    ModelGoalVerifier,
    persist_goal_verification,
)
from yewo.agent.memory_store import AppMemoryStore
from yewo.agent.narration import narrate_verification
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentTask,
    AgentTerminalReason,
    agent_terminal_category,
    goal_verification_budget_reason,
)
from yewo.grounding import GroundingProfile
from yewo.models import (
    ModelProfileRegistry,
)
from yewo.observations import ObservationStore, RedactionRegion
from yewo.runs import (
    InterruptionSignal,
    RunExecutionContext,
    RunSessionPolicy,
    RunTerminalCategory,
    RunWorkflowOutcome,
)

if TYPE_CHECKING:
    from yewo.agent.discovery import DiscoveryRequest, DiscoveryWorkflow


class YewoAgent:
    """Operate one mobile application task through the shared Device Agent."""

    def __init__(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        model_profiles: ModelProfileRegistry,
        grounding_profile: GroundingProfile,
        planner_profile_id: str,
        auditor_profile_id: str,
        recovery_profile_id: str,
        action_verifier_profile_id: str,
        scrub_text: Callable[[str], str],
        *,
        memory_root: Path | None = None,
        memory_store: AppMemoryStore | None = None,
        redaction_regions: Sequence[RedactionRegion] = (),
        declared_value_resolver: Callable[[str], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._task = task
        self._mode_policy = mode_policy
        if task.declared_value_names and declared_value_resolver is None:
            raise ValueError("Declared values require a resolver.")
        self._model_profiles = model_profiles
        self._grounding_profile = grounding_profile
        self._planner_profile_id = planner_profile_id
        self._auditor_profile_id = auditor_profile_id
        self._recovery_profile_id = recovery_profile_id
        self._action_verifier_profile_id = action_verifier_profile_id
        self._scrub_text = scrub_text
        if memory_root is not None and memory_store is not None:
            raise ValueError("Configure a Yewo memory root or memory store, not both.")
        self._memory_store = (
            memory_store
            if memory_store is not None
            else AppMemoryStore(memory_root)
            if memory_root is not None
            else None
        )
        self._redaction_regions = tuple(redaction_regions)
        self._declared_value_resolver = declared_value_resolver
        self._monotonic = monotonic

    def session_policy(self) -> RunSessionPolicy:
        """Apply the same declared regions to recordings before session startup."""
        return RunSessionPolicy(recording_redaction_regions=self._redaction_regions)

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
    ) -> RunWorkflowOutcome:
        """Run the shared agent and translate its terminal for the coordinator."""
        started_at = self._monotonic()
        if self._memory_store is not None:
            return self._build_discovery_workflow().execute(context, interruption)
        model_client = build_model_client(
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
            recovery_profile_id=self._recovery_profile_id,
            action_verifier_profile_id=self._action_verifier_profile_id,
            redaction_regions=self._redaction_regions,
            declared_value_resolver=self._declared_value_resolver,
        ).run(self._task, self._mode_policy, interruption)
        if result.terminal_reason is AgentTerminalReason.DONE_REQUESTED:
            objective_id = derived_memory_identifier("objective", self._task.objective)
            observation = ObservationStore(context.workspace.root).load_selected(
                {result.final_observation_id}
            )[0]
            budget_reason = goal_verification_budget_reason(
                self._mode_policy,
                result.usage.planning_requests + result.usage.grounding_requests,
                self._monotonic() - started_at,
            )
            verification = (
                GoalVerification(
                    verification_id=derived_memory_identifier(
                        "verification-budget",
                        objective_id,
                        observation.observation_id,
                    ),
                    objective_id=objective_id,
                    final_observation_id=observation.observation_id,
                    satisfied=False,
                    reason=budget_reason,
                )
                if budget_reason is not None
                else ModelGoalVerifier(model_client, self._auditor_profile_id).verify(
                    objective_id,
                    self._task.objective,
                    self._task.completion_conditions,
                    observation,
                )
            )
            persist_goal_verification(context.workspace.root, verification)
            narrate_verification(
                verification_id=verification.verification_id,
                final_observation_id=verification.final_observation_id,
                satisfied=verification.satisfied,
                reason=verification.reason,
            )
            return RunWorkflowOutcome(
                RunTerminalCategory.COMPLETED
                if verification.satisfied
                else RunTerminalCategory.INCOMPLETE,
                verification.reason,
            )
        return RunWorkflowOutcome(
            agent_terminal_category(result.terminal_reason),
            result.reason,
        )

    def _build_discovery_workflow(self) -> DiscoveryWorkflow:
        from yewo.agent.discovery import DiscoveryWorkflow

        return DiscoveryWorkflow(
            _discovery_request(self._task),
            self._mode_policy,
            self._model_profiles,
            self._grounding_profile,
            self._planner_profile_id,
            self._auditor_profile_id,
            self._recovery_profile_id,
            self._action_verifier_profile_id,
            None,
            self._scrub_text,
            redaction_regions=self._redaction_regions,
            declared_value_resolver=self._declared_value_resolver,
            memory_store=self._memory_store,
            require_reusable_evidence=False,
        )


def _discovery_request(task: AgentTask) -> DiscoveryRequest:
    from yewo.agent.discovery import DiscoveryObjective, DiscoveryRequest

    objective_id = derived_memory_identifier("objective", task.objective)
    return DiscoveryRequest(
        request_id=derived_memory_identifier("discovery", objective_id),
        objectives=(
            DiscoveryObjective(
                objective_id=objective_id,
                statement=task.objective,
                completion_conditions=task.completion_conditions,
            ),
        ),
        known_facts=task.known_facts,
        priorities=task.open_work,
        declared_value_names=task.declared_value_names,
    )
