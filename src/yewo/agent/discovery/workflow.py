"""Journey-directed discovery through YewoAgent mechanics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import time

from yewo.grounding import GroundingProfile
from yewo.models import ModelProfileRegistry
from yewo.observations import ObservationStore, RedactionRegion
from yewo.runs import InterruptionSignal, RunExecutionContext, RunSessionPolicy, RunWorkflowOutcome

from yewo.agent.app_memory import (
    AppMemory, VerifiedRoute, build_memory_compatibility, derived_memory_identifier,
)
from yewo.agent.composition import build_device_agent, build_model_client
from yewo.agent.discovery_checkpoint import select_discovery_checkpoint, select_verified_route
from yewo.agent.memory_store import AppMemoryStore
from yewo.agent.replay_evidence import load_replay_failure_snapshot
from yewo.agent.replay_rejoin import (
    build_replay_rejoin_plans,
    combine_repaired_route,
    combine_replayed_checkpoint_prefix,
)
from yewo.agent.screen_memory import merge_replay_variants
from yewo.agent.route_replay import (
    ReplayPathKind, RouteReplayResult, RouteReplayTerminal,
    replay_discovery_checkpoint,
    replay_route,
)
from yewo.agent.runtime_records import (
    AgentModePolicy, AgentResult, AgentTask, AgentTerminalReason,
    goal_verification_budget_reason,
    remaining_mode_policy,
)
from yewo.agent.planning_evidence import PlannerDecisionSource
from yewo.agent.discovery.promotion import AppMemoryBuilder
from yewo.agent.discovery.coverage import DISCOVERY_CONTROL_COVERAGE
from yewo.agent.discovery.conclusion import (
    persist_discovery_outcome,
    persist_discovery_replay_attempt,
    save_discovery_memory,
)
from yewo.agent.discovery.narration import (
    narrate_discovery_start,
    narrate_replay_handoff,
    narrate_route_selection,
)
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.goal_verifier import GoalVerification, GoalVerifier, ModelGoalVerifier


class DiscoveryWorkflow:
    """Run shared agent mechanics and promote only accepted discovery evidence."""

    def __init__(
        self,
        request: DiscoveryRequest,
        mode_policy: AgentModePolicy,
        model_profiles: ModelProfileRegistry,
        grounding_profile: GroundingProfile,
        planner_profile_id: str, auditor_profile_id: str, recovery_profile_id: str,
        action_verifier_profile_id: str,
        memory_root: Path | None,
        scrub_text: Callable[[str], str],
        *,
        redaction_regions: Sequence[RedactionRegion] = (),
        declared_value_resolver: Callable[[str], str] | None = None,
        memory_builder: AppMemoryBuilder | None = None,
        memory_store: AppMemoryStore | None = None,
        goal_verifier: GoalVerifier | None = None,
        require_reusable_evidence: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request = request
        self._mode_policy = mode_policy
        self._model_profiles = model_profiles
        self._grounding_profile = grounding_profile
        self._planner_profile_id = planner_profile_id
        self._auditor_profile_id = auditor_profile_id
        self._recovery_profile_id = recovery_profile_id
        self._action_verifier_profile_id = action_verifier_profile_id
        if memory_store is not None:
            self._memory_store = memory_store
        elif memory_root is not None:
            self._memory_store = AppMemoryStore(memory_root)
        else:
            raise ValueError("Discovery requires a Yewo memory root or memory store.")
        self._scrub_text = scrub_text
        self._redaction_regions = tuple(redaction_regions)
        if request.declared_value_names and declared_value_resolver is None:
            raise ValueError("Discovery declared values require a resolver.")
        self._declared_value_resolver = declared_value_resolver
        self._memory_builder = memory_builder or AppMemoryBuilder()
        self._goal_verifier = goal_verifier
        self._require_reusable_evidence = require_reusable_evidence
        self._monotonic = monotonic

    def session_policy(self) -> RunSessionPolicy:
        """Apply declared redaction regions before automation starts."""
        return RunSessionPolicy(recording_redaction_regions=self._redaction_regions)

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
    ) -> RunWorkflowOutcome:
        """Discover, promote accepted evidence, and persist truthful coverage."""
        started_at = self._monotonic()
        narrate_discovery_start(self._request)
        compatibility = build_memory_compatibility(
            context.session.application,
            context.workspace.start.request.application,
        )
        existing = self._memory_store.load(compatibility)
        model_client = build_model_client(
            context.workspace.root,
            context.workspace.start.run_id,
            self._model_profiles,
            self._scrub_text,
            bool(self._redaction_regions),
        )
        objective = self._request.objectives[0]
        route = (
            select_verified_route(
                existing,
                objective.statement,
                objective_id=objective.objective_id,
            )
            if existing is not None
            else None
        )
        checkpoint = (
            select_discovery_checkpoint(existing, objective.objective_id, objective.statement)
            if existing is not None and route is None
            else None
        )
        narrate_route_selection(route, checkpoint)
        replay: RouteReplayResult | None = None
        replay_evidence_snapshot = None
        if route is not None and existing is not None:
            replay = replay_route(
                context,
                existing,
                route,
                self._mode_policy,
                model_client,
                self._grounding_profile,
                interruption, recovery_profile_id=self._recovery_profile_id,
                action_verifier_profile_id=self._action_verifier_profile_id,
                redaction_regions=self._redaction_regions,
                recover_application=True,
                declared_value_resolver=self._declared_value_resolver,
            )
            persist_discovery_replay_attempt(context.workspace, replay)
            if replay.terminal is RouteReplayTerminal.COMPLETED:
                known_ids = {
                    variant.observation_id
                    for screen in existing.screens
                    for variant in screen.variants
                }
                fresh_observation_ids = _result_observation_ids(replay.agent_result) - known_ids
                replay_observations = ObservationStore(context.workspace.root).load_selected(
                    fresh_observation_ids
                )
                learned = merge_replay_variants(
                    existing,
                    {
                        observation.observation_id: observation
                        for observation in replay_observations
                    },
                    source_run_id=context.workspace.start.run_id,
                    updated_at=context.workspace.start.started_at,
                )
                if learned != existing:
                    existing = save_discovery_memory(
                        self._memory_store, learned, context.workspace.root, context.workspace,
                        self._request, replay.agent_result,
                        _existing_coverage(self._request, existing, route), replay, None,
                        existing.memory_id,
                    )
                result = DiscoveryResult(
                    request=self._request,
                    terminal_reason=replay.agent_result.terminal_reason,
                    coverage=_existing_coverage(self._request, existing, route),
                    agent_result=replay.agent_result,
                    replay=replay,
                    memory_id=existing.memory_id,
                )
                return persist_discovery_outcome(
                    context.workspace, result, self._require_reusable_evidence
                )
            if replay.terminal is not RouteReplayTerminal.DIVERGED:
                result = DiscoveryResult(
                    request=self._request,
                    terminal_reason=replay.agent_result.terminal_reason,
                    coverage=_replay_failure_coverage(self._request, replay),
                    agent_result=replay.agent_result,
                    replay=replay,
                    memory_id=existing.memory_id,
                )
                return persist_discovery_outcome(
                    context.workspace, result, self._require_reusable_evidence
                )
        elif checkpoint is not None and existing is not None:
            replay_evidence_snapshot, known_divergent_transition_id = (
                load_replay_failure_snapshot(
                    context.workspace.root.parent, checkpoint.checkpoint_id
                )
            )
            replay = replay_discovery_checkpoint(
                context,
                existing,
                checkpoint,
                self._mode_policy,
                model_client,
                self._grounding_profile,
                interruption,
                planner_profile_id=self._planner_profile_id, objective_id=objective.objective_id,
                recovery_profile_id=self._recovery_profile_id, task=_agent_task(self._request, existing, None),
                action_verifier_profile_id=self._action_verifier_profile_id,
                redaction_regions=self._redaction_regions,
                recover_application=True,
                declared_value_resolver=self._declared_value_resolver,
                known_divergent_transition_id=known_divergent_transition_id,
            )
            persist_discovery_replay_attempt(context.workspace, replay)
            if replay.terminal not in {
                RouteReplayTerminal.COMPLETED,
                RouteReplayTerminal.DIVERGED,
            }:
                result = DiscoveryResult(
                    request=self._request,
                    terminal_reason=replay.agent_result.terminal_reason,
                    coverage=_replay_failure_coverage(self._request, replay),
                    agent_result=replay.agent_result,
                    replay=replay,
                    memory_id=existing.memory_id,
                )
                return persist_discovery_outcome(
                    context.workspace, result, self._require_reusable_evidence,
                    replay_evidence_snapshot,
                )
        if replay is not None:
            narrate_replay_handoff(replay)
        repair_policy = self._mode_policy
        if replay is not None:
            repair_policy = remaining_mode_policy(
                self._mode_policy,
                replay.agent_result.usage,
                self._monotonic() - started_at,
            )
            if repair_policy is None:
                agent_result = replay.agent_result.model_copy(
                    update={
                        "terminal_reason": AgentTerminalReason.BUDGET_EXHAUSTED,
                        "reason": "Stored-route replay consumed the shared discovery budget.",
                    }
                )
                result = DiscoveryResult(
                    request=self._request,
                    terminal_reason=agent_result.terminal_reason,
                    coverage=_budget_coverage(self._request),
                    agent_result=agent_result,
                    replay=replay,
                    memory_id=existing.memory_id if existing is not None else None,
                )
                return persist_discovery_outcome(
                    context.workspace, result, self._require_reusable_evidence,
                    replay_evidence_snapshot,
                )
        agent_result = build_device_agent(
            context,
            model_client,
            self._grounding_profile,
            self._planner_profile_id,
            recovery_profile_id=self._recovery_profile_id, redaction_regions=self._redaction_regions,
            action_verifier_profile_id=self._action_verifier_profile_id,
            recover_application=True,
            declared_value_resolver=self._declared_value_resolver,
            screen_comparison=compatibility.screen_comparison,
            planning_memory=existing, planning_memory_objective_id=objective.objective_id,
            replay_rejoin_plans=_replay_rejoin_plans(replay), grounding_memory=existing,
            progress_context=(replay.progress_context if replay is not None else None),
        ).run(
            _agent_task(self._request, existing, replay),
            repair_policy,
            interruption,
        )
        observations = ObservationStore(context.workspace.root).load_selected(
            _result_observation_ids(agent_result)
        )
        observations_by_id = {
            observation.observation_id: observation for observation in observations
        }
        verification: GoalVerification | None = None
        if agent_result.terminal_reason is AgentTerminalReason.DONE_REQUESTED:
            final_observation = observations_by_id[agent_result.final_observation_id]
            used_model_requests = (
                agent_result.usage.planning_requests + agent_result.usage.grounding_requests
            )
            if replay is not None:
                used_model_requests += (
                    replay.agent_result.usage.planning_requests
                    + replay.agent_result.usage.grounding_requests
                )
            budget_reason = goal_verification_budget_reason(
                self._mode_policy,
                used_model_requests,
                self._monotonic() - started_at,
            )
            if budget_reason is not None:
                verification = GoalVerification(
                    verification_id=derived_memory_identifier(
                        "verification-budget",
                        objective.objective_id,
                        final_observation.observation_id,
                    ),
                    objective_id=objective.objective_id,
                    final_observation_id=final_observation.observation_id,
                    satisfied=False,
                    reason=budget_reason,
                )
            else:
                verifier = self._goal_verifier or ModelGoalVerifier(
                    model_client,
                    self._auditor_profile_id,
                )
                verification = verifier.verify(
                    objective.objective_id,
                    objective.statement,
                    objective.completion_conditions,
                    final_observation,
                )
        promotion = self._memory_builder.promote(
            self._request,
            agent_result,
            observations_by_id,
            compatibility,
            verified_objective_ids=(
                frozenset({objective.objective_id})
                if verification is not None and verification.satisfied
                else frozenset()
            ),
            source_run_id=context.workspace.start.run_id,
            updated_at=context.workspace.start.started_at,
            existing=existing,
        )
        memory = promotion.memory
        coverage = promotion.coverage
        if memory is not None and replay is not None:
            memory = combine_replayed_checkpoint_prefix(
                memory,
                objective.objective_id,
                objective.statement,
                replay.replay_transition_ids[: replay.completed_segment_count],
                context.workspace.start.run_id,
                context.workspace.start.started_at,
            )
        if (
            memory is not None
            and existing is not None
            and route is not None
            and replay is not None
            and replay.terminal is RouteReplayTerminal.DIVERGED
        ):
            repaired = combine_repaired_route(
                memory,
                route,
                replay.completed_segment_count,
                coverage.entries[0].route_id,
            )
            if repaired is not None:
                memory, combined, capability = repaired
                coverage = DiscoveryCoverage(
                    entries=(
                        coverage.entries[0].model_copy(
                            update={
                                "capability_id": capability.capability_id,
                                "route_id": combined.route_id,
                                "evidence_observation_ids": combined.evidence_observation_ids,
                                "reason": "A verified replay prefix and repair reached the state.",
                            }
                        ),
                    )
                )
        if memory is not None and memory != existing:
            memory = save_discovery_memory(
                self._memory_store, memory, context.workspace.root, context.workspace,
                self._request, agent_result, coverage, replay, verification,
                existing.memory_id if existing is not None else None, replay_evidence_snapshot,
            )
        result = DiscoveryResult(
            request=self._request,
            terminal_reason=agent_result.terminal_reason,
            coverage=coverage,
            agent_result=agent_result,
            replay=replay,
            verification=verification,
            memory_id=memory.memory_id if memory is not None else None,
        )
        return persist_discovery_outcome(
            context.workspace, result, self._require_reusable_evidence,
            replay_evidence_snapshot,
        )


def _result_observation_ids(result: AgentResult) -> set[str]:
    identifiers = {result.final_observation_id}
    for step in result.steps:
        identifiers.add(step.source_observation_id)
        identifiers.update(step.settlement_observation_ids)
        if step.settled_observation_id is not None:
            identifiers.add(step.settled_observation_id)
        if step.recovery_observation_id is not None:
            identifiers.add(step.recovery_observation_id)
    return identifiers


def _budget_coverage(request: DiscoveryRequest) -> DiscoveryCoverage:
    objective = request.objectives[0]
    return DiscoveryCoverage(
        entries=(
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.UNRESOLVED,
                reason="The run exhausted the shared replay-and-repair discovery budget.",
            ),
        )
    )


def _agent_task(
    request: DiscoveryRequest,
    existing: AppMemory | None,
    replay: RouteReplayResult | None,
) -> AgentTask:
    objective_statements = tuple(objective.statement for objective in request.objectives)
    memory_facts = _memory_facts(existing)
    if replay is None:
        replay_facts = ()
    elif replay.terminal is RouteReplayTerminal.DIVERGED:
        replay_facts = (
            "The stored path diverged at segment "
            f"{replay.failed_segment_index}. Continue discovery from the current state.",
        )
    else:
        replay_facts = (
            "Yewo resumed the accepted incomplete path. Continue discovery from its endpoint.",
        )
    requested_work = request.priorities or objective_statements
    return AgentTask(
        objective="Discover evidence for: " + objective_statements[0],
        known_facts=(
            *request.known_facts,
            *memory_facts,
            *replay_facts,
        ),
        open_work=(*requested_work, DISCOVERY_CONTROL_COVERAGE),
        completion_conditions=(request.objectives[0].completion_conditions or objective_statements),
        declared_value_names=request.declared_value_names,
        targeted_type_available=any(
            transition.action.input_contract is not None for transition in existing.transitions
        )
        if existing is not None
        else False,
        discovery_mode=True,
    )


def _replay_rejoin_plans(replay: RouteReplayResult | None):
    if replay is None:
        return ()
    return build_replay_rejoin_plans(
        replay.path_id,
        (
            PlannerDecisionSource.VERIFIED_ROUTE
            if replay.path_kind is ReplayPathKind.VERIFIED_ROUTE
            else PlannerDecisionSource.DISCOVERY_CHECKPOINT
        ),
        replay.replay_transition_ids,
        replay.downstream_rejoin_transition_ids,
        replay.rejoin_segment_index,
        replay_diverged=replay.terminal is RouteReplayTerminal.DIVERGED,
    )


def _memory_facts(memory: AppMemory | None) -> tuple[str, ...]:
    if memory is None:
        return ()
    capability_facts = tuple(capability.statement for capability in memory.capabilities)
    return tuple(dict.fromkeys(capability_facts))


def _existing_coverage(
    request: DiscoveryRequest,
    memory: AppMemory,
    route: VerifiedRoute,
) -> DiscoveryCoverage:
    objective = request.objectives[0]
    capability = next(
        item
        for item in memory.capabilities
        if item.objective_id == objective.objective_id and item.route_id == route.route_id
    )
    return DiscoveryCoverage(
        entries=(
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.SUPPORTED,
                capability_id=capability.capability_id,
                route_id=route.route_id,
                evidence_observation_ids=route.evidence_observation_ids,
                reason="The compatible stored route replayed with fresh grounding.",
            ),
        )
    )


def _replay_failure_coverage(
    request: DiscoveryRequest,
    replay: RouteReplayResult,
) -> DiscoveryCoverage:
    state = (
        DiscoveryObjectiveState.BLOCKED
        if replay.terminal
        in {
            RouteReplayTerminal.APPLICATION_FAILURE,
            RouteReplayTerminal.DEVICE_REFUSED,
        }
        else DiscoveryObjectiveState.UNRESOLVED
    )
    return DiscoveryCoverage(
        entries=(
            DiscoveryCoverageEntry(
                objective_id=request.objectives[0].objective_id,
                state=state,
                reason=f"Stored path replay stopped: {replay.reason}",
            ),
        )
    )
