"""Scenario-local execution of verified app-level route knowledge."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum
from typing import Protocol

from pydantic import Field, model_validator

from yewo.discovery.artifacts import (
    DiscoveryCompatibilityKey,
    DiscoveryContract,
    verify_artifact,
)
from yewo.discovery.knowledge import AppMap, RouteMemory
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionResult,
    RouteMemoryExecutor,
    RouteMemoryLiveObservation,
    RouteMemoryReusePolicy,
    RouteMemoryRuntime,
    RouteMemoryTerminal,
)


class WarmExecutionTerminal(str, Enum):
    READY = "ready"
    UNVERIFIED = "unverified"
    INTERRUPTED = "interrupted"


class LocalRediscoveryTerminal(str, Enum):
    COMPLETE = "complete"
    NOT_FOUND = "not_found"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERRUPTED = "interrupted"


class LocalRediscoveryBudget(DiscoveryContract):
    """Hard limits for one failed route segment, not an app exploration."""

    observations: int = Field(default=6, ge=1, le=12)
    actions: int = Field(default=3, ge=0, le=6)
    model_proposals: int = Field(default=2, ge=0, le=4)
    duration_ms: int = Field(default=30_000, ge=1, le=60_000)
    cost_micro_usd: int = Field(default=25_000, ge=0, le=100_000)


class QuarantinedMapProposal(DiscoveryContract):
    """Run evidence that post-run policy may review but runtime cannot apply."""

    proposal_id: str
    source_observation_id: str
    destination_observation_id: str
    action_kind: str
    quarantine_reason: str = "Awaiting post-run verification."


class LocalRediscoveryRequest(DiscoveryContract):
    scenario_uuid: str
    route_memory_artifact_id: str
    capability: str
    semantic_instruction: str
    failed_segment_id: str | None
    completed_segment_ids: tuple[str, ...] = ()
    budget: LocalRediscoveryBudget


class ExecutionCost(DiscoveryContract):
    duration_ms: int = Field(ge=0)
    actions: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    cost_micro_usd: int = Field(ge=0)
    cost_unknown_calls: int = Field(default=0, ge=0)


class LocalRediscoveryResult(DiscoveryContract):
    terminal: LocalRediscoveryTerminal
    reason: str
    observations: int = Field(ge=0)
    actions: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    cost_micro_usd: int = Field(ge=0)
    proposed_updates: tuple[QuarantinedMapProposal, ...] = ()

    @model_validator(mode="after")
    def _validate_usage(self) -> "LocalRediscoveryResult":
        if self.terminal is LocalRediscoveryTerminal.COMPLETE and not self.reason:
            raise ValueError("Completed local rediscovery requires a reason.")
        return self


class ScenarioExecutionCosts(DiscoveryContract):
    discovery: ExecutionCost = Field(
        default_factory=lambda: ExecutionCost(
            duration_ms=0,
            actions=0,
            model_calls=0,
            cost_micro_usd=0,
        )
    )
    route_reuse: ExecutionCost
    local_rediscovery: ExecutionCost
    scenario: ExecutionCost | None = None


class ScenarioWarmExecution(DiscoveryContract):
    scenario_uuid: str
    app_map_artifact_id: str
    route_memory_ids: tuple[str, ...]
    terminal: WarmExecutionTerminal
    route_results: tuple[RouteMemoryExecutionResult, ...] = ()
    local_rediscovery: LocalRediscoveryResult | None = None
    quarantined_proposals: tuple[QuarantinedMapProposal, ...] = ()
    costs: ScenarioExecutionCosts
    reason: str


class ScenarioRouteSelection(Protocol):
    scenario_uuid: str
    route_memory_ids: list[str]


class ScenarioRouteRuntimeContext(Protocol):
    @property
    def initial_observation(self) -> RouteMemoryLiveObservation: ...

    @property
    def runtime(self) -> RouteMemoryRuntime: ...


class ScenarioRouteRuntimeFactory(Protocol):
    def build(
        self,
        route_memory: RouteMemory,
        *,
        context_id: str,
    ) -> ScenarioRouteRuntimeContext: ...


class LocalRediscovery(Protocol):
    def recover(self, request: LocalRediscoveryRequest) -> LocalRediscoveryResult: ...


class LiveRouteRuntimeProvider(Protocol):
    """Bind shared route mechanics to one scenario's live device context."""

    def runtime_factory_for(
        self,
        scenario: ScenarioRouteSelection,
        device_control: object,
        perception: object,
    ) -> ScenarioRouteRuntimeFactory: ...


class LiveLocalRediscoveryProvider(Protocol):
    def rediscovery_for(
        self,
        scenario: ScenarioRouteSelection,
        device_control: object,
        perception: object,
    ) -> LocalRediscovery | None: ...


class ScenarioRoutePreparer(Protocol):
    """Device-facing boundary invoked after one scenario's reset."""

    def prepare(
        self,
        scenario: ScenarioRouteSelection,
        device_control: object,
        perception: object,
    ) -> ScenarioWarmExecution: ...


def unavailable_warm_execution(
    scenario: ScenarioRouteSelection,
    reason: str,
) -> ScenarioWarmExecution:
    """Represent a strict route requirement that no execution boundary can prove."""
    empty_cost = ExecutionCost(
        duration_ms=0,
        actions=0,
        model_calls=0,
        cost_micro_usd=0,
    )
    return ScenarioWarmExecution(
        scenario_uuid=scenario.scenario_uuid,
        app_map_artifact_id="",
        route_memory_ids=tuple(scenario.route_memory_ids),
        terminal=WarmExecutionTerminal.UNVERIFIED,
        costs=ScenarioExecutionCosts(
            route_reuse=empty_cost,
            local_rediscovery=empty_cost,
        ),
        reason=reason,
    )


class WarmScenarioCoordinator:
    """Reuse immutable routes and bound recovery to the first failed segment."""

    def __init__(
        self,
        app_map: AppMap,
        route_memories: tuple[RouteMemory, ...],
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not verify_artifact(app_map):
            raise ValueError("Warm execution requires a sealed AppMap.")
        memories_by_id = {memory.header.artifact_id: memory for memory in route_memories}
        if len(memories_by_id) != len(route_memories):
            raise ValueError("Warm execution RouteMemory identifiers must be unique.")
        for memory in route_memories:
            if not verify_artifact(memory):
                raise ValueError("Warm execution requires sealed RouteMemory.")
            if memory.source_app_map_artifact_id != app_map.header.artifact_id:
                raise ValueError("Warm execution RouteMemory belongs to another AppMap.")
            if not memory.header.compatibility.is_compatible_with(app_map.header.compatibility):
                raise ValueError("Warm execution RouteMemory is incompatible with its AppMap.")
        self.app_map = app_map
        self.memories_by_id = memories_by_id
        self.clock = clock

    def prepare(
        self,
        scenario: ScenarioRouteSelection,
        compatibility: DiscoveryCompatibilityKey,
        runtime_factory: ScenarioRouteRuntimeFactory,
        *,
        local_rediscovery: LocalRediscovery | None = None,
        rediscovery_budget: LocalRediscoveryBudget | None = None,
        reuse_counts: dict[str, int] | None = None,
        reuse_policy: RouteMemoryReusePolicy | None = None,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> ScenarioWarmExecution:
        route_ids = tuple(scenario.route_memory_ids)
        unknown_ids = tuple(
            route_id for route_id in route_ids if route_id not in self.memories_by_id
        )
        if unknown_ids:
            return self._result(
                scenario,
                route_ids,
                WarmExecutionTerminal.UNVERIFIED,
                (),
                None,
                0,
                0,
                f"Scenario references unavailable RouteMemory: {', '.join(unknown_ids)}.",
            )

        started_at = self.clock()
        route_results: list[RouteMemoryExecutionResult] = []
        route_actions = 0
        for route_id in route_ids:
            if interrupted():
                return self._result(
                    scenario,
                    route_ids,
                    WarmExecutionTerminal.INTERRUPTED,
                    route_results,
                    None,
                    self._duration_ms(started_at),
                    route_actions,
                    "Warm scenario execution was interrupted.",
                )
            memory = self.memories_by_id[route_id]
            context = runtime_factory.build(
                memory,
                context_id=f"scenario:{scenario.scenario_uuid}:route:{route_id}",
            )
            route_result = RouteMemoryExecutor().execute(
                memory,
                compatibility,
                context.initial_observation,
                context.runtime,
                context_id=f"scenario:{scenario.scenario_uuid}:route:{route_id}",
                successful_reuse_count=(reuse_counts or {}).get(route_id, 0),
                reuse_policy=reuse_policy,
                interrupted=interrupted,
            )
            route_results.append(route_result)
            route_actions += len(route_result.dispatched_segment_ids)
            if route_result.terminal is RouteMemoryTerminal.COMPLETE:
                continue
            if route_result.terminal is RouteMemoryTerminal.INTERRUPTED:
                return self._result(
                    scenario,
                    route_ids,
                    WarmExecutionTerminal.INTERRUPTED,
                    route_results,
                    None,
                    self._duration_ms(started_at),
                    route_actions,
                    route_result.reason,
                )
            route_duration_ms = self._duration_ms(started_at)
            rediscovery = self._rediscover(
                scenario,
                memory,
                route_result,
                local_rediscovery,
                rediscovery_budget or LocalRediscoveryBudget(),
            )
            terminal = (
                WarmExecutionTerminal.READY
                if rediscovery.terminal is LocalRediscoveryTerminal.COMPLETE
                else WarmExecutionTerminal.INTERRUPTED
                if rediscovery.terminal is LocalRediscoveryTerminal.INTERRUPTED
                else WarmExecutionTerminal.UNVERIFIED
            )
            return self._result(
                scenario,
                route_ids,
                terminal,
                route_results,
                rediscovery,
                route_duration_ms,
                route_actions,
                rediscovery.reason,
            )

        return self._result(
            scenario,
            route_ids,
            WarmExecutionTerminal.READY,
            route_results,
            None,
            self._duration_ms(started_at),
            route_actions,
            "Every selected RouteMemory reached its guarded destination.",
        )

    @staticmethod
    def _rediscover(
        scenario: ScenarioRouteSelection,
        memory: RouteMemory,
        route_result: RouteMemoryExecutionResult,
        local_rediscovery: LocalRediscovery | None,
        budget: LocalRediscoveryBudget,
    ) -> LocalRediscoveryResult:
        if local_rediscovery is None:
            return LocalRediscoveryResult(
                terminal=LocalRediscoveryTerminal.NOT_FOUND,
                reason=(
                    "Route reuse stopped at the first failed live predicate. "
                    "No local rediscovery boundary was configured."
                ),
                observations=0,
                actions=0,
                model_calls=0,
                duration_ms=0,
                cost_micro_usd=0,
            )
        result = local_rediscovery.recover(
            LocalRediscoveryRequest(
                scenario_uuid=scenario.scenario_uuid,
                route_memory_artifact_id=memory.header.artifact_id,
                capability=memory.capability,
                semantic_instruction=memory.semantic_instruction,
                failed_segment_id=route_result.quarantined_segment_id,
                completed_segment_ids=route_result.completed_segment_ids,
                budget=budget,
            )
        )
        if (
            result.observations > budget.observations
            or result.actions > budget.actions
            or result.model_calls > budget.model_proposals
            or result.duration_ms > budget.duration_ms
            or result.cost_micro_usd > budget.cost_micro_usd
        ):
            return LocalRediscoveryResult(
                terminal=LocalRediscoveryTerminal.BUDGET_EXHAUSTED,
                reason="Local rediscovery reported usage beyond its authorized budget.",
                observations=result.observations,
                actions=result.actions,
                model_calls=result.model_calls,
                duration_ms=result.duration_ms,
                cost_micro_usd=result.cost_micro_usd,
                proposed_updates=result.proposed_updates,
            )
        return result

    def _duration_ms(self, started_at: float) -> int:
        return max(0, round((self.clock() - started_at) * 1000))

    def _result(
        self,
        scenario: ScenarioRouteSelection,
        route_ids: tuple[str, ...],
        terminal: WarmExecutionTerminal,
        route_results: list[RouteMemoryExecutionResult] | tuple[RouteMemoryExecutionResult, ...],
        rediscovery: LocalRediscoveryResult | None,
        route_duration_ms: int,
        route_actions: int,
        reason: str,
    ) -> ScenarioWarmExecution:
        local_cost = ExecutionCost(
            duration_ms=rediscovery.duration_ms if rediscovery else 0,
            actions=rediscovery.actions if rediscovery else 0,
            model_calls=rediscovery.model_calls if rediscovery else 0,
            cost_micro_usd=rediscovery.cost_micro_usd if rediscovery else 0,
        )
        proposals = rediscovery.proposed_updates if rediscovery else ()
        return ScenarioWarmExecution(
            scenario_uuid=scenario.scenario_uuid,
            app_map_artifact_id=self.app_map.header.artifact_id,
            route_memory_ids=route_ids,
            terminal=terminal,
            route_results=tuple(route_results),
            local_rediscovery=rediscovery,
            quarantined_proposals=proposals,
            costs=ScenarioExecutionCosts(
                route_reuse=ExecutionCost(
                    duration_ms=route_duration_ms,
                    actions=route_actions,
                    model_calls=0,
                    cost_micro_usd=0,
                ),
                local_rediscovery=local_cost,
            ),
            reason=reason,
        )


class VerifiedScenarioRoutePreparer:
    """Runner adapter over sealed knowledge and scenario-local live mechanics."""

    def __init__(
        self,
        app_map: AppMap,
        route_memories: tuple[RouteMemory, ...],
        compatibility: DiscoveryCompatibilityKey,
        runtime_provider: LiveRouteRuntimeProvider,
        *,
        rediscovery_provider: LiveLocalRediscoveryProvider | None = None,
        rediscovery_budget: LocalRediscoveryBudget | None = None,
        reuse_policy: RouteMemoryReusePolicy | None = None,
        interrupted: Callable[[], bool] = lambda: False,
    ):
        self.coordinator = WarmScenarioCoordinator(app_map, route_memories)
        self.compatibility = compatibility
        self.runtime_provider = runtime_provider
        self.rediscovery_provider = rediscovery_provider
        self.rediscovery_budget = rediscovery_budget or LocalRediscoveryBudget()
        self.reuse_policy = reuse_policy
        self.interrupted = interrupted
        self.reuse_counts = {memory.header.artifact_id: 0 for memory in route_memories}

    def prepare(
        self,
        scenario: ScenarioRouteSelection,
        device_control: object,
        perception: object,
    ) -> ScenarioWarmExecution:
        runtime_factory = self.runtime_provider.runtime_factory_for(
            scenario,
            device_control,
            perception,
        )
        rediscovery = (
            self.rediscovery_provider.rediscovery_for(
                scenario,
                device_control,
                perception,
            )
            if self.rediscovery_provider is not None
            else None
        )
        result = self.coordinator.prepare(
            scenario,
            self.compatibility,
            runtime_factory,
            local_rediscovery=rediscovery,
            rediscovery_budget=self.rediscovery_budget,
            reuse_counts=dict(self.reuse_counts),
            reuse_policy=self.reuse_policy,
            interrupted=self.interrupted,
        )
        if result.terminal is WarmExecutionTerminal.READY:
            for route_result in result.route_results:
                if route_result.terminal is RouteMemoryTerminal.COMPLETE:
                    route_id = route_result.route_memory_artifact_id
                    self.reuse_counts[route_id] = self.reuse_counts.get(route_id, 0) + 1
        return result
