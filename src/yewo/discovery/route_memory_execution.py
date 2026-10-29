"""Guarded live execution for immutable RouteMemory artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import Field

from yewo.agent.actuation.actions import Action, ActionType, PlannerTapAction
from yewo.agent.actuation.safety import (
    ActionAuthorization,
    ActionSafetyPolicy,
    ActionSource,
)
from yewo.agent.memory.control_identity import ControlIdentity
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.artifacts import DiscoveryCompatibilityKey, DiscoveryContract, verify_artifact
from yewo.discovery.contracts import DiscoveryObservation
from yewo.discovery.knowledge import (
    ControlReacquisition,
    RouteMemory,
    RouteMemoryLifecycle,
    RouteMemorySegment,
    RouteStateGuard,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt


class RouteMemoryReusePolicy(DiscoveryContract):
    """Periodic verification policy for successful memory reuse."""

    shadow_revalidation_interval: int = Field(gt=0)


class RouteMemoryTerminal(str, Enum):
    COMPLETE = "complete"
    QUARANTINED = "quarantined"
    INELIGIBLE = "ineligible"
    SHADOW_REVALIDATION_REQUIRED = "shadow_revalidation_required"
    INTERRUPTED = "interrupted"


class RouteMemoryExecutionResult(DiscoveryContract):
    route_memory_artifact_id: str
    terminal: RouteMemoryTerminal
    completed_segment_ids: tuple[str, ...] = ()
    dispatched_segment_ids: tuple[str, ...] = ()
    quarantined_segment_id: str | None = None
    reason: str


class RouteMemoryLiveObservation(DiscoveryContract):
    observation: DiscoveryObservation
    screen_family_id: str | None = None
    screen_size: tuple[int, int]


@dataclass(frozen=True)
class RouteTargetResolution:
    """Ephemeral live action resolved without persisting its coordinates."""

    source_observation_id: str
    historical_control_identity: str
    matched_role: str
    matched_text_digest: str
    action: Action | PlannerTapAction
    live_control_identity: ControlIdentity


@dataclass(frozen=True)
class RouteMemoryExecutionReceipt:
    dispatch: DiscoveryDispatchReceipt
    settlement: SettlementReceipt
    destination: RouteMemoryLiveObservation


class RouteMemoryRuntime(Protocol):
    def reacquire(
        self,
        target: ControlReacquisition,
        action_kind: str,
        observation: RouteMemoryLiveObservation,
    ) -> RouteTargetResolution | None: ...

    def execute(
        self,
        authorization: ActionAuthorization,
        resolution: RouteTargetResolution,
    ) -> RouteMemoryExecutionReceipt: ...


def is_shadow_revalidation_due(
    successful_reuse_count: int,
    policy: RouteMemoryReusePolicy,
) -> bool:
    if successful_reuse_count < 0:
        raise ValueError("RouteMemory reuse count cannot be negative.")
    return (successful_reuse_count + 1) % policy.shadow_revalidation_interval == 0


class RouteMemoryExecutor:
    """Validate and dispatch one live-reacquired segment at a time."""

    def __init__(self) -> None:
        self.action_policy = ActionSafetyPolicy()

    def execute(
        self,
        route_memory: RouteMemory,
        compatibility: DiscoveryCompatibilityKey,
        initial: RouteMemoryLiveObservation,
        runtime: RouteMemoryRuntime,
        *,
        context_id: str,
        successful_reuse_count: int = 0,
        reuse_policy: RouteMemoryReusePolicy | None = None,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> RouteMemoryExecutionResult:
        eligibility = self._eligibility_terminal(
            route_memory,
            compatibility,
            successful_reuse_count,
            reuse_policy,
        )
        if eligibility is not None:
            return eligibility
        current = initial
        completed: list[str] = []
        dispatched: list[str] = []
        for segment in route_memory.segments:
            if interrupted():
                return self._result(
                    route_memory,
                    RouteMemoryTerminal.INTERRUPTED,
                    completed,
                    "RouteMemory execution was interrupted.",
                    dispatched_segment_ids=dispatched,
                )
            refusal = _guard_refusal(segment.source, current)
            if refusal is not None:
                return self._quarantine(route_memory, segment, completed, refusal, dispatched)
            resolution = runtime.reacquire(
                segment.action.target,
                segment.action.action_kind,
                current,
            )
            resolution_refusal = _resolution_refusal(segment, current, resolution)
            if resolution_refusal is not None:
                return self._quarantine(
                    route_memory, segment, completed, resolution_refusal, dispatched
                )
            if resolution is None:
                raise AssertionError("A valid resolution cannot be absent.")
            try:
                authorization = self.action_policy.authorize(
                    resolution.action,
                    source=ActionSource.SAVED_ROUTE,
                    observation_id=current.observation.observation_id,
                    screen_phash=current.observation.phash,
                    screen_size=current.screen_size,
                    context_id=context_id,
                    keyboard_visible=current.observation.keyboard_visible,
                    expected_destination_phash=_expected_destination_phash(segment.destination),
                )
            except (TypeError, ValueError) as exc:
                return self._quarantine(route_memory, segment, completed, str(exc), dispatched)
            if not authorization.control_identity.matches(resolution.live_control_identity):
                return self._quarantine(
                    route_memory,
                    segment,
                    completed,
                    "The authorized action differs from the live resolved control.",
                    dispatched,
                )
            try:
                execution = runtime.execute(authorization, resolution)
            except Exception:  # noqa: BLE001 - runtime is an external boundary
                return self._quarantine(
                    route_memory,
                    segment,
                    completed,
                    "The RouteMemory runtime failed during dispatch.",
                    dispatched,
                )
            dispatched.append(segment.segment_id)
            execution_refusal = _execution_refusal(segment, current, authorization, execution)
            if execution_refusal is not None:
                return self._quarantine(
                    route_memory, segment, completed, execution_refusal, dispatched
                )
            completed.append(segment.segment_id)
            current = execution.destination
        return self._result(
            route_memory,
            RouteMemoryTerminal.COMPLETE,
            completed,
            "Every RouteMemory segment reached its guarded destination.",
            dispatched_segment_ids=dispatched,
        )

    @staticmethod
    def _eligibility_terminal(
        route_memory: RouteMemory,
        compatibility: DiscoveryCompatibilityKey,
        successful_reuse_count: int,
        reuse_policy: RouteMemoryReusePolicy | None,
    ) -> RouteMemoryExecutionResult | None:
        if not verify_artifact(route_memory):
            return RouteMemoryExecutor._result(
                route_memory,
                RouteMemoryTerminal.INELIGIBLE,
                [],
                "RouteMemory integrity verification failed.",
            )
        if route_memory.lifecycle is not RouteMemoryLifecycle.ACTIVE:
            return RouteMemoryExecutor._result(
                route_memory,
                RouteMemoryTerminal.INELIGIBLE,
                [],
                "RouteMemory is not active.",
            )
        if not route_memory.header.compatibility.is_compatible_with(compatibility):
            return RouteMemoryExecutor._result(
                route_memory,
                RouteMemoryTerminal.INELIGIBLE,
                [],
                "RouteMemory is incompatible with the live execution context.",
            )
        if reuse_policy is not None and is_shadow_revalidation_due(
            successful_reuse_count, reuse_policy
        ):
            return RouteMemoryExecutor._result(
                route_memory,
                RouteMemoryTerminal.SHADOW_REVALIDATION_REQUIRED,
                [],
                "RouteMemory reached its periodic shadow-revalidation interval.",
            )
        return None

    @staticmethod
    def _quarantine(
        route_memory: RouteMemory,
        segment: RouteMemorySegment,
        completed: list[str],
        reason: str,
        dispatched: list[str] | None = None,
    ) -> RouteMemoryExecutionResult:
        return RouteMemoryExecutor._result(
            route_memory,
            RouteMemoryTerminal.QUARANTINED,
            completed,
            reason,
            segment.segment_id,
            dispatched_segment_ids=dispatched,
        )

    @staticmethod
    def _result(
        route_memory: RouteMemory,
        terminal: RouteMemoryTerminal,
        completed: list[str],
        reason: str,
        quarantined_segment_id: str | None = None,
        dispatched_segment_ids: list[str] | None = None,
    ) -> RouteMemoryExecutionResult:
        return RouteMemoryExecutionResult(
            route_memory_artifact_id=route_memory.header.artifact_id,
            terminal=terminal,
            completed_segment_ids=tuple(completed),
            dispatched_segment_ids=tuple(dispatched_segment_ids or ()),
            quarantined_segment_id=quarantined_segment_id,
            reason=reason,
        )


def _guard_refusal(
    guard: RouteStateGuard,
    live: RouteMemoryLiveObservation,
) -> str | None:
    for predicate in guard.predicates:
        if guard.identity_scope == "exact_state":
            identity_matches = (
                live.observation.screen_state_id == predicate.screen_state_id
                and is_same_screen(live.observation.phash, predicate.phash)
            )
        else:
            identity_matches = (
                live.screen_family_id is not None
                and live.screen_family_id == predicate.screen_family_id
            )
        if not identity_matches:
            continue
        if (
            predicate.keyboard_visible is not None
            and live.observation.keyboard_visible != predicate.keyboard_visible
        ):
            continue
        if (
            predicate.overlay_visible is not None
            and live.observation.overlay_visible != predicate.overlay_visible
        ):
            continue
        if (
            predicate.foreground_app_id is not None
            and live.observation.foreground_app_id != predicate.foreground_app_id
        ):
            continue
        return None
    return f"The live observation does not satisfy RouteMemory guard {guard.node_id}."


def _resolution_refusal(
    segment: RouteMemorySegment,
    current: RouteMemoryLiveObservation,
    resolution: RouteTargetResolution | None,
) -> str | None:
    if resolution is None:
        return "The RouteMemory target could not be reacquired live."
    target = segment.action.target
    if resolution.source_observation_id != current.observation.observation_id:
        return "Target reacquisition used a stale source observation."
    if resolution.historical_control_identity != target.control_identity:
        return "Target reacquisition resolved another historical control."
    if resolution.matched_role != target.role:
        return "Target reacquisition resolved another control role."
    if resolution.matched_text_digest != target.text_digest:
        return "Target reacquisition resolved another control text digest."
    action_type = resolution.action.type
    action_kind = action_type.value if isinstance(action_type, ActionType) else action_type
    if action_kind != segment.action.action_kind:
        return "Target reacquisition changed the authorized action kind."
    return None


def _execution_refusal(
    segment: RouteMemorySegment,
    source: RouteMemoryLiveObservation,
    authorization: ActionAuthorization,
    execution: RouteMemoryExecutionReceipt,
) -> str | None:
    if not execution.dispatch.success:
        return "The RouteMemory dispatch failed."
    if execution.dispatch.authorization_id != authorization.authorization_id:
        return "The dispatch does not belong to the RouteMemory authorization."
    settlement = execution.settlement
    if settlement.status is not SettlementStatus.DURABLE_PROGRESS:
        return "The RouteMemory action did not make durable progress."
    if not settlement.control_identity.matches(authorization.control_identity):
        return "The settlement belongs to another live control."
    required_observations = {
        source.observation.observation_id,
        execution.destination.observation.observation_id,
    }
    if not required_observations.issubset(set(settlement.observation_ids)):
        return "The settlement lacks exact source or destination observations."
    if not is_same_screen(settlement.destination_phash, execution.destination.observation.phash):
        return "The settlement destination differs from the live observation."
    return _guard_refusal(segment.destination, execution.destination)


def _expected_destination_phash(guard: RouteStateGuard) -> str | None:
    if guard.identity_scope != "exact_state" or len(guard.predicates) != 1:
        return None
    return guard.predicates[0].phash
