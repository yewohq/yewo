"""Perception and device adapter for guarded RouteMemory execution."""

from __future__ import annotations

from dataclasses import dataclass

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.safety import ActionAuthorization, execute_authorized
from yewo.agent.grounding.marks import Mark, project_targets
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.perception import settle
from yewo.agent.perception.elements import element_roles
from yewo.agent.run.settlement import SettlementController, SettlementReceipt
from yewo.agent.step import observation as step_observation
from yewo.discovery.contracts import DiscoveryObservation
from yewo.discovery.control_surface import (
    ControlSurface,
    build_control_surface,
)
from yewo.discovery.knowledge import (
    ControlReacquisition,
    RouteMemory,
    RouteStateGuard,
    ScreenPredicate,
)
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionReceipt,
    RouteMemoryLiveObservation,
    RouteTargetResolution,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt
from yewo.identity import new_uuid
from yewo.testing.warm_execution import (
    ScenarioRouteRuntimeContext,
    ScenarioRouteSelection,
)


def _roles(mark: Mark) -> frozenset[str]:
    return element_roles(mark.element)


def _overlay_visible(marks: tuple[Mark, ...]) -> bool:
    overlay_roles = {"dialog", "modal", "overlay", "alert"}
    return any(_roles(mark).intersection(overlay_roles) for mark in marks)


def _matches_auxiliary_state(
    predicate: ScreenPredicate,
    *,
    keyboard_visible: bool | None,
    overlay_visible: bool,
    foreground_app_id: str | None,
) -> bool:
    return (
        (predicate.keyboard_visible is None or predicate.keyboard_visible == keyboard_visible)
        and (predicate.overlay_visible is None or predicate.overlay_visible == overlay_visible)
        and (
            predicate.foreground_app_id is None or predicate.foreground_app_id == foreground_app_id
        )
    )


@dataclass(frozen=True)
class _CapturedRouteState:
    live: RouteMemoryLiveObservation
    marks: tuple[Mark, ...]
    surface: ControlSurface


@dataclass(frozen=True)
class _RouteRuntimeContext:
    initial_observation: RouteMemoryLiveObservation
    runtime: "_PerceptionRouteRuntime"


class _PerceptionRouteRuntime:
    def __init__(self, route_memory, device_control, perception):
        self.route_memory = route_memory
        self.device_control = device_control
        self.perception = perception
        self.segment_index = 0
        self.current: _CapturedRouteState | None = None

    def capture(self, guard: RouteStateGuard) -> _CapturedRouteState:
        snapshot = step_observation.capture(self.perception, None, None)
        if snapshot.current_phash is None or snapshot.scan_info is None:
            raise RuntimeError("RouteMemory requires a fingerprinted live observation.")
        screen_size = (
            snapshot.scan_info.screen_width,
            snapshot.scan_info.screen_height,
        )
        keyboard_visible = self.device_control.is_keyboard_open()
        projection = project_targets(
            snapshot.gui_elements,
            *screen_size,
            keyboard_visible,
        )
        overlay_visible = _overlay_visible(projection.marks)
        foreground_app_id = self.device_control.app_id
        predicate = next(
            (
                candidate
                for candidate in guard.predicates
                if is_same_screen(candidate.phash, snapshot.current_phash)
                and _matches_auxiliary_state(
                    candidate,
                    keyboard_visible=keyboard_visible,
                    overlay_visible=overlay_visible,
                    foreground_app_id=foreground_app_id,
                )
            ),
            None,
        )
        screen_state_id = (
            predicate.screen_state_id
            if predicate is not None
            else f"unmatched:{snapshot.current_phash}"
        )
        live = RouteMemoryLiveObservation(
            observation=DiscoveryObservation(
                observation_id=snapshot.observation_id,
                screen_state_id=screen_state_id,
                screen_family_candidate_id=(
                    predicate.screen_family_id if predicate is not None else None
                ),
                phash=snapshot.current_phash,
                keyboard_visible=keyboard_visible,
                overlay_visible=overlay_visible,
                foreground_app_id=foreground_app_id,
            ),
            screen_family_id=(predicate.screen_family_id if predicate is not None else None),
            screen_size=screen_size,
        )
        if snapshot.perception_frame is None:
            raise RuntimeError("RouteMemory reacquisition requires a fused perception frame.")
        captured = _CapturedRouteState(
            live=live,
            marks=projection.marks,
            surface=build_control_surface(snapshot.perception_frame, screen_size),
        )
        self.current = captured
        return captured

    def reacquire(
        self,
        target: ControlReacquisition,
        action_kind: str,
        observation: RouteMemoryLiveObservation,
    ) -> RouteTargetResolution | None:
        if action_kind != "tap" or self.current is None:
            return None
        if self.current.live != observation:
            return None
        role = target.role.casefold()
        candidates = tuple(
            control
            for control in self.current.surface.controls
            if role == control.role and target.text_digest == control.text_digest
        )
        if len(candidates) != 1:
            return None
        control = candidates[0]
        action = TapAction(
            coordinates=Coordinates(x=control.center[0], y=control.center[1]),
            reason="Reacquire one verified RouteMemory target.",
        )
        return RouteTargetResolution(
            source_observation_id=observation.observation.observation_id,
            historical_control_identity=target.control_identity,
            matched_role=target.role,
            matched_text_digest=target.text_digest,
            action=action,
            live_control_identity=build_control_identity(
                action,
                observation.observation.phash,
                observation.screen_size,
            ),
        )

    def execute(
        self,
        authorization: ActionAuthorization,
        resolution: RouteTargetResolution,
    ) -> RouteMemoryExecutionReceipt:
        del resolution
        current = self.current
        if current is None:
            raise RuntimeError("RouteMemory execution requires a live source.")
        segment = self.route_memory.segments[self.segment_index]
        settlement = SettlementController()
        settlement.begin_action(
            authorization.control_identity,
            current.live.observation.phash,
            current.live.observation.observation_id,
        )
        execute_authorized(authorization, self.device_control)
        settle.settle(self.device_control, 4.0)
        destination = self.capture(segment.destination)
        receipt = settlement.record_dispatch(
            authorization.control_identity,
            current.live.observation.phash,
            destination.live.observation.phash,
            failed=False,
        )
        if receipt is None:
            settlement.observe(
                destination.live.observation.phash,
                f"{destination.live.observation.observation_id}-stable",
            )
            receipt = settlement.observe(
                destination.live.observation.phash,
                destination.live.observation.observation_id,
            )
        if not isinstance(receipt, SettlementReceipt):
            raise RuntimeError("The RouteMemory action did not settle durably.")
        self.segment_index += 1
        return RouteMemoryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=new_uuid(),
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=receipt,
            destination=destination.live,
        )


class _PerceptionRouteRuntimeFactory:
    def __init__(self, device_control, perception):
        self.device_control = device_control
        self.perception = perception

    def build(
        self,
        route_memory: RouteMemory,
        *,
        context_id: str,
    ) -> ScenarioRouteRuntimeContext:
        del context_id
        runtime = _PerceptionRouteRuntime(
            route_memory,
            self.device_control,
            self.perception,
        )
        initial = runtime.capture(route_memory.start)
        return _RouteRuntimeContext(
            initial_observation=initial.live,
            runtime=runtime,
        )


class PerceptionRouteRuntimeProvider:
    """Create fresh target and settlement state for every scenario."""

    def runtime_factory_for(
        self,
        scenario: ScenarioRouteSelection,
        device_control: object,
        perception: object,
    ) -> _PerceptionRouteRuntimeFactory:
        del scenario
        return _PerceptionRouteRuntimeFactory(device_control, perception)
