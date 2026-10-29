"""Warm scenarios reuse mechanics without sharing scenario evidence."""

from dataclasses import dataclass

import pytest

from PIL import Image

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.discovery.artifacts import serialize_artifact
from yewo.discovery.contracts import DiscoveryObservation
from yewo.discovery.knowledge import (
    ScreenPredicate,
)
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionReceipt,
    RouteMemoryLiveObservation,
    RouteTargetResolution,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt
from yewo.testing.warm_execution import (
    LocalRediscoveryBudget,
    LocalRediscoveryResult,
    LocalRediscoveryTerminal,
    QuarantinedMapProposal,
    WarmExecutionTerminal,
    WarmScenarioCoordinator,
)
from yewo.testing.live_route_runtime import PerceptionRouteRuntimeProvider
from knowledge_fakes import build_verified_knowledge
from perception_fakes import frame

SCREEN_SIZE = (400, 800)


@dataclass
class _Scenario:
    scenario_uuid: str
    route_memory_ids: list[str]


def _live(predicate: ScreenPredicate, scenario_id: str, sequence: int):
    return RouteMemoryLiveObservation(
        observation=DiscoveryObservation(
            observation_id=f"{scenario_id}-observation-{sequence}",
            screen_state_id=predicate.screen_state_id,
            phash=predicate.phash,
            keyboard_visible=predicate.keyboard_visible,
            overlay_visible=predicate.overlay_visible,
            foreground_app_id=predicate.foreground_app_id,
        ),
        screen_size=SCREEN_SIZE,
    )


class _Runtime:
    def __init__(self, memory, scenario_id: str, *, reacquire: bool):
        self.memory = memory
        self.scenario_id = scenario_id
        self.can_reacquire = reacquire
        self.source = None
        self.executions = 0

    def reacquire(self, target, action_kind, observation):
        self.source = observation
        if not self.can_reacquire or action_kind != "tap":
            return None
        action = TapAction(
            coordinates=Coordinates(x=100, y=200),
            reason="Fresh scenario-local target",
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

    def execute(self, authorization: ActionAuthorization, resolution):
        del resolution
        self.executions += 1
        destination = _live(
            self.memory.segments[0].destination.predicates[0],
            self.scenario_id,
            2,
        )
        return RouteMemoryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=f"{self.scenario_id}-dispatch",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id=f"{self.scenario_id}-settlement",
                control_identity=authorization.control_identity,
                source_phash=self.source.observation.phash,
                destination_phash=destination.observation.phash,
                status=SettlementStatus.DURABLE_PROGRESS,
                reason="The controlled destination settled.",
                observation_ids=(
                    self.source.observation.observation_id,
                    destination.observation.observation_id,
                ),
            ),
            destination=destination,
        )


@dataclass
class _Context:
    initial_observation: RouteMemoryLiveObservation
    runtime: _Runtime


class _RuntimeFactory:
    def __init__(self, *, reacquire: bool = True):
        self.reacquire = reacquire
        self.context_ids = []
        self.runtimes = []

    def build(self, route_memory, *, context_id):
        scenario_id = context_id.split(":")[1]
        runtime = _Runtime(route_memory, scenario_id, reacquire=self.reacquire)
        self.context_ids.append(context_id)
        self.runtimes.append(runtime)
        return _Context(
            initial_observation=_live(
                route_memory.start.predicates[0],
                scenario_id,
                1,
            ),
            runtime=runtime,
        )


class _Rediscovery:
    def __init__(self, result: LocalRediscoveryResult):
        self.result = result
        self.requests = []

    def recover(self, request):
        self.requests.append(request)
        return self.result


def _rediscovery_result(**changes):
    values = {
        "terminal": LocalRediscoveryTerminal.COMPLETE,
        "reason": "A bounded alternate transition reached the form.",
        "observations": 2,
        "actions": 1,
        "model_calls": 1,
        "duration_ms": 100,
        "cost_micro_usd": 100,
        "proposed_updates": (
            QuarantinedMapProposal(
                proposal_id="proposal-1",
                source_observation_id="source-live",
                destination_observation_id="destination-live",
                action_kind="tap",
            ),
        ),
    }
    values.update(changes)
    return LocalRediscoveryResult(**values)


def test_second_scenario_reuses_route_with_fresh_context_and_no_shared_state(
    header,
    compatibility,
):
    app_map, memory = build_verified_knowledge(header)
    original_map = serialize_artifact(app_map)
    original_memory = serialize_artifact(memory)
    factory = _RuntimeFactory()
    coordinator = WarmScenarioCoordinator(app_map, (memory,))

    first = coordinator.prepare(
        _Scenario("scenario-1", [memory.header.artifact_id]),
        compatibility,
        factory,
    )
    second = coordinator.prepare(
        _Scenario("scenario-2", [memory.header.artifact_id]),
        compatibility,
        factory,
    )

    assert first.terminal is WarmExecutionTerminal.READY
    assert second.terminal is WarmExecutionTerminal.READY
    assert second.local_rediscovery is None
    assert factory.context_ids[0] != factory.context_ids[1]
    assert factory.runtimes[0] is not factory.runtimes[1]
    assert all(runtime.executions == 1 for runtime in factory.runtimes)
    assert serialize_artifact(app_map) == original_map
    assert serialize_artifact(memory) == original_memory


def test_failed_live_predicate_stops_replay_and_uses_one_bounded_local_recovery(
    header,
    compatibility,
):
    app_map, memory = build_verified_knowledge(header)
    factory = _RuntimeFactory(reacquire=False)
    rediscovery = _Rediscovery(_rediscovery_result())

    result = WarmScenarioCoordinator(app_map, (memory,)).prepare(
        _Scenario("scenario-2", [memory.header.artifact_id]),
        compatibility,
        factory,
        local_rediscovery=rediscovery,
    )

    assert result.terminal is WarmExecutionTerminal.READY
    assert factory.runtimes[0].executions == 0
    assert len(rediscovery.requests) == 1
    request = rediscovery.requests[0]
    assert request.failed_segment_id == "segment-open-form"
    assert request.semantic_instruction == "Open the form"
    assert request.budget == LocalRediscoveryBudget()
    assert result.quarantined_proposals[0].proposal_id == "proposal-1"


def test_local_recovery_cannot_exceed_its_budget(header, compatibility):
    app_map, memory = build_verified_knowledge(header)
    rediscovery = _Rediscovery(_rediscovery_result(actions=7))

    result = WarmScenarioCoordinator(app_map, (memory,)).prepare(
        _Scenario("scenario-2", [memory.header.artifact_id]),
        compatibility,
        _RuntimeFactory(reacquire=False),
        local_rediscovery=rediscovery,
    )

    assert result.terminal is WarmExecutionTerminal.UNVERIFIED
    assert result.local_rediscovery.terminal is LocalRediscoveryTerminal.BUDGET_EXHAUSTED
    assert "authorized budget" in result.reason


def test_unknown_route_is_unverified_without_touching_the_runtime(header, compatibility):
    app_map, memory = build_verified_knowledge(header)
    factory = _RuntimeFactory()

    result = WarmScenarioCoordinator(app_map, (memory,)).prepare(
        _Scenario("scenario-2", ["unknown-memory"]),
        compatibility,
        factory,
    )

    assert result.terminal is WarmExecutionTerminal.UNVERIFIED
    assert factory.runtimes == []


def test_failed_route_without_local_boundary_does_not_restart_discovery(
    header,
    compatibility,
):
    app_map, memory = build_verified_knowledge(header)

    result = WarmScenarioCoordinator(app_map, (memory,)).prepare(
        _Scenario("scenario-2", [memory.header.artifact_id]),
        compatibility,
        _RuntimeFactory(reacquire=False),
    )

    assert result.terminal is WarmExecutionTerminal.UNVERIFIED
    assert result.local_rediscovery.terminal is LocalRediscoveryTerminal.NOT_FOUND
    assert "No local rediscovery boundary" in result.reason


class _Device:
    app_id = "com.example.app"

    @staticmethod
    def is_keyboard_open():
        return False


class _Perception:
    """One screen, stated once.

    Reacquisition reads the fused frame, so the elements here are the only
    account of what is on screen.
    """

    current_hash = "0" * 64
    last_scan_info = ScanInfo(
        mode="full",
        reason="first_pass",
        phash="0" * 64,
        screen_width=400,
        screen_height=800,
    )

    def __init__(self, elements):
        self.elements = elements
        self.current_frame = frame(elements=tuple(elements))

    def perceive(self):
        image = Image.new("RGB", (400, 800), "white")
        return self.elements, image, image


def _button(y: int = 100):
    return GUIElement(
        element="button",
        type="button",
        text="Open form",
        confidence=0.99,
        semantic_roles=frozenset({"button"}),
        bounding_box=BoundingBox(x_min=20, y_min=y, x_max=180, y_max=y + 60),
    )


def test_live_provider_reacquires_one_target_without_historical_coordinates(header):
    _, memory = build_verified_knowledge(header)
    factory = PerceptionRouteRuntimeProvider().runtime_factory_for(
        _Scenario("scenario", [memory.header.artifact_id]),
        _Device(),
        _Perception([_button()]),
    )
    context = factory.build(memory, context_id="scenario:scenario:route:memory")

    resolution = context.runtime.reacquire(
        memory.segments[0].action.target,
        "tap",
        context.initial_observation,
    )

    assert resolution is not None
    assert resolution.action.coordinates.x == 100
    assert resolution.action.coordinates.y == 130
    assert resolution.historical_control_identity == "historical-control"


def test_live_provider_refuses_ambiguous_semantic_targets(header):
    _, memory = build_verified_knowledge(header)
    factory = PerceptionRouteRuntimeProvider().runtime_factory_for(
        _Scenario("scenario", [memory.header.artifact_id]),
        _Device(),
        _Perception([_button(100), _button(300)]),
    )
    context = factory.build(memory, context_id="scenario:scenario:route:memory")

    assert (
        context.runtime.reacquire(
            memory.segments[0].action.target,
            "tap",
            context.initial_observation,
        )
        is None
    )


def test_reacquisition_refuses_a_reading_that_carries_no_fused_frame(header):
    """Without a frame there is no surface, and a guess is not a reacquisition."""
    _, memory = build_verified_knowledge(header)
    perception = _Perception([_button()])
    perception.current_frame = None
    factory = PerceptionRouteRuntimeProvider().runtime_factory_for(
        _Scenario("scenario", [memory.header.artifact_id]),
        _Device(),
        perception,
    )

    with pytest.raises(RuntimeError, match="fused perception frame"):
        factory.build(memory, context_id="scenario:scenario:route:memory")
