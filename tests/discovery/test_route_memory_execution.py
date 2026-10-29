"""Guarded RouteMemory execution across isolated scenario contexts."""

import pytest

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.artifacts import seal_artifact, verify_artifact
from yewo.discovery.contracts import DiscoveryObservation, SideEffectClass
from yewo.discovery.knowledge import (
    AppMap,
    AppMapEdge,
    AppMapNode,
    RouteMemoryLifecycle,
    ScreenPredicate,
)
from yewo.discovery.route_memory import (
    RouteMemoryPromotionReceipt,
    RouteMemoryRequest,
    compile_route_memory_candidate,
    promote_route_memory,
)
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionReceipt,
    RouteMemoryExecutor,
    RouteMemoryLiveObservation,
    RouteMemoryReusePolicy,
    RouteMemoryTerminal,
    RouteTargetResolution,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt

SCREEN_SIZE = (400, 800)


def _predicate(name: str, phash: str) -> ScreenPredicate:
    return ScreenPredicate(
        screen_state_id=f"screen-{name}",
        phash=phash,
        keyboard_visible=False,
        overlay_visible=False,
        foreground_app_id="com.example.app",
    )


def _memory(header):
    screens = (("list", "0" * 64), ("drawer", "f" * 64), ("form", "a" * 64))
    nodes = tuple(
        AppMapNode(
            node_id=name,
            predicates=(_predicate(name, phash),),
            observation_ids=(f"historical-{name}",),
        )
        for name, phash in screens
    )
    edges = tuple(
        AppMapEdge(
            transition_identity=f"transition-{number}",
            source_node_id=source,
            destination_node_id=destination,
            action_kind="tap",
            control_identity=f"control-{number}",
            control_role="button",
            control_text_digest=f"digest-{number}",
            side_effect=SideEffectClass.READ_ONLY,
            reversible=True,
            authorization_receipt_id=f"authorization-{number}",
            dispatch_receipt_id=f"dispatch-{number}",
            settlement_receipt_id=f"settlement-{number}",
            evidence_observation_ids=(
                f"historical-{source}",
                f"historical-{destination}",
            ),
        )
        for number, source, destination in (
            (1, "list", "drawer"),
            (2, "drawer", "form"),
        )
    )
    app_map = seal_artifact(
        AppMap(
            header=header("app-map"),
            source_discovery_artifact_ids=("discovery-1",),
            nodes=nodes,
            edges=edges,
        )
    )
    candidate = compile_route_memory_candidate(
        app_map,
        RouteMemoryRequest(
            capability="open_schedule_entry_form",
            semantic_instruction="Open the schedule entry form",
            source_node_id="list",
            destination_node_id="form",
            max_segments=2,
        ),
    )
    return promote_route_memory(
        candidate,
        RouteMemoryPromotionReceipt(
            source_candidate_artifact_id=candidate.header.artifact_id,
            verified_segment_ids=tuple(segment.segment_id for segment in candidate.segments),
            evidence_receipt_ids=tuple(
                receipt_id
                for segment in candidate.segments
                for receipt_id in (
                    segment.authorization_receipt_id,
                    segment.dispatch_receipt_id,
                    segment.settlement_receipt_id,
                )
            ),
        ),
    )


def _live(name: str, phash: str, scenario: str, index: int) -> RouteMemoryLiveObservation:
    return RouteMemoryLiveObservation(
        observation=DiscoveryObservation(
            observation_id=f"{scenario}-observation-{index}",
            screen_state_id=f"screen-{name}",
            phash=phash,
            keyboard_visible=False,
            overlay_visible=False,
            foreground_app_id="com.example.app",
        ),
        screen_size=SCREEN_SIZE,
    )


class _Runtime:
    def __init__(
        self,
        scenario: str,
        *,
        refuse_reacquisition_at: int | None = None,
        settlement_status_at: tuple[int, SettlementStatus] | None = None,
    ):
        self.scenario = scenario
        self.refuse_reacquisition_at = refuse_reacquisition_at
        self.settlement_status_at = settlement_status_at
        self.reacquisitions = 0
        self.executions = 0
        self.current: RouteMemoryLiveObservation | None = None
        self.destinations = (
            _live("drawer", "f" * 64, scenario, 2),
            _live("form", "a" * 64, scenario, 3),
        )

    def reacquire(self, target, action_kind, observation):
        self.reacquisitions += 1
        self.current = observation
        if self.reacquisitions == self.refuse_reacquisition_at:
            return None
        action = TapAction(
            coordinates=Coordinates(x=50 + self.reacquisitions, y=100),
            reason="Live RouteMemory target reacquisition",
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
        source = self.current
        if source is None:
            raise AssertionError("Reacquisition must precede execution.")
        index = self.executions
        self.executions += 1
        destination = self.destinations[index]
        status = SettlementStatus.DURABLE_PROGRESS
        if self.settlement_status_at is not None and self.settlement_status_at[0] == index + 1:
            status = self.settlement_status_at[1]
        return RouteMemoryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=f"{self.scenario}-dispatch-{index + 1}",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id=f"{self.scenario}-settlement-{index + 1}",
                control_identity=authorization.control_identity,
                source_phash=source.observation.phash,
                destination_phash=destination.observation.phash,
                status=status,
                reason="Controlled RouteMemory audit",
                observation_ids=(
                    source.observation.observation_id,
                    destination.observation.observation_id,
                ),
            ),
            destination=destination,
        )


@pytest.mark.parametrize("scenario", ["happy", "negative", "boundary", "empty"])
def test_one_memory_reaches_form_for_four_isolated_scenarios(header, compatibility, scenario):
    memory = _memory(header)
    runtime = _Runtime(scenario)

    result = RouteMemoryExecutor().execute(
        memory,
        compatibility,
        _live("list", "0" * 64, scenario, 1),
        runtime,
        context_id=f"scenario-{scenario}",
    )

    assert result.terminal is RouteMemoryTerminal.COMPLETE
    assert result.completed_segment_ids == tuple(segment.segment_id for segment in memory.segments)
    assert result.dispatched_segment_ids == tuple(segment.segment_id for segment in memory.segments)
    assert runtime.executions == 2
    assert scenario not in memory.model_dump_json()


def test_local_quarantine_preserves_completed_prefix_without_mutating_memory(header, compatibility):
    memory = _memory(header)
    runtime = _Runtime("negative", refuse_reacquisition_at=2)

    result = RouteMemoryExecutor().execute(
        memory,
        compatibility,
        _live("list", "0" * 64, "negative", 1),
        runtime,
        context_id="scenario-negative",
    )

    assert result.terminal is RouteMemoryTerminal.QUARANTINED
    assert result.completed_segment_ids == (memory.segments[0].segment_id,)
    assert result.dispatched_segment_ids == (memory.segments[0].segment_id,)
    assert result.quarantined_segment_id == memory.segments[1].segment_id
    assert memory.lifecycle is RouteMemoryLifecycle.ACTIVE
    assert verify_artifact(memory)


def test_failed_settlement_is_attributed_to_current_segment(header, compatibility):
    memory = _memory(header)
    runtime = _Runtime(
        "boundary",
        settlement_status_at=(2, SettlementStatus.NO_VISIBLE_CHANGE),
    )

    result = RouteMemoryExecutor().execute(
        memory,
        compatibility,
        _live("list", "0" * 64, "boundary", 1),
        runtime,
        context_id="scenario-boundary",
    )

    assert result.terminal is RouteMemoryTerminal.QUARANTINED
    assert result.completed_segment_ids == (memory.segments[0].segment_id,)
    assert result.dispatched_segment_ids == tuple(segment.segment_id for segment in memory.segments)
    assert "durable progress" in result.reason


def test_guard_incompatibility_and_shadow_due_dispatch_nothing(header, compatibility):
    memory = _memory(header)
    runtime = _Runtime("empty")
    executor = RouteMemoryExecutor()

    mismatch = executor.execute(
        memory,
        compatibility,
        _live("other", "1" * 64, "empty", 1),
        runtime,
        context_id="scenario-empty",
    )
    shadow = executor.execute(
        memory,
        compatibility,
        _live("list", "0" * 64, "empty", 1),
        runtime,
        context_id="scenario-empty",
        successful_reuse_count=2,
        reuse_policy=RouteMemoryReusePolicy(shadow_revalidation_interval=3),
    )

    assert mismatch.terminal is RouteMemoryTerminal.QUARANTINED
    assert mismatch.dispatched_segment_ids == ()
    assert shadow.terminal is RouteMemoryTerminal.SHADOW_REVALIDATION_REQUIRED
    assert runtime.executions == 0


def test_incompatible_or_non_active_memory_is_ineligible(header, compatibility):
    memory = _memory(header)
    changed_compatibility = compatibility.model_copy(update={"app_artifact_sha256": "d" * 64})
    inactive = seal_artifact(
        memory.model_copy(
            update={
                "header": memory.header.model_copy(update={"integrity_sha256": None}),
                "lifecycle": RouteMemoryLifecycle.SUSPECT,
            }
        )
    )
    runtime = _Runtime("happy")
    initial = _live("list", "0" * 64, "happy", 1)

    incompatible = RouteMemoryExecutor().execute(
        memory,
        changed_compatibility,
        initial,
        runtime,
        context_id="scenario-happy",
    )
    non_active = RouteMemoryExecutor().execute(
        inactive,
        compatibility,
        initial,
        runtime,
        context_id="scenario-happy",
    )

    assert incompatible.terminal is RouteMemoryTerminal.INELIGIBLE
    assert non_active.terminal is RouteMemoryTerminal.INELIGIBLE
    assert runtime.executions == 0
