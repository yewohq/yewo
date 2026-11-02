"""Incomplete discovery checkpoint tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent import (
    AgentTerminalReason,
    CapabilityEvidence,
    MemoryCompatibility,
    RouteActionSource,
    RouteSegment,
    VerifiedRoute,
)
from yewo.agent.actuation import ActionKind, SideEffectClass
from yewo.agent.discovery import AppMemoryBuilder, DiscoveryObjective, DiscoveryRequest
from yewo.agent.discovery_checkpoint import (
    DiscoveryCheckpoint,
    build_discovery_checkpoint,
    retain_discovery_graph,
)
from yewo.agent.planner import PlanningRequest
from yewo.run_narration import RunNarrator, narration_scope

from agent.discovery.support import agent_result, observation


def _incomplete_memory():
    source = observation("obs_1", "0000000000000000")
    destination = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=source.observation_id,
        offset_seconds=1,
    )
    promotion = AppMemoryBuilder().promote(
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="account_form",
                    statement="The account form is reachable.",
                ),
            ),
        ),
        agent_result(source, destination, terminal=AgentTerminalReason.BUDGET_EXHAUSTED),
        {source.observation_id: source, destination.observation_id: destination},
        MemoryCompatibility(application=source.application),
        verified_objective_ids=frozenset(),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return promotion.memory, source


def test_checkpoint_actions_are_reused_without_claiming_a_verified_route() -> None:
    memory, source_observation = _incomplete_memory()
    checkpoint = memory.checkpoints[0]
    assert checkpoint.transition_ids == (memory.transitions[0].transition_id,)
    action_source = RouteActionSource(
        memory,
        checkpoint,
        identifier_factory=lambda: "proposal_checkpoint",
    )

    proposal = action_source.plan(
        PlanningRequest(
            observation=source_observation,
            objective="Resume discovery.",
            completion_conditions=("The checkpoint endpoint is reached.",),
            available_actions=frozenset({ActionKind.TAP}),
        )
    )

    assert proposal.action is ActionKind.TAP
    assert proposal.target_expression == "the add button"
    assert memory.routes == ()
    assert memory.capabilities == ()


def test_checkpoint_removes_a_reversible_navigation_cycle() -> None:
    memory, _ = _incomplete_memory()
    first = memory.transitions[0]
    second = first.model_copy(
        update={
            "transition_id": "transition_back",
            "source_screen_id": first.destination_screen_id,
            "destination_screen_id": first.source_screen_id,
            "side_effect": SideEffectClass.NAVIGATION,
        }
    )
    frontier = first.model_copy(
        update={
            "transition_id": "transition_frontier",
            "destination_screen_id": "screen_frontier",
        }
    )

    checkpoint = build_discovery_checkpoint(
        "account_form",
        "The account form is reachable.",
        [first, second, frontier],
        "run_1",
        datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert checkpoint is not None
    assert checkpoint.transition_ids == ("transition_frontier",)


def test_checkpoint_keeps_the_contiguous_suffix_after_an_evidence_gap() -> None:
    memory, _ = _incomplete_memory()
    first = memory.transitions[0]
    resumed = first.model_copy(
        update={
            "transition_id": "transition_resumed",
            "source_screen_id": "screen_resumed",
            "destination_screen_id": "screen_middle",
        }
    )
    frontier = first.model_copy(
        update={
            "transition_id": "transition_frontier",
            "source_screen_id": "screen_middle",
            "destination_screen_id": "screen_frontier",
        }
    )

    checkpoint = build_discovery_checkpoint(
        "account_form",
        "The account form is reachable.",
        [first, resumed, frontier],
        "run_1",
        datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert checkpoint is not None
    assert checkpoint.transition_ids == (
        "transition_resumed",
        "transition_frontier",
    )


def test_graph_retention_preserves_transition_chronology() -> None:
    memory, _ = _incomplete_memory()
    first = memory.transitions[0]
    transitions = [
        first.model_copy(
            update={
                "transition_id": f"t{index}",
                "source_screen_id": f"screen_{index}",
                "destination_screen_id": f"screen_{index + 1}",
            }
        )
        for index in range(1, 6)
    ]
    checkpoint = DiscoveryCheckpoint(
        checkpoint_id="checkpoint_new",
        objective_id="account_form",
        objective="The account form is reachable.",
        start_screen_id="screen_4",
        destination_screen_id="screen_6",
        transition_ids=("t4", "t5"),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    retained = retain_discovery_graph(
        transitions,
        [],
        [checkpoint],
        [],
        maximum=4,
    )

    assert tuple(item.transition_id for item in retained.transitions) == (
        "t2",
        "t3",
        "t4",
        "t5",
    )


def test_graph_retention_evicts_the_oldest_checkpoint_before_required_evidence(
    tmp_path: Path,
) -> None:
    memory, _ = _incomplete_memory()
    first = memory.transitions[0]
    transitions = [
        first.model_copy(
            update={
                "transition_id": f"t{index}",
                "source_screen_id": f"screen_{index}",
                "destination_screen_id": f"screen_{index + 1}",
            }
        )
        for index in range(1, 5)
    ]
    checkpoints = [
        DiscoveryCheckpoint(
            checkpoint_id=f"checkpoint_{name}",
            objective_id=f"objective_{name}",
            objective=f"Objective {name}",
            start_screen_id=f"screen_{start}",
            destination_screen_id=f"screen_{end + 1}",
            transition_ids=tuple(f"t{index}" for index in range(start, end + 1)),
            source_run_id="run_1",
            updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        )
        for name, start, end in (("old", 1, 2), ("new", 3, 4))
    ]

    with narration_scope(RunNarrator(tmp_path)):
        retained = retain_discovery_graph(
            transitions,
            [],
            checkpoints,
            [],
            maximum=2,
        )

    assert tuple(item.checkpoint_id for item in retained.checkpoints) == (
        "checkpoint_new",
    )
    assert tuple(item.transition_id for item in retained.transitions) == ("t3", "t4")
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Memory retention removed 1 older checkpoint record(s)." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "retention=evicted kind=checkpoint count=1 ids=checkpoint_old" in trace


def test_graph_retention_bounds_routes_and_capabilities() -> None:
    memory, _ = _incomplete_memory()
    first = memory.transitions[0]
    transitions = [
        first.model_copy(
            update={
                "transition_id": f"t{index}",
                "source_screen_id": f"screen_{index}",
                "destination_screen_id": f"screen_{index + 1}",
            }
        )
        for index in range(1, 4)
    ]
    routes = [
        VerifiedRoute(
            route_id=f"route_{index}",
            objective_id=f"objective_{index}",
            objective=f"Objective {index}",
            start_screen_id=transition.source_screen_id,
            destination_screen_id=transition.destination_screen_id,
            segments=(
                RouteSegment(
                    transition_id=transition.transition_id,
                    source_screen_id=transition.source_screen_id,
                    destination_screen_id=transition.destination_screen_id,
                    action=transition.action,
                ),
            ),
            evidence_observation_ids=(
                transition.source_observation_id,
                transition.destination_observation_id,
            ),
        )
        for index, transition in enumerate(transitions, start=1)
    ]
    capabilities = [
        CapabilityEvidence(
            capability_id=f"capability_{index}",
            objective_id=route.objective_id,
            statement=f"Capability {index}",
            route_id=route.route_id,
            evidence_observation_ids=route.evidence_observation_ids,
        )
        for index, route in enumerate(routes, start=1)
    ]

    retained = retain_discovery_graph(
        transitions,
        routes,
        [],
        capabilities,
        maximum=2,
    )

    assert tuple(route.route_id for route in retained.routes) == ("route_2", "route_3")
    assert tuple(item.capability_id for item in retained.capabilities) == (
        "capability_2",
        "capability_3",
    )
