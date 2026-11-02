"""YewoAgent verified-route replay tests."""

from datetime import datetime, timezone

from yewo.agent import (
    AgentResult,
    AgentTask,
    AgentTerminalReason,
    AgentUsage,
    MemoryCompatibility,
    RouteActionSource,
    ScreenComparisonPolicy,
    select_verified_route,
)
from yewo.agent.actuation import ActionKind, LiteralText
from yewo.agent.planner import PlanningHistoryEntry, PlanningRequest
from yewo.agent.discovery import (
    AppMemoryBuilder,
    DiscoveryObjective,
    DiscoveryRequest,
)

from agent.discovery.support import agent_result, observation
from agent.test_runtime import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    _mode_policy,
    _observation,
    _runtime,
)
from grounding.support import DeterministicGrounder


def _memory_route_and_evidence(
    *,
    maximum_phash_distance: int = 0,
    maximum_replay_phash_distance: int = 2,
    action: ActionKind = ActionKind.TAP,
):
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(before, after, action=action)
    request = DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="account_form",
                statement="The account form is reachable.",
            ),
        ),
    )
    promotion = AppMemoryBuilder().promote(
        request,
        result,
        {before.observation_id: before, after.observation_id: after},
        MemoryCompatibility(
            application=before.application,
            screen_comparison=ScreenComparisonPolicy(
                maximum_phash_distance=maximum_phash_distance,
                maximum_replay_phash_distance=maximum_replay_phash_distance,
            ),
        ),
        verified_objective_ids=frozenset({"account_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return promotion.memory, promotion.memory.routes[0], before, after, result


def _request(current, history=(), available=frozenset({ActionKind.TAP})):
    return PlanningRequest(
        observation=current,
        objective="Replay the account-form route.",
        completion_conditions=("The route destination guard matches.",),
        history=history,
        available_actions=available,
    )


def test_route_source_emits_semantic_action_then_done_after_destination_guard() -> None:
    memory, route, before, after, evidence = _memory_route_and_evidence()
    identifiers = iter(("proposal_replay", "proposal_done"))
    source = RouteActionSource(memory, route, identifier_factory=lambda: next(identifiers))

    action = source.plan(_request(before))
    action_decision = source.decision_metadata()
    outcome = evidence.steps[0].outcome
    assert outcome is not None
    terminal = source.plan(
        _request(
            after,
            history=(PlanningHistoryEntry(proposal=action, outcome=outcome),),
        )
    )

    assert action.action is ActionKind.TAP
    assert action.target_expression == "the add button"
    assert action.model_dump().get("target") is None
    assert terminal.action is ActionKind.DONE
    assert action_decision.reused_flow_id == route.route_id
    assert action_decision.reused_transition_ids == (route.segments[0].transition_id,)
    assert action_decision.reused_segment_index == 0
    assert source.completed_segment_count == 1
    assert source.failed_segment_index is None


def test_route_source_replays_disposable_text_without_a_declared_value() -> None:
    memory, route, before, _, _ = _memory_route_and_evidence(action=ActionKind.TYPE)
    source = RouteActionSource(memory, route, identifier_factory=lambda: "proposal_replay")

    action = source.plan(_request(before, available=frozenset({ActionKind.TYPE})))

    assert action.action is ActionKind.TYPE
    assert action.text == LiteralText(value="hello")


def test_route_source_collapses_learned_focus_and_type_into_targeted_type() -> None:
    first = observation("obs_1", "0000000000000000")
    focused = observation(
        "obs_2",
        "0101010101010101",
        predecessor_observation_id=first.observation_id,
        offset_seconds=1,
    )
    typed = observation(
        "obs_3",
        "1111111111111111",
        predecessor_observation_id=focused.observation_id,
        offset_seconds=2,
    )
    focus_result = agent_result(first, focused, action=ActionKind.FOCUS)
    type_result = agent_result(focused, typed, action=ActionKind.TYPE)
    type_step = type_result.steps[0]
    type_outcome = type_step.outcome
    assert type_outcome is not None
    type_step = type_step.model_copy(
        update={
            "step_id": "type_step",
            "step_index": 2,
            "outcome": type_outcome.model_copy(update={"outcome_id": "type_outcome"}),
        }
    )
    terminal_step = type_result.steps[1].model_copy(
        update={"step_id": "terminal_step", "step_index": 3}
    )
    evidence = AgentResult(
        terminal_reason=AgentTerminalReason.DONE_REQUESTED,
        reason="The planner requested completion.",
        final_observation_id=typed.observation_id,
        usage=AgentUsage(steps=3, planning_requests=3, grounding_requests=1, dispatched_actions=2),
        steps=(focus_result.steps[0], type_step, terminal_step),
    )
    request = DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="account_form",
                statement="The account form is reachable.",
            ),
        ),
    )
    promotion = AppMemoryBuilder().promote(
        request,
        evidence,
        {item.observation_id: item for item in (first, focused, typed)},
        MemoryCompatibility(application=first.application),
        verified_objective_ids=frozenset({"account_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    route = promotion.memory.routes[0]
    divergent = RouteActionSource(
        promotion.memory,
        route,
        known_divergent_transition_id=route.segments[0].transition_id,
    )
    blocked = divergent.plan(_request(first, available=frozenset({ActionKind.TYPE})))
    identifiers = iter(("proposal_replay", "proposal_done"))
    source = RouteActionSource(
        promotion.memory,
        route,
        identifier_factory=lambda: next(identifiers),
    )

    action = source.plan(_request(first, available=frozenset({ActionKind.TYPE})))
    replay_outcome = type_outcome.model_copy(update={"source_observation_id": first.observation_id})
    terminal = source.plan(
        _request(
            typed,
            history=(PlanningHistoryEntry(proposal=action, outcome=replay_outcome),),
            available=frozenset({ActionKind.TYPE}),
        )
    )

    assert action.action is ActionKind.TYPE
    assert action.target_expression == "the name field"
    assert action.text == LiteralText(value="hello")
    assert terminal.action is ActionKind.DONE
    assert source.completed_segment_count == 2
    assert blocked.action is ActionKind.CANNOT_PROGRESS
    assert divergent.rejoin_segment_index == 1


def test_route_selection_requires_an_exact_capability_backed_objective() -> None:
    memory, route, _, _, _ = _memory_route_and_evidence()

    assert select_verified_route(memory, "The account form is reachable.") == route
    assert select_verified_route(memory, "The settings screen is reachable.") is None


def test_route_source_stops_before_action_when_source_guard_diverges() -> None:
    memory, route, _, _, _ = _memory_route_and_evidence()
    changed = observation("obs_changed", "2222222222222222")
    source = RouteActionSource(memory, route, identifier_factory=lambda: "terminal_1")

    terminal = source.plan(_request(changed))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_segment_count == 0
    assert source.failed_segment_index == 0


def test_known_repeated_divergence_stops_before_dispatching_that_transition() -> None:
    memory, route, before, _, _ = _memory_route_and_evidence()
    transition_id = route.segments[0].transition_id
    source = RouteActionSource(
        memory,
        route,
        known_divergent_transition_id=transition_id,
        identifier_factory=lambda: "terminal_1",
    )

    terminal = source.plan(_request(before))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_segment_count == 0
    assert source.failed_segment_index == 0
    assert source.rejoin_segment_index == 1
    assert source.replay_transition_ids[0] == transition_id


def test_route_source_stops_when_destination_guard_diverges() -> None:
    memory, route, before, _, evidence = _memory_route_and_evidence()
    identifiers = iter(("proposal_replay", "proposal_terminal"))
    source = RouteActionSource(memory, route, identifier_factory=lambda: next(identifiers))
    action = source.plan(_request(before))
    changed = observation(
        "obs_changed",
        "2222222222222222",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    outcome = evidence.steps[0].outcome
    assert outcome is not None

    terminal = source.plan(
        _request(
            changed,
            history=(PlanningHistoryEntry(proposal=action, outcome=outcome),),
        )
    )

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.completed_segment_count == 0
    assert source.failed_segment_index == 0


def test_route_guard_tolerates_two_transient_phash_bits() -> None:
    memory, route, _, _, unused_evidence = _memory_route_and_evidence()
    current_source = observation("source_variant", "0000000000000003")
    source = RouteActionSource(memory, route, identifier_factory=lambda: "proposal_replay")

    action = source.plan(_request(current_source))

    assert action.action is ActionKind.TAP
    assert source.failed_segment_index is None


def test_route_guard_rejects_phash_difference_beyond_replay_policy() -> None:
    memory, route, _, _, unused_evidence = _memory_route_and_evidence()
    current_source = observation("source_variant", "0000000000000007")
    source = RouteActionSource(memory, route, identifier_factory=lambda: "proposal_terminal")

    terminal = source.plan(_request(current_source))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.failed_segment_index == 0


def test_route_source_stops_when_active_policy_excludes_route_action() -> None:
    memory, route, before, _, _ = _memory_route_and_evidence()
    source = RouteActionSource(memory, route, identifier_factory=lambda: "terminal_1")

    terminal = source.plan(_request(before, available=frozenset({ActionKind.WAIT})))

    assert terminal.action is ActionKind.CANNOT_PROGRESS
    assert source.failed_segment_index == 0


def test_replay_grounds_the_stored_expression_on_current_pixels(
    tmp_path,
) -> None:
    memory, route, _, _, _ = _memory_route_and_evidence()
    current_source = _observation("current_source", "0000000000000000")
    current_destination = _observation(
        "current_destination",
        "1111111111111111",
        offset_seconds=1,
    )
    stable_destination = _observation(
        "stable_destination",
        "1111111111111111",
        offset_seconds=2,
    )
    observations = ObservationSequence([current_source, current_destination, stable_destination])
    device = RuntimeDevice()
    source = RouteActionSource(memory, route)

    result = _runtime(
        tmp_path,
        source,
        observations,
        device,
        grounder=DeterministicGrounder(150, 200),
    ).run(
        AgentTask(
            objective="Replay the account-form route.",
            completion_conditions=("The route destination guard matches.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert device.calls[0][0] == "tap"
    assert device.calls[0][1][:2] == (150, 200)
    assert result.steps[0].target_grounding is not None
    assert result.steps[0].target_grounding.request.screenshot.artifact_id == (
        current_source.redacted_screenshot.artifact_id
    )
