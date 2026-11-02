"""Launch-prefix stitching and downstream retained-path rejoin tests."""

from datetime import datetime, timezone

from yewo.agent import (
    AgentResult,
    AgentTerminalReason,
    AgentUsage,
    MemoryCompatibility,
    RouteActionSource,
)
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.app_memory import AppMemory, InputContract, InputValueShape
from yewo.agent.discovery import AppMemoryBuilder, DiscoveryObjective, DiscoveryRequest
from yewo.agent.discovery_checkpoint import build_discovery_checkpoint
from yewo.agent.planner import PlanningHistoryEntry, PlanningRequest, TaskProgressContext
from yewo.agent.replay_rejoin import (
    ReplayRejoinActionSource,
    ReplayRejoinPlan,
    build_replay_rejoin_plans,
    combine_replayed_checkpoint_prefix,
    select_checkpoint_replay_segments,
)
from yewo.agent.planning_evidence import PlannerDecisionSource
from yewo.observations import Observation

from agent.actuation.support import proposal
from agent.discovery.support import agent_result, observation


class RecordingRepairPlanner:
    """Return one repair action and record delegated requests."""

    def __init__(self) -> None:
        self.requests: list[PlanningRequest] = []

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Represent one model-backed repair decision."""
        del request
        return 1

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.requests.append(request)
        return proposal(
            ActionKind.WAIT,
            proposal_id=f"repair_{len(self.requests)}",
            source_observation_id=request.observation.observation_id,
        )


def _request(
    current: Observation,
    history: tuple[PlanningHistoryEntry, ...] = (),
) -> PlanningRequest:
    return PlanningRequest(
        observation=current,
        objective="Resume the retained path.",
        completion_conditions=("The retained frontier is reached.",),
        history=history,
        available_actions=frozenset({ActionKind.TAP, ActionKind.WAIT}),
    )


def _chain_memory() -> tuple[AppMemory, tuple[Observation, ...], AgentResult]:
    observations = tuple(
        observation(
            f"obs_{index}",
            str(index) * 16,
            predecessor_observation_id=f"obs_{index - 1}" if index > 1 else None,
            offset_seconds=index,
        )
        for index in range(1, 6)
    )
    steps = []
    for index, (source, destination) in enumerate(
        zip(observations, observations[1:], strict=False),
        start=1,
    ):
        single = agent_result(
            source,
            destination,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        step = single.steps[0]
        authorization = step.authorization
        receipt = step.receipt
        outcome = step.outcome
        assert authorization is not None
        assert receipt is not None
        assert outcome is not None
        proposal_id = f"proposal_{index}"
        authorization_id = f"authorization_{index}"
        receipt_id = f"receipt_{index}"
        steps.append(
            step.model_copy(
                update={
                    "step_id": f"step_{index}",
                    "step_index": index,
                    "proposal": step.proposal.model_copy(update={"proposal_id": proposal_id}),
                    "authorization": authorization.model_copy(
                        update={
                            "authorization_id": authorization_id,
                            "proposal_id": proposal_id,
                        }
                    ),
                    "receipt": receipt.model_copy(
                        update={
                            "receipt_id": receipt_id,
                            "authorization_id": authorization_id,
                            "proposal_id": proposal_id,
                        }
                    ),
                    "outcome": outcome.model_copy(
                        update={
                            "outcome_id": f"outcome_{index}",
                            "authorization_id": authorization_id,
                            "receipt_id": receipt_id,
                        }
                    ),
                }
            )
        )
    evidence = AgentResult(
        terminal_reason=AgentTerminalReason.BUDGET_EXHAUSTED,
        reason="The controlled chain stopped at its frontier.",
        final_observation_id=observations[-1].observation_id,
        usage=AgentUsage(
            steps=len(steps),
            planning_requests=len(steps),
            grounding_requests=len(steps),
            dispatched_actions=len(steps),
        ),
        steps=tuple(steps),
    )
    request = DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="workflow",
                statement="The workflow frontier is reachable.",
            ),
        ),
    )
    promotion = AppMemoryBuilder().promote(
        request,
        evidence,
        {item.observation_id: item for item in observations},
        MemoryCompatibility(application=observations[0].application),
        verified_objective_ids=frozenset(),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    memory = promotion.memory
    suffix = build_discovery_checkpoint(
        "workflow",
        "The workflow frontier is reachable.",
        list(memory.transitions[2:]),
        "run_1",
        datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert suffix is not None
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "checkpoints": (suffix,),
        }
    )
    return memory, observations, evidence


def _replace_second_transition_source(
    memory: AppMemory,
    *,
    masked_pixels_sha256: str | None = None,
) -> AppMemory:
    source_observation_id = memory.transitions[1].source_observation_id
    source_variant = next(
        variant
        for screen in memory.screens
        for variant in screen.variants
        if variant.observation_id == source_observation_id
    )
    duplicate = source_variant.model_copy(
        update={
            "observation_id": "cross_run_source",
            "source_run_id": "run_2",
            "fingerprint": source_variant.fingerprint.model_copy(
                update={
                    "masked_pixels_sha256": masked_pixels_sha256
                    or source_variant.fingerprint.masked_pixels_sha256,
                }
            ),
        }
    )
    screens = tuple(
        screen.model_copy(update={"variants": (*screen.variants, duplicate)})
        if source_observation_id in {variant.observation_id for variant in screen.variants}
        else screen
        for screen in memory.screens
    )
    transitions = (
        memory.transitions[0],
        memory.transitions[1].model_copy(
            update={"source_observation_id": duplicate.observation_id}
        ),
        *memory.transitions[2:],
    )
    return AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "screens": screens,
            "transitions": transitions,
        }
    )


def test_checkpoint_stitches_a_launch_prefix_before_its_retained_suffix() -> None:
    memory, observations, evidence = _chain_memory()
    checkpoint = memory.checkpoints[0]

    selection = select_checkpoint_replay_segments(memory, checkpoint, observations[0])

    assert tuple(segment.transition_id for segment in selection.replay_segments) == tuple(
        transition.transition_id for transition in memory.transitions
    )
    assert selection.downstream_segments == ()
    identifiers = iter(("replay_1", "replay_2"))
    source = RouteActionSource(
        memory,
        checkpoint,
        identifier_factory=lambda: next(identifiers),
    )
    first = source.plan(_request(observations[0]))
    first_outcome = evidence.steps[0].outcome
    assert first_outcome is not None
    second = source.plan(
        _request(
            observations[1],
            (PlanningHistoryEntry(proposal=first, outcome=first_outcome),),
        )
    )

    assert first.action is ActionKind.TAP
    assert second.action is ActionKind.TAP
    assert source.replay_transition_ids == tuple(
        transition.transition_id for transition in memory.transitions
    )


def test_checkpoint_stitches_identical_states_captured_in_different_runs() -> None:
    memory, observations, unused_evidence = _chain_memory()
    memory = _replace_second_transition_source(memory)

    selection = select_checkpoint_replay_segments(
        memory,
        memory.checkpoints[0],
        observations[0],
    )

    assert tuple(segment.transition_id for segment in selection.replay_segments) == tuple(
        transition.transition_id for transition in memory.transitions
    )
    assert selection.downstream_segments == ()


def test_checkpoint_does_not_guess_an_approach_across_different_exact_pixels() -> None:
    memory, observations, unused_evidence = _chain_memory()
    memory = _replace_second_transition_source(memory, masked_pixels_sha256="f" * 64)

    selection = select_checkpoint_replay_segments(
        memory,
        memory.checkpoints[0],
        observations[0],
    )

    assert selection.replay_segments == ()
    assert selection.downstream_segments


def test_disconnected_checkpoint_keeps_only_its_suffix_without_a_frontier() -> None:
    memory, observations, unused_evidence = _chain_memory()
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "transitions": (
                memory.transitions[0],
                *memory.transitions[2:],
            ),
        }
    )

    selection = select_checkpoint_replay_segments(
        memory,
        memory.checkpoints[0],
        observations[0],
    )

    assert selection.replay_segments == ()
    assert tuple(segment.transition_id for segment in selection.downstream_segments) == (
        memory.transitions[1].transition_id,
        memory.transitions[2].transition_id,
    )


def test_disconnected_checkpoint_does_not_replay_an_arbitrary_longest_approach() -> None:
    memory, observations, unused_evidence = _chain_memory()
    branch = memory.transitions[1].model_copy(
        update={
            "transition_id": "branch_to_screen_5",
            "destination_screen_id": memory.transitions[3].destination_screen_id,
            "destination_observation_id": observations[4].observation_id,
        }
    )
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "transitions": (
                memory.transitions[0],
                branch,
                *memory.transitions[2:],
            ),
        }
    )

    selection = select_checkpoint_replay_segments(
        memory,
        memory.checkpoints[0],
        observations[0],
    )

    assert selection.replay_segments == ()
    assert selection.downstream_segments


def test_reconciled_frontier_selects_its_shortest_guarded_path() -> None:
    memory, observations, unused_evidence = _chain_memory()
    checkpoint = memory.checkpoints[0]

    selection = select_checkpoint_replay_segments(
        memory,
        checkpoint,
        observations[0],
        target_transition_id=memory.transitions[1].transition_id,
    )

    assert tuple(segment.transition_id for segment in selection.replay_segments) == (
        memory.transitions[0].transition_id,
        memory.transitions[1].transition_id,
    )
    assert selection.downstream_segments == ()

    source = RouteActionSource(memory, checkpoint)
    source.plan(
        _request(observations[0]).model_copy(
            update={
                "progress_context": TaskProgressContext(
                    remaining_work=("Reach the retained frontier.",),
                    replay_target_transition_id=memory.transitions[1].transition_id,
                )
            }
        )
    )
    assert source.replay_transition_ids == tuple(
        transition.transition_id for transition in memory.transitions[:2]
    )


def test_repair_rejoins_screen_four_and_replays_the_remaining_fast_path() -> None:
    memory, observations, unused_evidence = _chain_memory()
    repair = RecordingRepairPlanner()
    plan = ReplayRejoinPlan(
        path_id=memory.checkpoints[0].checkpoint_id,
        decision_source=PlannerDecisionSource.DISCOVERY_CHECKPOINT,
        transition_ids=tuple(transition.transition_id for transition in memory.transitions),
        minimum_segment_index=3,
    )
    source = ReplayRejoinActionSource(
        repair,
        memory,
        plan,
        identifier_factory=lambda: "rejoined_action",
    )
    changed_screen_three = observation("changed_3", "a" * 16)

    assert source.required_model_requests(_request(changed_screen_three)) == 1
    repair_action = source.plan(_request(changed_screen_three))
    assert repair_action.action is ActionKind.WAIT
    assert len(repair.requests) == 1

    screen_four_request = _request(observations[3])
    assert source.required_model_requests(screen_four_request) == 0
    replayed = source.plan(screen_four_request)

    assert replayed.action is ActionKind.TAP
    assert replayed.source_observation_id == observations[3].observation_id
    assert source.decision_metadata().source is PlannerDecisionSource.DISCOVERY_CHECKPOINT
    assert source.decision_metadata().reused_flow_id == memory.checkpoints[0].checkpoint_id


def test_repair_does_not_rejoin_beyond_replay_phash_tolerance() -> None:
    memory, observations, unused_evidence = _chain_memory()
    repair = RecordingRepairPlanner()
    source = ReplayRejoinActionSource(
        repair,
        memory,
        ReplayRejoinPlan(
            path_id=memory.checkpoints[0].checkpoint_id,
            decision_source=PlannerDecisionSource.DISCOVERY_CHECKPOINT,
            transition_ids=tuple(item.transition_id for item in memory.transitions),
            minimum_segment_index=3,
        ),
        identifier_factory=lambda: "rejoined_action",
    )
    stored = observations[3]
    similar = stored.model_copy(
        update={
            "observation_id": "similar_screen_four",
            "screen_fingerprint": stored.screen_fingerprint.model_copy(
                update={
                        "value": f"{int(stored.screen_fingerprint.value, 16) ^ 7:016x}",
                    "masked_pixels_sha256": "f" * 64,
                }
            ),
        }
    )

    assert source.required_model_requests(_request(similar)) == 1
    delegated = source.plan(_request(similar))

    assert delegated.action is ActionKind.WAIT
    assert len(repair.requests) == 1


def test_failed_rejoined_action_returns_to_repair_without_automatic_retry() -> None:
    memory, observations, unused_evidence = _chain_memory()
    repair = RecordingRepairPlanner()
    source = ReplayRejoinActionSource(
        repair,
        memory,
        ReplayRejoinPlan(
            path_id=memory.checkpoints[0].checkpoint_id,
            decision_source=PlannerDecisionSource.DISCOVERY_CHECKPOINT,
            transition_ids=tuple(item.transition_id for item in memory.transitions),
            minimum_segment_index=2,
        ),
        identifier_factory=lambda: "rejoined_action",
    )

    replayed = source.plan(_request(observations[2]))
    delegated = source.plan(_request(observations[2]))

    assert replayed.action is ActionKind.TAP
    assert delegated.action is ActionKind.WAIT
    assert len(repair.requests) == 1


def test_declared_value_transition_is_not_added_as_a_hidden_checkpoint_prefix() -> None:
    memory, observations, unused_evidence = _chain_memory()
    declared_action = memory.transitions[0].action.model_copy(
        update={
            "action": ActionKind.TYPE,
            "declared_value_name": "account_password",
            "input_contract": InputContract(value_shape=InputValueShape.TEXT),
        }
    )
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "transitions": (
                memory.transitions[0].model_copy(update={"action": declared_action}),
                *memory.transitions[1:],
            ),
        }
    )

    selection = select_checkpoint_replay_segments(
        memory,
        memory.checkpoints[0],
        observations[0],
        allow_declared_values=False,
    )
    source = RouteActionSource(memory, memory.checkpoints[0])
    terminal = source.plan(_request(observations[0]))

    assert selection.replay_segments == ()
    assert selection.downstream_segments
    assert terminal.action is ActionKind.CANNOT_PROGRESS


def test_divergence_retains_attempted_and_disconnected_downstream_suffixes() -> None:
    memory, unused_observations, unused_evidence = _chain_memory()
    transition_ids = tuple(item.transition_id for item in memory.transitions)

    plans = build_replay_rejoin_plans(
        memory.checkpoints[0].checkpoint_id,
        PlannerDecisionSource.DISCOVERY_CHECKPOINT,
        transition_ids[:3],
        transition_ids[3:],
        2,
        replay_diverged=True,
    )

    assert tuple(plan.transition_ids for plan in plans) == (
        transition_ids[:3],
        transition_ids[3:],
    )
    assert tuple(plan.minimum_segment_index for plan in plans) == (2, 0)


def test_successful_repair_heals_checkpoint_with_the_launch_prefix() -> None:
    memory, unused_observations, unused_evidence = _chain_memory()
    repaired = build_discovery_checkpoint(
        "workflow",
        "The workflow frontier is reachable.",
        list(memory.transitions[1:]),
        "run_2",
        datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert repaired is not None
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "source_run_ids": (*memory.source_run_ids, "run_2"),
            "checkpoints": (repaired,),
        }
    )

    healed = combine_replayed_checkpoint_prefix(
        memory,
        "workflow",
        "The workflow frontier is reachable.",
        (memory.transitions[0].transition_id,),
        "run_2",
        datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert healed.checkpoints[0].start_screen_id == memory.transitions[0].source_screen_id
    assert healed.checkpoints[0].transition_ids == tuple(
        transition.transition_id for transition in memory.transitions
    )
