"""Verified transition evidence can revisit frontier work without model guessing."""

from yewo.agent.memory.replay_store import RelativeBounds
from yewo.agent.actuation.safety import ActionSafetyPolicy
from yewo.discovery.contracts import (
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTransitionEvidence,
    FrontierState,
    JourneyGoal,
    ObservedControl,
    RelevantFrontierEntry,
    SideEffectClass,
)
from yewo.discovery.frontier_navigation import (
    FrontierReachability,
    select_frontier_replay_candidate,
    select_frontier_replay_step,
    stranded_frontier_ids,
)
from yewo.discovery.runtime import DiscoveryObservationBatch
from discovery_runtime_fakes import _candidate


def _navigation_run(header, budgets):
    goal = JourneyGoal(goal_id="goal", statement="reach the editor").activate()
    root_control = ObservedControl(
        control_identity="open-menu",
        role="button",
        bounds=RelativeBounds(left=0, top=0, right=100, bottom=100),
        actionable=True,
    )
    target_control = ObservedControl(
        control_identity="edit",
        role="text",
        bounds=RelativeBounds(left=0, top=100, right=200, bottom=200),
        actionable=True,
    )
    root = DiscoveryObservation(
        observation_id="root-observation",
        screen_state_id="root-state",
        phash="0" * 16,
        controls=(root_control,),
    )
    target = DiscoveryObservation(
        observation_id="target-observation",
        screen_state_id="target-state",
        phash="f" * 16,
        controls=(target_control,),
    )
    transition = DiscoveryTransitionEvidence(
        transition_identity="open-menu-transition",
        source_observation_id=root.observation_id,
        destination_observation_id=target.observation_id,
        control_identity=root_control.control_identity,
        action_kind="tap",
        authorization_receipt_id="authorization",
        dispatch_receipt_id="dispatch",
        settlement_receipt_id="settlement",
        side_effect=SideEffectClass.REVERSIBLE,
        reversible=True,
    )
    return DiscoveryRun(
        header=header("discovery-run"),
        goals=(goal,),
        observations=(root, target),
        transitions=(transition,),
        frontier=(
            RelevantFrontierEntry(
                frontier_id="root-frontier",
                source_observation_id=root.observation_id,
                control_identity=root_control.control_identity,
                relevant_goal_ids=(goal.goal_id,),
                state=FrontierState.VERIFIED,
                transition_identity=transition.transition_identity,
            ),
            RelevantFrontierEntry(
                frontier_id="target-frontier",
                source_observation_id=target.observation_id,
                control_identity=target_control.control_identity,
                relevant_goal_ids=(goal.goal_id,),
            ),
        ),
        budgets=budgets,
    )


def test_the_first_verified_step_toward_available_work_is_selected(header, budgets):
    step = select_frontier_replay_step(
        _navigation_run(header, budgets),
        "root-state",
    )

    assert step is not None
    assert step.control_identity == "open-menu"
    assert step.destination_screen_state_id == "target-state"
    assert step.expected_destination_phash == "f" * 16


def test_an_offer_this_frame_cannot_present_is_stranded_on_its_own_screen(
    header, budgets
):
    """Standing on the screen that owns an offer the frame does not carry.

    Left open, that one offer makes the screen read as work in hand, so the
    run never looks for a route to the screens it could still reach.
    """
    run = _navigation_run(header, budgets)
    target = run.observations[1]
    batch = DiscoveryObservationBatch(observation=target, candidates=())

    assert stranded_frontier_ids(run, batch) == {"target-frontier"}


def test_an_offer_this_frame_does_present_is_not_stranded(header, budgets):
    run = _navigation_run(header, budgets)
    target = run.observations[1]
    offered = _candidate(
        "goal",
        target.observation_id,
        source_state=target.screen_state_id,
    )
    run = run.model_copy(
        update={
            "frontier": (
                run.frontier[0],
                run.frontier[1].model_copy(
                    update={"control_identity": offered.control_identity}
                ),
            )
        }
    )
    batch = DiscoveryObservationBatch(observation=target, candidates=(offered,))

    assert stranded_frontier_ids(run, batch) == set()


def test_work_in_hand_is_not_reported_as_work_out_of_reach(header, budgets):
    """Two answers, because the caller's next move differs between them.

    Standing on the screen that owns the work means no travel is needed.
    Reading that as an absent route writes off every other screen with it.
    """
    assert (
        select_frontier_replay_step(
            _navigation_run(header, budgets),
            "target-state",
        )
        is FrontierReachability.WORK_IS_HERE
    )


def test_unverified_transitions_are_never_replayed(header, budgets):
    run = _navigation_run(header, budgets)
    run = run.model_copy(
        update={
            "frontier": (
                run.frontier[0].model_copy(update={"state": FrontierState.ATTEMPTED}),
                run.frontier[1],
            )
        }
    )

    assert (
        select_frontier_replay_step(run, "root-state")
        is FrontierReachability.NO_ROUTE
    )


def test_replay_reauthorizes_the_current_detection_before_returning_it(
    header, budgets
):
    run = _navigation_run(header, budgets)
    candidate = _candidate(
        "goal",
        "root-observation",
        source_state="root-state",
        x=50,
        y=50,
    )
    root = run.observations[0].model_copy(
        update={
            "controls": (
                run.observations[0].controls[0].model_copy(
                    update={"control_identity": candidate.control_identity}
                ),
            )
        }
    )
    transition = run.transitions[0].model_copy(
        update={"control_identity": candidate.control_identity}
    )
    frontier = run.frontier[0].model_copy(
        update={"control_identity": candidate.control_identity}
    )
    run = run.model_copy(
        update={
            "observations": (root, run.observations[1]),
            "transitions": (transition,),
            "frontier": (frontier, run.frontier[1]),
        }
    )
    batch = DiscoveryObservationBatch(observation=root, candidates=(candidate,))

    selection = select_frontier_replay_candidate(
        run,
        batch,
        ActionSafetyPolicy(),
        frozenset({SideEffectClass.READ_ONLY, SideEffectClass.REVERSIBLE}),
    )

    assert selection is not None
    assert selection.candidate.expected_destination_phash == "f" * 16
    assert selection.replay_destination_state_id == "target-state"
