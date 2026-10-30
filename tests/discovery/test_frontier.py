"""What a runtime may offer, checked against the observation that produced it.

A runtime offering a control the observation did not report is refused here,
so no later step acts on a target the run has no evidence for.
"""

import pytest

from yewo.agent.actuation.action_models import Coordinates, PlannerTapAction
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryObservation,
    JourneyGoal,
    ObservedControl,
    SideEffectClass,
)
from yewo.discovery.frontier import extend_frontier
from yewo.discovery.runtime import DiscoveryActionCandidate

GOAL = JourneyGoal(goal_id="goal-1", statement="A user places an order.")
BOUNDS = RelativeBounds(left=10, top=10, right=200, bottom=100)


def _observation(controls=()):
    return DiscoveryObservation(
        observation_id="observation-1",
        screen_state_id="state-1",
        phash="0" * 16,
        controls=controls,
    )


def _control(identity="control-1", actionable=True):
    return ObservedControl(
        control_identity=identity,
        role="view",
        bounds=BOUNDS,
        actionable=actionable,
    )


def _candidate(
    identity="control-1",
    observation_id="observation-1",
    state_id="state-1",
    goal_ids=("goal-1",),
):
    return DiscoveryActionCandidate(
        source_observation_id=observation_id,
        source_screen_state_id=state_id,
        control_identity=identity,
        relevant_goal_ids=goal_ids,
        action=PlannerTapAction(
            coordinates=Coordinates(x=100, y=50), reason="explore"
        ),
        side_effect=SideEffectClass.REVERSIBLE,
        reversible=True,
        source_phash="0" * 16,
        screen_size=(400, 800),
    )


def test_a_new_offer_becomes_work_and_activates_the_goal_it_serves():
    extension = extend_frontier(
        _observation((_control(),)), (_candidate(),), (GOAL,), ()
    )

    assert len(extension.frontier) == 1
    assert extension.frontier[0].relevant_goal_ids == ("goal-1",)
    assert extension.goals[0].state is CapabilityState.ACTIVE


def test_the_same_offer_seen_twice_is_work_once():
    first = extend_frontier(
        _observation((_control(),)), (_candidate(),), (GOAL,), ()
    )

    second = extend_frontier(
        _observation((_control(),)), (_candidate(),), first.goals, first.frontier
    )

    assert len(second.frontier) == 1


def test_an_offer_serving_only_settled_goals_is_not_work():
    settled = GOAL.activate().close(CapabilityState.REACHED, ("observation-1",))

    extension = extend_frontier(
        _observation((_control(),)), (_candidate(),), (settled,), ()
    )

    assert extension.frontier == ()


def test_an_offer_naming_another_observation_is_refused():
    with pytest.raises(ValueError, match="exact source observation"):
        extend_frontier(
            _observation((_control(),)),
            (_candidate(observation_id="somewhere-else"),),
            (GOAL,),
            (),
        )


def test_an_offer_naming_another_screen_state_is_refused():
    with pytest.raises(ValueError, match="exact source state"):
        extend_frontier(
            _observation((_control(),)),
            (_candidate(state_id="another-state"),),
            (GOAL,),
            (),
        )


def test_an_offer_the_observation_never_reported_is_refused():
    """A runtime cannot invent a target the run holds no evidence for."""
    with pytest.raises(ValueError, match="actionable observed control"):
        extend_frontier(
            _observation((_control(),)),
            (_candidate(identity="a-control-nobody-saw"),),
            (GOAL,),
            (),
        )


def test_an_offer_reported_as_inert_is_refused():
    with pytest.raises(ValueError, match="actionable observed control"):
        extend_frontier(
            _observation((_control(actionable=False),)),
            (_candidate(),),
            (GOAL,),
            (),
        )


def test_an_offer_naming_no_goal_is_refused():
    with pytest.raises(ValueError, match="known journey goals"):
        extend_frontier(
            _observation((_control(),)), (_candidate(goal_ids=()),), (GOAL,), ()
        )


def test_an_offer_naming_a_goal_this_run_does_not_hold_is_refused():
    with pytest.raises(ValueError, match="known journey goals"):
        extend_frontier(
            _observation((_control(),)),
            (_candidate(goal_ids=("goal-nobody-declared",)),),
            (GOAL,),
            (),
        )


def test_an_already_active_goal_is_not_activated_again():
    active = GOAL.activate()

    extension = extend_frontier(
        _observation((_control(),)), (_candidate(),), (active,), ()
    )

    assert extension.goals[0] is active
