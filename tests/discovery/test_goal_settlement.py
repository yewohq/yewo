"""What one set of receipts is allowed to settle.

A goal closes from a receipt, never from a claim. These cases hold the receipt
to naming the observation and the transition the controller accepted, and hold
the frontier to carrying no offer whose every goal is answered.
"""

import pytest

from yewo.discovery.contracts import (
    CapabilityState,
    FrontierState,
    JourneyGoal,
    RelevantFrontierEntry,
)
from yewo.discovery.goal_settlement import settle_goals
from yewo.discovery.runtime import CapabilityEvidenceReceipt

GOAL = JourneyGoal(goal_id="goal-1", statement="A user places an order.")
OBSERVATION = "observation-1"
TRANSITION = "transition-1"


def _offer(goal_ids=("goal-1",), state=FrontierState.AVAILABLE):
    return RelevantFrontierEntry(
        frontier_id="frontier-1",
        source_observation_id=OBSERVATION,
        control_identity="control-1",
        relevant_goal_ids=goal_ids,
        state=state,
    )


def _receipt(kind="route_verified", transition=TRANSITION, observation=OBSERVATION):
    return CapabilityEvidenceReceipt(
        goal_id="goal-1",
        observation_id=observation,
        kind=kind,
        predicate_id="predicate-1",
        transition_identity=transition,
    )


def test_a_verified_route_closes_its_goal():
    settlement = settle_goals(
        (_receipt(),), (GOAL.activate(),), (), OBSERVATION, TRANSITION
    )

    assert settlement.goals[0].state is CapabilityState.ROUTE_VERIFIED
    assert settlement.goals[0].evidence_observation_ids == (OBSERVATION,)


def test_an_observation_without_a_transition_closes_its_goal_as_reached():
    settlement = settle_goals(
        (_receipt(kind="observed", transition=None),),
        (GOAL.activate(),),
        (),
        OBSERVATION,
        None,
    )

    assert settlement.goals[0].state is CapabilityState.REACHED


def test_a_pending_goal_is_activated_before_it_closes():
    settlement = settle_goals((_receipt(),), (GOAL,), (), OBSERVATION, TRANSITION)

    assert settlement.goals[0].state is CapabilityState.ROUTE_VERIFIED


def test_an_offer_whose_every_goal_is_answered_is_retired():
    """A run may not hold available work nothing needs."""
    settlement = settle_goals(
        (_receipt(),), (GOAL.activate(),), (_offer(),), OBSERVATION, TRANSITION
    )

    assert settlement.frontier[0].state is FrontierState.ATTEMPTED


def test_an_offer_still_serving_an_open_goal_is_left_alone():
    other = JourneyGoal(goal_id="goal-2", statement="A user signs in.")

    settlement = settle_goals(
        (_receipt(),),
        (GOAL.activate(), other),
        (_offer(goal_ids=("goal-1", "goal-2")),),
        OBSERVATION,
        TRANSITION,
    )

    assert settlement.frontier[0].state is FrontierState.AVAILABLE


def test_an_already_closed_offer_is_not_closed_again():
    settlement = settle_goals(
        (_receipt(),),
        (GOAL.activate(),),
        (_offer(state=FrontierState.VERIFIED),),
        OBSERVATION,
        TRANSITION,
    )

    assert settlement.frontier[0].state is FrontierState.VERIFIED


def test_a_receipt_naming_another_observation_is_refused():
    with pytest.raises(ValueError, match="exact observation"):
        settle_goals(
            (_receipt(observation="somewhere-else"),),
            (GOAL.activate(),),
            (),
            OBSERVATION,
            TRANSITION,
        )


def test_a_receipt_naming_a_goal_this_run_does_not_hold_is_refused():
    other = JourneyGoal(goal_id="goal-2", statement="A user signs in.")

    with pytest.raises(ValueError, match="unknown goal"):
        settle_goals((_receipt(),), (other,), (), OBSERVATION, TRANSITION)


def test_a_route_claim_naming_an_unaccepted_transition_is_refused():
    with pytest.raises(ValueError, match="accepted transition"):
        settle_goals(
            (_receipt(transition="never-accepted"),),
            (GOAL.activate(),),
            (),
            OBSERVATION,
            TRANSITION,
        )


def test_a_route_claim_with_no_accepted_transition_at_all_is_refused():
    with pytest.raises(ValueError, match="accepted transition"):
        settle_goals((_receipt(),), (GOAL.activate(),), (), OBSERVATION, None)


def test_an_observed_claim_cannot_also_claim_a_transition():
    with pytest.raises(ValueError, match="cannot claim a transition"):
        settle_goals(
            (_receipt(kind="observed"),),
            (GOAL.activate(),),
            (),
            OBSERVATION,
            TRANSITION,
        )


def test_a_settled_goal_is_left_as_it_was():
    settled = GOAL.activate().close(CapabilityState.REACHED, (OBSERVATION,))

    settlement = settle_goals((_receipt(),), (settled,), (), OBSERVATION, TRANSITION)

    assert settlement.goals[0] is settled
