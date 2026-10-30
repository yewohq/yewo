"""A journey states what to look for. It is not evidence about the app.

These cases hold the boundary to that: the goal carries the journey's own
words, it is identified the same way on every run, and it refuses a journey
whose description would put a declared secret into shared app knowledge.
"""

import pytest

from yewo.discovery.contracts import CapabilityState
from yewo.testing.journey_goals import build_journey_goals
from yewo.testing.models import UserJourney


def _journey(description="A user signs in and places an order.", values=None):
    return UserJourney(
        id="checkout",
        name="Checkout",
        description=description,
        values=values or {},
    )


def test_one_journey_gives_one_required_pending_goal():
    goals = build_journey_goals(_journey())

    assert len(goals) == 1
    assert goals[0].required
    assert goals[0].state is CapabilityState.PENDING
    assert goals[0].evidence_observation_ids == ()


def test_the_goal_carries_the_journey_description_verbatim():
    """Nothing decomposes it. A sub-step would assert structure nobody observed."""
    goals = build_journey_goals(_journey())

    assert goals[0].statement == "A user signs in and places an order."


def test_the_statement_is_whitespace_normalized():
    """So reformatting a journey file does not rename its goal."""
    wrapped = build_journey_goals(_journey("A user signs in\n  and places   an order."))
    inline = build_journey_goals(_journey("A user signs in and places an order."))

    assert wrapped[0].statement == inline[0].statement
    assert wrapped[0].goal_id == inline[0].goal_id


def test_the_goal_identifier_is_derived_so_two_runs_agree():
    """A stored AppMap names its capabilities by goal id. A drawn id would orphan it."""
    first = build_journey_goals(_journey())
    second = build_journey_goals(_journey())

    assert first[0].goal_id == second[0].goal_id


def test_a_different_journey_gets_a_different_goal_identifier():
    other = UserJourney(id="signup", name="Signup", description="A user signs up.")

    assert build_journey_goals(_journey())[0].goal_id != build_journey_goals(other)[0].goal_id


def test_a_changed_description_is_a_different_goal():
    changed = build_journey_goals(_journey("A user signs in and cancels an order."))

    assert build_journey_goals(_journey())[0].goal_id != changed[0].goal_id


def test_a_journey_stating_no_goal_is_refused_by_name():
    with pytest.raises(ValueError, match="checkout"):
        build_journey_goals(_journey("   \n  "))


def test_a_description_repeating_a_declared_secret_is_refused():
    """The statement reaches the AppMap, which is shared knowledge."""
    exposing = _journey(
        "A user signs in with 4242424242424242 and places an order.",
        values={"card": "4242424242424242"},
    )

    with pytest.raises(ValueError, match="isolated scenario value"):
        build_journey_goals(exposing)


def test_a_declared_value_the_description_does_not_repeat_is_allowed():
    safe = _journey(values={"card": "4242424242424242"})

    assert build_journey_goals(safe)[0].statement.endswith("places an order.")
