"""Turn a declared journey into the goals bounded discovery pursues.

This is the boundary between what the product is meant to do and what Yewo
goes looking for. A goal states what to look for. It asserts nothing about the
application, so it names no screen, control, or step. Discovery answers it by
reaching the behaviour or by failing to.

The goal statement is the journey's own description, copied. V0 does not break
a journey into predicted sub-steps, because a sub-step such as "reach the cart"
would assert an application structure nobody has observed. Discovery is what
observes structure. A finer set of goals is decided from a map a run produced,
not from a guess made before one ran.
"""

from __future__ import annotations

from yewo.discovery.artifacts import validate_shared_knowledge_payload
from yewo.discovery.contracts import JourneyGoal
from yewo.identity import derived_uuid
from yewo.testing.models import UserJourney

GOAL_IDENTITY_VERSION = "journey-goal-v1"
"""Namespace for a goal's derived identifier.

Change it when the derivation changes. A stored AppMap names its capabilities
by goal identifier, so a changed derivation makes earlier knowledge
unreadable rather than wrong.
"""


def build_journey_goals(journey: UserJourney) -> tuple[JourneyGoal, ...]:
    """The goals one journey gives discovery.

    V0 returns exactly one goal, carrying the journey's whole description. The
    return stays a sequence because `DiscoveryRun` holds a sequence, and a
    later split needs no new signature.

    Args:
        journey: One resolved journey, including the values it declared
            sensitive.

    Returns:
        One required goal, pending, identified by a value derived from the
        journey rather than drawn at random.

    Raises:
        ValueError: The description states no goal, or it repeats a value the
            journey declared sensitive.
    """
    statement = " ".join(journey.description.split())
    if not statement:
        raise ValueError(
            f"Journey '{journey.id}' declares no description, so discovery has "
            "no goal to pursue."
        )
    validate_shared_knowledge_payload(
        statement,
        scenario_owned_values=tuple(journey.values.values()),
    )
    return (
        JourneyGoal(
            goal_id=derived_uuid(GOAL_IDENTITY_VERSION, journey.id, statement),
            statement=statement,
            required=True,
        ),
    )
