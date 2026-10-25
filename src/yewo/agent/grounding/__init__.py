"""How a target the planner chose becomes a point on the device.

Each mode names a target differently. Marks mode names it by a number drawn on
the screenshot. Description mode names it in words. Both end in a point, and
everything after that point is shared.
"""

GROUNDING_COORDINATES = "coordinates"
"""Deprecated. The planner answers with a point of its own, and nothing
checks that point against what was detected. Marks mode answers the same
question and refuses an answer that names nothing."""
GROUNDING_MARKS = "marks"
GROUNDING_DESCRIBE = "describe"

from yewo.agent.grounding.placement import (  # noqa: E402
    MAX_SCREEN_FRACTION,
    MIN_PRESENCE,
    Grounder,
    Placement,
)

__all__ = [
    "GROUNDING_DESCRIBE",
    "GROUNDING_MARKS",
    "MAX_SCREEN_FRACTION",
    "MIN_PRESENCE",
    "Grounder",
    "Placement",
]
