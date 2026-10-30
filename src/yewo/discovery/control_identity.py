"""Name one control by the screen it sits on, not by the picture of that screen.

A perceptual fingerprint moves whenever any pixel does, a clock included. A run
already settles that movement once. It matches each new fingerprint against the
screens it has seen, so one screen keeps one identity.

Naming a control from the raw fingerprint throws that work away. Every control
on an unchanged screen is renamed the moment the clock ticks. The frontier then
re-offers the whole screen under the new names. The old names can never be
matched off it again, so a frontier holding them is never honestly exhausted.

The memory layer keys on the fingerprint deliberately. It asks a different
question: whether this is the same pixels as last time. That derivation stays
where it is. This one reuses its action semantics, so a tap and a long press on
one point remain different controls.
"""

from __future__ import annotations

from yewo.agent.actuation.actions import Action, PlannerTapAction
from yewo.agent.memory.control_identity import build_control_identity
from yewo.identity import derived_uuid

CONTROL_IDENTITY_VERSION = "control-identity-v2-screen-state"
"""Which naming scheme a stored frontier was recorded under.

Travels in the compatibility key. Knowledge named under the fingerprint scheme
is then revalidated rather than read as if it named the same controls.
"""


def discovery_control_identity(
    screen_state_id: str,
    action: Action | PlannerTapAction,
    screen_size: tuple[int, int],
) -> str:
    """Name the control one action addresses on one settled screen.

    Args:
        screen_state_id: The screen identity the run settled on, not a frame's
            fingerprint.
        action: The action a candidate carries, which supplies the point and
            the interaction that together address one control.
        screen_size: The device's width and height, so the point is recorded
            relative to the screen rather than in pixels.

    Returns:
        One identifier, stable across every frame of that settled screen.
    """
    identity = build_control_identity(action, None, screen_size)
    return derived_uuid(
        CONTROL_IDENTITY_VERSION,
        screen_state_id,
        identity.action_type,
        identity.coordinate_space,
        identity.points,
        identity.qualifier,
    )
