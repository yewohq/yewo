"""How many steps in a row left the screen untouched, and the one way out.

A step that never reaches the device costs a model call and changes nothing.
Repeated, it spends a scenario's budget while reporting "steps exhausted",
which reads like a long journey rather than a stuck one.

Counting is per screen. Scrolling or dismissing a keyboard changes the screen,
so a planner that reacts to a refusal gets a fresh count, and only one that
keeps asking the same screen runs out.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

STALL_LIMIT = 3
"""Blocked steps on one screen before the scenario stops.

Two leaves room to read a refusal and try something else. The third says
reading it did not help.
"""


STALL_LIMIT_MESSAGE = "Nothing on this screen accepted an action after repeated attempts."


@dataclass
class Stall:
    """Blocked steps on the screen they were blocked on."""

    phash: Optional[str] = None
    count: int = 0

    def action_dispatched(self) -> None:
        """An action reached the device.

        Says nothing about whether the screen then changed, only that something
        was sent. Stagnation on a screen that does accept actions is the loop
        detector's business.
        """
        self.phash = None
        self.count = 0

    def record_blocked(self, phash: Optional[str]) -> None:
        """Records a step that never reached the device.

        An unidentified screen cannot be compared with the next one, so it
        never matches and never accumulates, which is the lenient reading.
        """
        if phash is None or phash != self.phash:
            self.phash, self.count = phash, 0
        self.count += 1

    @property
    def is_blocked(self) -> bool:
        """Whether the last step on this screen failed to reach the device."""
        return self.count > 0

    @property
    def is_exhausted(self) -> bool:
        """Whether this screen has used up its attempts."""
        return self.count >= STALL_LIMIT


def reveal_behind_keyboard(device_control, snapshot, ctx) -> bool:
    """Pulls the keyboard down once per screen when a step cannot proceed.

    The keyboard hides the lower half, and only what is on screen gets a
    number, so a target the planner cannot name is often underneath it.
    Done rather than asked for, because a planner told to scroll does not
    reliably do it.

    Once per screen. If the keyboard returns and the step fails again,
    something other than the keyboard is wrong.
    """
    screen = snapshot.current_phash
    if not device_control.is_keyboard_open() or screen is None:
        return False
    if screen in ctx.keyboard_dismissed:
        return False

    logger.info("Step blocked with the keyboard open. Closing it to see the rest.")
    if not device_control.dismiss_keyboard():
        return False
    ctx.keyboard_dismissed.add(screen)
    return True
