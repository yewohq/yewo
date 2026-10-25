"""Whether the text about to be typed will land in the field that was tapped.

Text goes to whatever holds focus. A tap can miss, and it would then write
into the previously focused field with nothing to show that it had. So Focus
reads focus back, checks it against the point asked for, and refuses the
action when it cannot.

A point stops being evidence the moment the layout moves. Tapping a field
opens the keyboard, and a form that scrolls to clear it carries the field off
the point that was used to tap it.

Focus holds the screen area as well as the driver, because some apps render as
one surface and focus it whole. An element that covers the screen contains
every point, so it cannot answer the question. It passes unchecked rather than
blocking every type in such an app.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from yewo.defaults import FOCUS_POLL_INTERVAL, FOCUS_POLL_TIMEOUT
from yewo.errors import ActionRejected, TargetNotInteractive

logger = logging.getLogger(__name__)

FULL_SCREEN_SHARE = 0.8


@dataclass(frozen=True)
class FocusedFieldReading:
    """The active element's value and current on-screen bounds."""

    value: Optional[str]
    bounds: Optional[tuple[int, int, int, int]]
    is_secure: Optional[bool] = None

NO_TEXT = (
    "did not reach the target you chose. The cursor did not move, so that "
    "element holds no text. It is a label, or a picture of a field. Choose an "
    "element that holds text. If no element on screen is the one you need, "
    "scroll or close the keyboard. Nothing was entered."
)
"""Why a refusal happened, in terms of the next thing to try.

The planner reads this and chooses again, so it has to name a different move.
In marks mode the planner never sees a coordinate, so a message about stale
coordinates would name nothing it can act on.
"""

ELSEWHERE = (
    "landed on a different element. The screen moved after you read it. Read "
    "the screen again. Then choose a target. Nothing was entered."
)


class Focus:
    """Reads focus back from one driver, against one screen size."""

    def __init__(self, driver, screen_area: int):
        self.driver = driver
        self.screen_area = screen_area

    def element(self):
        """The element holding focus, or None when the driver cannot say.

        Broad at this boundary. An unreachable session and a page with no focus
        mean the same thing here: nothing can be verified.
        """
        try:
            return self.driver.switch_to.active_element
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read the focused element: %s", exc)
            return None

    def _rect(self):
        """The focused element's bounds, or None when there are none to read."""
        element = self.element()
        if element is None:
            return None
        try:
            return element.rect
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read focused element bounds: %s", exc)
            return None

    def _covers_screen(self, rect) -> bool:
        width, height = rect.get("width", 0), rect.get("height", 0)
        return bool(self.screen_area) and width * height >= FULL_SCREEN_SHARE * self.screen_area

    @staticmethod
    def _holds(rect, x, y) -> bool:
        width, height = rect.get("width", 0), rect.get("height", 0)
        return (
            rect.get("x", 0) <= x <= rect.get("x", 0) + width
            and rect.get("y", 0) <= y <= rect.get("y", 0) + height
        )

    def contains(self, x, y) -> bool:
        """Whether the active element contains the requested coordinates.

        An element covering the screen answers for every point, so it cannot
        stand in for a real field.
        """
        rect = self._rect()
        if rect is None or self._covers_screen(rect):
            return False
        return self._holds(rect, x, y)

    def wait_for(self, x, y, before=None) -> None:
        """Polls until focus settles, instead of sleeping a flat second.

        A form that scrolls for the keyboard never puts the field back under
        the point, so waiting for that alone spends the whole budget.
        """
        for _ in range(max(1, int(FOCUS_POLL_TIMEOUT / FOCUS_POLL_INTERVAL))):
            rect = self._rect()
            if rect is not None and (
                self._covers_screen(rect) or self._holds(rect, x, y) or rect != before
            ):
                return
            time.sleep(FOCUS_POLL_INTERVAL)

    def rect(self):
        """The focused element's bounds, for comparing focus before and after."""
        return self._rect()

    def value(self):
        """The focused field's current text, or None when the driver cannot prove it.

        The platform driver owns these attributes.  They are stronger evidence
        than OCR because they come from the exact element whose focus was
        checked before typing.  An empty string is retained: it proves the
        field is empty rather than that the read was unavailable.
        """
        element = self.element()
        if element is None:
            return None
        return self._value_of(element)

    @staticmethod
    def _value_of(element):
        """Normalize one driver element's text at the driver boundary."""
        for name in ("value", "text"):
            try:
                observed = element.get_attribute(name)
            except Exception:  # noqa: BLE001 - driver boundary
                continue
            if isinstance(observed, str):
                return observed
        try:
            observed = element.text
        except Exception:  # noqa: BLE001 - driver boundary
            return None
        return observed if isinstance(observed, str) else None

    def reading(self) -> FocusedFieldReading:
        """Read value and bounds from the same active element."""
        element = self.element()
        if element is None:
            return FocusedFieldReading(value=None, bounds=None)
        try:
            rect = element.rect
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read focused element bounds: %s", exc)
            rect = None
        bounds = None
        if isinstance(rect, dict):
            x = rect.get("x")
            y = rect.get("y")
            width = rect.get("width")
            height = rect.get("height")
            if (
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and isinstance(width, (int, float))
                and isinstance(height, (int, float))
            ):
                bounds = (round(x), round(y), round(x + width), round(y + height))
        return FocusedFieldReading(
            value=self._value_of(element),
            bounds=bounds,
            is_secure=self._secure_state_of(element),
        )

    @staticmethod
    def _secure_state_of(element) -> Optional[bool]:
        """Normalize platform secure-field attributes at the driver boundary."""
        for name in ("password", "secure"):
            try:
                observed = element.get_attribute(name)
            except Exception:  # noqa: BLE001 - driver boundary
                continue
            if isinstance(observed, bool):
                return observed
            if isinstance(observed, str):
                normalized = observed.strip().casefold()
                if normalized in ("true", "1"):
                    return True
                if normalized in ("false", "0"):
                    return False
        for name in ("type", "className"):
            try:
                observed = element.get_attribute(name)
            except Exception:  # noqa: BLE001 - driver boundary
                continue
            if not isinstance(observed, str):
                continue
            normalized = observed.strip().casefold()
            if normalized == "password" or "securetextfield" in normalized:
                return True
        return None

    def assert_at(self, x, y, action: str, before=None, layout_moved=False) -> None:
        """Raises unless the tap put the cursor somewhere the text can go.

        Focus that did not move from `before` means the tap landed on
        something holding no text. `layout_moved` says the keyboard took the
        form with it, so the point no longer describes where the field is.
        """
        rect = self._rect()
        if rect is None:
            raise ActionRejected(
                f"{action} at ({x}, {y}) could not be checked, because nothing "
                "reported focus. Nothing was entered. Read the screen again and "
                "choose the field to use."
            )

        logger.debug("Focused element after tap: %s", rect)
        if self._covers_screen(rect):
            logger.debug("Focused element covers the screen, so the check is skipped.")
            return

        if self._holds(rect, x, y):
            return

        # An unchanged rect proves nothing scrolled, so it outranks layout_moved.
        if before is not None and rect == before:
            raise TargetNotInteractive(f"{action} {NO_TEXT}")

        if layout_moved:
            logger.info(
                "Focus is at %s, away from (%s, %s), because the keyboard moved "
                "the form after the tap.",
                rect,
                x,
                y,
            )
            return

        raise ActionRejected(f"{action} {ELSEWHERE}")
