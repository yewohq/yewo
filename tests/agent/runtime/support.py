"""Shared device fake for the runtime loop tests."""

from datetime import datetime, timedelta, timezone
from typing import Any

from yewo.agent.actuation import ActionKind, ActionRefusal, ActionRefusalCode
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.devices.focus import FocusedFieldReading
from yewo.devices.interaction import ScrollDirection
from yewo.observations import Observation, ScreenFingerprint, VisualSettlementProbe
from yewo.observations.errors import ObservationCaptureError

from agent.actuation.support import proposal
from observations.factories import render_observation


class RuntimeDevice:
    """Record dispatches and expose only secret-free post-action checks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.active_text = ""
        self.reject = False

    def _record(self, operation: str, *arguments: Any) -> None:
        if self.reject:
            from yewo.devices.errors import DeviceActionRejectedError

            raise DeviceActionRejectedError("Configured refusal with private detail.")
        self.calls.append((operation, arguments))

    def tap(self, x: int, y: int, *, count: int = 1, interval_seconds: float = 0.1) -> None:
        self._record("tap", x, y, count, interval_seconds)

    def focus(self, x: int, y: int) -> object:
        self._record("focus", x, y)
        return object()

    def type_focused_text(self, text: str) -> bool | None:
        self._record("type_focused_text", text)
        self.active_text = text
        return True

    def clear_focused_text(self) -> bool | None:
        self._record("clear_focused_text")
        self.active_text = ""
        return True

    def double_tap(self, x: int, y: int) -> None:
        self._record("double_tap", x, y)

    def long_press(self, x: int, y: int, duration_seconds: float) -> None:
        self._record("long_press", x, y, duration_seconds)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 1,
    ) -> None:
        self._record("drag", start_x, start_y, end_x, end_y, duration_seconds)

    def scroll(
        self,
        direction: ScrollDirection,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
        distance_fraction: float = 0.6,
    ) -> None:
        self._record(
            "scroll",
            direction,
            left,
            top,
            right,
            bottom,
            distance_fraction,
        )

    def refresh(self, **bounds: int | None) -> None:
        self._record("refresh", bounds)

    def back(self) -> None:
        self._record("back")

    def dismiss_keyboard(self) -> bool:
        self._record("dismiss_keyboard")
        return True

    def wait(self, duration_seconds: float) -> None:
        self._record("wait", duration_seconds)

    def focused_field_reading(self) -> FocusedFieldReading:
        return FocusedFieldReading(True, (0, 0, 100, 50), True)


SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX = 200, 400


class RenderedScreenSequence:
    """Capture observations from real pixels, with the run's covers drawn into them.

    A hand-written fingerprint cannot reflect real pixels, so a double that supplies one hides
    every decision that compares one screen against another.
    """

    def __init__(self, screens: list[bytes]) -> None:
        self._screens = iter(screens)
        self._current: Observation | None = None
        self._pending_probe: Observation | None = None
        self.capture_count = 0

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
        if self._pending_probe is not None:
            self._current = self._pending_probe.model_copy(
                update={"predecessor_observation_id": predecessor_observation_id}
            )
            self._pending_probe = None
            return self._current
        self.capture_count += 1
        self._current = render_observation(
            next(self._screens),
            observation_id=f"obs_{self.capture_count}",
            captured_at=datetime(2026, 10, 30, tzinfo=timezone.utc)
            + timedelta(seconds=self.capture_count),
            predecessor_observation_id=predecessor_observation_id,
        )
        return self._current

    def capture_settlement_probe(self) -> VisualSettlementProbe:
        self.capture_count += 1
        self._pending_probe = render_observation(
            next(self._screens),
            observation_id=f"obs_{self.capture_count}",
            captured_at=datetime(2026, 10, 30, tzinfo=timezone.utc)
            + timedelta(seconds=self.capture_count),
        )
        self._current = self._pending_probe
        return VisualSettlementProbe.from_observation(self._pending_probe)

    def capture_screen_fingerprint(self) -> ScreenFingerprint:
        if self._current is None:
            raise ObservationCaptureError("No screen has been captured yet.")
        return self._current.screen_fingerprint


def exhaust_screen(
    tracker: ActionNoProgressTracker,
    screen: Observation,
    label: str,
) -> None:
    """Refuse three grounding attempts on one screen, which is what exhausts it."""
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"{label}-{index}",
            target_expression=f"{label} control {index}",
            source_observation_id=screen.observation_id,
        )
        tracker.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="The grounder did not accept this target.",
            ),
        )
