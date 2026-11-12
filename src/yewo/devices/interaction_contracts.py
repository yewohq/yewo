"""Driver and platform-backend contracts for device interaction."""

from __future__ import annotations

import time
from typing import Protocol

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.focus import TextElement
from yewo.devices.interaction_records import TextInputConstraint
from yewo.devices.screen import ScreenCalibration, ScreenRegionReading
from yewo.devices.scroll import ScrollDirection, ScrollDispatch


class _SwitchTo(Protocol):
    @property
    def active_element(self) -> TextElement: ...

    @property
    def alert(self) -> object: ...


class InteractionClock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemInteractionClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class InteractionDriver(ApplicationLifecycleDriver, Protocol):
    """Driver methods used by core interaction and lifecycle operations."""

    @property
    def switch_to(self) -> _SwitchTo: ...

    def get_screenshot_as_png(self) -> bytes: ...

    def get_window_size(self) -> dict[str, int]: ...

    def is_keyboard_shown(self) -> bool: ...

    def hide_keyboard(self) -> object: ...

    def press_keycode(self, keycode: int, metastate: int | None = None) -> object: ...

    def back(self) -> None: ...


class PlatformInteractionBackend(Protocol):
    """Dispatch platform-specific interaction and context operations."""

    def dismiss_keyboard(self, driver: InteractionDriver) -> bool: ...

    def tap(self, driver: InteractionDriver, x: int, y: int) -> None: ...

    def double_tap(self, driver: InteractionDriver, x: int, y: int) -> None: ...

    def long_press(
        self, driver: InteractionDriver, x: int, y: int, duration_seconds: float
    ) -> None: ...

    def drag(
        self,
        driver: InteractionDriver,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_seconds: float,
    ) -> None: ...

    def scroll(
        self,
        driver: InteractionDriver,
        *,
        left: int,
        top: int,
        right: int,
        bottom: int,
        direction: ScrollDirection,
        distance_fraction: float,
    ) -> ScrollDispatch: ...

    def clear_text(
        self,
        driver: InteractionDriver,
        element: TextElement,
    ) -> None: ...

    def focused_text_constraint(
        self,
        driver: InteractionDriver,
    ) -> TextInputConstraint | None: ...

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def application_context(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def device_orientation(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def system_regions(
        self,
        driver: ApplicationLifecycleDriver,
        calibration: ScreenCalibration,
    ) -> tuple[ScreenRegionReading, ...] | None: ...

    def clear_application_data(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
    ) -> None: ...

    def open_deep_link(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
        url: str,
    ) -> None: ...


__all__ = [
    "InteractionClock",
    "InteractionDriver",
    "PlatformInteractionBackend",
    "SystemInteractionClock",
]
