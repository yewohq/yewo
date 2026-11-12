"""Application-scoped interaction and verified lifecycle behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from yewo.applications.records import ApplicationDescriptor
from yewo.devices.application_lifecycle import (
    ApplicationLifecycle,
    ApplicationLifecycleDriver,
    ApplicationState,
    ForegroundApplicationReader,
    LifecycleClock,
)
from yewo.devices.errors import (
    ApplicationFirstPaintError,
    DeviceActionRejectedError,
    DeviceInteractionError,
)
from yewo.devices.inventory import DeviceKind, DevicePlatform
from yewo.devices.screen import is_blank
from yewo.observations.service import ScreenCapture


class ApplicationPlatformBackend(ForegroundApplicationReader, Protocol):
    """Platform operations needed by application-scoped interaction."""

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


class ApplicationInteraction:
    """Operate one application and verify lifecycle transitions."""

    def __init__(
        self,
        driver: ApplicationLifecycleDriver,
        platform: DevicePlatform,
        device_kind: DeviceKind | None,
        application: ApplicationDescriptor,
        backend: ApplicationPlatformBackend,
        clock: LifecycleClock,
        screenshot: Callable[[], ScreenCapture],
        *,
        transition_timeout_seconds: float,
        poll_interval_seconds: float,
        maximum_background_seconds: int,
    ) -> None:
        self._driver = driver
        self._platform = platform
        self._device_kind = device_kind
        self._application = application
        self._backend = backend
        self._clock = clock
        self._screenshot = screenshot
        self._lifecycle = ApplicationLifecycle(
            driver,
            application.application_id,
            backend,
            clock,
            transition_timeout_seconds=transition_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            maximum_background_seconds=maximum_background_seconds,
        )

    def state(self) -> ApplicationState:
        return self._lifecycle.state()

    def is_installed(self) -> bool:
        return self._lifecycle.is_installed()

    def is_foreground(self) -> bool:
        return self._lifecycle.is_foreground()

    def activate(self) -> None:
        self._lifecycle.activate()

    def terminate(self) -> None:
        self._lifecycle.terminate()

    def relaunch(self) -> None:
        self._lifecycle.relaunch()

    def background_and_resume(self, duration_seconds: int) -> None:
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
            raise DeviceActionRejectedError(
                "Background duration must be a whole number of seconds."
            )
        self._lifecycle.background_and_resume(duration_seconds)

    def clear_data(self) -> None:
        if self._platform is DevicePlatform.IOS and self._device_kind is not DeviceKind.VIRTUAL:
            raise DeviceActionRejectedError(
                "Clearing application data is supported only on an iOS simulator."
            )
        self._backend.clear_application_data(
            self._driver,
            self._application.application_id,
        )

    def open_deep_link(self, url: str) -> None:
        if not url.strip():
            raise DeviceActionRejectedError("A deep link URL cannot be empty.")
        self._backend.open_deep_link(
            self._driver,
            self._application.application_id,
            url,
        )

    def recover(self) -> bool:
        return self._lifecycle.recover()

    def active_application_id(self) -> str | None:
        return self._lifecycle.active_application_id()

    def is_active(self) -> bool | None:
        return self._lifecycle.is_active()

    def showing_external_ui(self) -> bool | None:
        return self._lifecycle.is_external_ui_visible()

    def wait_for_first_paint(
        self,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
        blank_standard_deviation: float,
    ) -> bytes:
        """Return the first nonblank screenshot before a bounded deadline."""
        if timeout_seconds <= 0 or poll_interval_seconds <= 0 or blank_standard_deviation < 0:
            raise DeviceInteractionError(
                "Paint time values must be positive. The blank threshold cannot be negative."
            )
        if not self._lifecycle.wait_until_active():
            raise ApplicationFirstPaintError(
                "The application did not become active before first paint."
            )
        deadline = self._clock.monotonic() + timeout_seconds
        while True:
            is_active = self.is_active()
            if is_active is not False:
                capture = self._screenshot()
                # Judged on the bytes the device gave, not on anything drawn over them. An
                # overlay is paint, and paint makes a blank screen look painted.
                if not is_blank(capture.comparable, blank_standard_deviation):
                    return capture.image
            remaining_seconds = deadline - self._clock.monotonic()
            if remaining_seconds <= 0:
                if is_active is False:
                    state = self.state().name.lower()
                    active_application_id = self.active_application_id() or "unknown"
                    raise ApplicationFirstPaintError(
                        "The application did not remain in the foreground before first paint. "
                        f"The application state is {state}. "
                        f"The foreground application is {active_application_id}."
                    )
                raise ApplicationFirstPaintError(
                    f"The application remained blank for {timeout_seconds:g} seconds."
                )
            self._clock.sleep(min(poll_interval_seconds, remaining_seconds))
