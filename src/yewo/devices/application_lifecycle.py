"""Bounded application lifecycle transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Protocol

from yewo.devices.errors import DeviceActionRejectedError, DeviceInteractionError
from yewo.devices.inventory import DevicePlatform


class ApplicationState(IntEnum):
    """Appium application lifecycle states."""

    NOT_INSTALLED = 0
    NOT_RUNNING = 1
    BACKGROUND_SUSPENDED = 2
    BACKGROUND = 3
    FOREGROUND = 4


DEFAULT_MAXIMUM_BACKGROUND_SECONDS = 30


class ApplicationActivityStatus(StrEnum):
    """Strength of evidence that the application owns the foreground."""

    VERIFIED_ACTIVE = "verified_active"
    STATE_ONLY_ACTIVE = "state_only_active"
    INACTIVE = "inactive"


class TrustedSystemSurface(StrEnum):
    """A platform-owned surface that can participate in an app journey."""

    RUNTIME_PERMISSION = "runtime_permission"
    SYSTEM_UI = "system_ui"


_ANDROID_PERMISSION_CONTROLLER_IDS = frozenset(
    {
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
    }
)

_ANDROID_SYSTEM_UI_IDS = frozenset({"com.android.systemui"})


def trusted_system_surface(
    platform: DevicePlatform,
    foreground_application_id: str | None,
) -> TrustedSystemSurface | None:
    """Classify a narrowly approved platform-owned foreground surface."""
    if (
        platform is DevicePlatform.ANDROID
        and foreground_application_id in _ANDROID_PERMISSION_CONTROLLER_IDS
    ):
        return TrustedSystemSurface.RUNTIME_PERMISSION
    if platform is DevicePlatform.ANDROID and foreground_application_id in _ANDROID_SYSTEM_UI_IDS:
        return TrustedSystemSurface.SYSTEM_UI
    return None


@dataclass(frozen=True)
class ApplicationActivityEvidence:
    """One application-state and foreground-identity reading."""

    state: ApplicationState
    foreground_application_id: str | None
    status: ApplicationActivityStatus


class ApplicationLifecycleDriver(Protocol):
    """Driver operations used by application lifecycle transitions."""

    def execute_script(self, script: str, params: dict[str, Any]) -> Any: ...

    def is_app_installed(self, application_id: str) -> bool: ...

    def query_app_state(self, application_id: str) -> int: ...

    def activate_app(self, application_id: str) -> object: ...

    def terminate_app(self, application_id: str) -> bool: ...

    def background_app(self, seconds: int) -> object: ...


class ForegroundApplicationReader(Protocol):
    """Read the application that currently receives input."""

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...


class LifecycleClock(Protocol):
    """Provide monotonic time for bounded lifecycle polling."""

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class ApplicationLifecycle:
    """Dispatch one lifecycle command and verify its observed result."""

    def __init__(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
        foreground_reader: ForegroundApplicationReader,
        clock: LifecycleClock,
        *,
        transition_timeout_seconds: float,
        poll_interval_seconds: float,
        maximum_background_seconds: int = DEFAULT_MAXIMUM_BACKGROUND_SECONDS,
    ) -> None:
        if transition_timeout_seconds <= 0:
            raise ValueError("The application transition timeout must be greater than zero.")
        if poll_interval_seconds <= 0:
            raise ValueError("The application poll interval must be greater than zero.")
        if maximum_background_seconds < 1:
            raise ValueError("The maximum background duration must be at least one second.")
        self._driver = driver
        self._application_id = application_id
        self._foreground_reader = foreground_reader
        self._clock = clock
        self._transition_timeout_seconds = transition_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._maximum_background_seconds = maximum_background_seconds

    def state(self) -> ApplicationState:
        try:
            return ApplicationState(self._driver.query_app_state(self._application_id))
        except (ValueError, TypeError) as error:
            raise DeviceInteractionError(
                "The driver returned an invalid application state."
            ) from error

    def is_installed(self) -> bool:
        return self._driver.is_app_installed(self._application_id)

    def is_foreground(self) -> bool:
        return self.state() is ApplicationState.FOREGROUND

    def active_application_id(self) -> str | None:
        return self._foreground_reader.active_application_id(self._driver)

    def activity_evidence(self) -> ApplicationActivityEvidence:
        """Return foreground evidence without converting missing identity to false."""
        state = self.state()
        foreground_application_id = self.active_application_id()
        if state is not ApplicationState.FOREGROUND:
            status = ApplicationActivityStatus.INACTIVE
        elif foreground_application_id == self._application_id:
            status = ApplicationActivityStatus.VERIFIED_ACTIVE
        elif foreground_application_id is None:
            status = ApplicationActivityStatus.STATE_ONLY_ACTIVE
        else:
            status = ApplicationActivityStatus.INACTIVE
        return ApplicationActivityEvidence(
            state=state,
            foreground_application_id=foreground_application_id,
            status=status,
        )

    def is_active(self) -> bool | None:
        status = self.activity_evidence().status
        if status is ApplicationActivityStatus.VERIFIED_ACTIVE:
            return True
        if status is ApplicationActivityStatus.STATE_ONLY_ACTIVE:
            return None
        return False

    def is_external_ui_visible(self) -> bool | None:
        active_application_id = self.active_application_id()
        if active_application_id is None:
            return None
        return active_application_id != self._application_id

    def activate(self) -> None:
        evidence = self._activate_and_wait()
        if not self._is_active_evidence(evidence):
            raise self._transition_error("activate", evidence)

    def terminate(self) -> None:
        if self.state() is ApplicationState.NOT_INSTALLED:
            raise DeviceActionRejectedError(
                "Cannot terminate an application that is not installed."
            )
        self._driver.terminate_app(self._application_id)
        evidence = self._wait_until(
            lambda current: (
                current.state in {ApplicationState.NOT_RUNNING, ApplicationState.NOT_INSTALLED}
            )
        )
        if evidence.state is ApplicationState.NOT_INSTALLED:
            raise DeviceActionRejectedError(
                "The application became unavailable while termination was verified."
            )
        if evidence.state is not ApplicationState.NOT_RUNNING:
            raise self._transition_error("terminate", evidence)

    def relaunch(self) -> None:
        self.terminate()
        evidence = self._activate_and_wait()
        if not self._is_active_evidence(evidence):
            raise self._transition_error("relaunch", evidence)

    def background_and_resume(self, duration_seconds: int) -> None:
        if duration_seconds < 1 or duration_seconds > self._maximum_background_seconds:
            raise DeviceActionRejectedError("Background duration is outside the allowed range.")
        self._driver.background_app(duration_seconds)
        evidence = self._wait_until(self._is_active_evidence)
        if not self._is_active_evidence(evidence):
            raise self._transition_error("resume", evidence)

    def recover(self) -> bool:
        if not self.is_installed():
            return False
        if self.is_active() is not False:
            return True
        return self._is_active_evidence(self._activate_and_wait())

    def require_active(self, operation: str) -> None:
        """Wait for active foreground ownership or raise a diagnostic timeout."""
        evidence = self._wait_until(self._is_active_evidence)
        if not self._is_active_evidence(evidence):
            raise self._transition_error(operation, evidence)

    def wait_until_active(self) -> bool:
        """Poll read-only state until the application owns the foreground."""
        return self._is_active_evidence(self._wait_until(self._is_active_evidence))

    def _wait_until(
        self,
        is_acceptable: Callable[[ApplicationActivityEvidence], bool],
    ) -> ApplicationActivityEvidence:
        deadline = self._clock.monotonic() + self._transition_timeout_seconds
        while True:
            evidence = self.activity_evidence()
            if is_acceptable(evidence):
                return evidence
            remaining_seconds = deadline - self._clock.monotonic()
            if remaining_seconds <= 0:
                return evidence
            self._clock.sleep(min(self._poll_interval_seconds, remaining_seconds))

    def _activate_and_wait(self) -> ApplicationActivityEvidence:
        self._driver.activate_app(self._application_id)
        return self._wait_until(self._is_active_evidence)

    @staticmethod
    def _is_active_evidence(evidence: ApplicationActivityEvidence) -> bool:
        return evidence.status in {
            ApplicationActivityStatus.VERIFIED_ACTIVE,
            ApplicationActivityStatus.STATE_ONLY_ACTIVE,
        }

    def _transition_error(
        self,
        operation: str,
        evidence: ApplicationActivityEvidence,
    ) -> DeviceActionRejectedError:
        active_application_id = evidence.foreground_application_id or "unknown"
        return DeviceActionRejectedError(
            f"Could not {operation} the application within "
            f"{self._transition_timeout_seconds:g} seconds. "
            f"The application state is {evidence.state.name.lower()}. "
            f"The foreground application is {active_application_id}."
        )
