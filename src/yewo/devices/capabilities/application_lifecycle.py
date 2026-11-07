"""Transactional application-lifecycle capability adapter."""

from __future__ import annotations

from typing import Protocol

from yewo.devices.capabilities.records import (
    CapabilityStateObservation,
    CapabilitySupportResult,
    CapabilitySupportState,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import DevicePlatform
from yewo.project_config.capabilities import AppStateCapabilitySpec, CapabilityKind


class ApplicationLifecycleCapabilityBoundary(Protocol):
    """Verified lifecycle operations required by the capability adapter."""

    def state(self) -> ApplicationState: ...

    def activate(self) -> None: ...

    def background_and_resume(self, duration_seconds: int) -> None: ...


class ApplicationLifecycleCapabilityAdapter:
    """Apply and restore the background-resume application capability."""

    def __init__(
        self,
        platform: DevicePlatform,
        lifecycle: ApplicationLifecycleCapabilityBoundary,
    ) -> None:
        self._platform = platform
        self._lifecycle = lifecycle

    def support(self, request: AppStateCapabilitySpec) -> CapabilitySupportResult:
        if request.action != "background_resume":
            return self._support(
                CapabilitySupportState.UNSUPPORTED,
                "This capability adapter supports only background_resume.",
            )
        duration_seconds = request.duration_seconds
        if duration_seconds is None or not float(duration_seconds).is_integer():
            return self._support(
                CapabilitySupportState.UNSUPPORTED,
                "Background duration must be a whole number of seconds.",
            )
        return self._support(
            CapabilitySupportState.SUPPORTED,
            f"{self._platform.value} supports verified application background and resume.",
        )

    def read_state(self, request: AppStateCapabilitySpec) -> CapabilityStateObservation:
        del request
        return CapabilityStateObservation(
            value=self._lifecycle.state().name.lower(),
            source="Appium queryAppState",
        )

    def validate_original_state(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
    ) -> CapabilitySupportResult:
        del request
        if original_state.value != ApplicationState.FOREGROUND.name.lower():
            return self._support(
                CapabilitySupportState.UNAVAILABLE,
                "Background-resume requires an application that starts in the foreground.",
            )
        return self._support(
            CapabilitySupportState.SUPPORTED,
            "The original foreground application state is readable and restorable.",
        )

    def requested_state(self, request: AppStateCapabilitySpec) -> str:
        return f"background_resume:{int(request.duration_seconds or 0)}s"

    def apply(self, request: AppStateCapabilitySpec) -> None:
        self._lifecycle.background_and_resume(int(request.duration_seconds or 0))

    def application_verified(
        self,
        request: AppStateCapabilitySpec,
        observed_state: CapabilityStateObservation,
    ) -> bool:
        del request
        return observed_state.value == ApplicationState.FOREGROUND.name.lower()

    def restore(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
    ) -> None:
        del request
        if original_state.value != ApplicationState.FOREGROUND.name.lower():
            raise ValueError("The original application state is not restorable here.")
        if self._lifecycle.state() is not ApplicationState.FOREGROUND:
            self._lifecycle.activate()

    def restoration_verified(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
        observed_state: CapabilityStateObservation,
    ) -> bool:
        del request
        return observed_state.value == original_state.value

    @staticmethod
    def _support(
        state: CapabilitySupportState,
        reason: str,
    ) -> CapabilitySupportResult:
        return CapabilitySupportResult(
            capability=CapabilityKind.APP_STATE,
            state=state,
            reason=reason,
        )
