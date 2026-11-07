"""Tests for transactional background-resume capability behavior."""

from yewo.devices.capabilities import (
    ApplicationLifecycleCapabilityAdapter,
    CapabilitySupportState,
    CapabilityTransactionCoordinator,
    CapabilityTransactionState,
)
from yewo.devices import DevicePlatform
from yewo.devices.application_lifecycle import ApplicationState
from yewo.project_config.capabilities import AppStateCapabilitySpec


class _Lifecycle:
    def __init__(self, state: ApplicationState = ApplicationState.FOREGROUND) -> None:
        self.current_state = state
        self.background_durations: list[int] = []
        self.activation_count = 0

    def state(self) -> ApplicationState:
        return self.current_state

    def activate(self) -> None:
        self.activation_count += 1
        self.current_state = ApplicationState.FOREGROUND

    def background_and_resume(self, duration_seconds: int) -> None:
        self.background_durations.append(duration_seconds)
        self.current_state = ApplicationState.FOREGROUND


def _coordinator(lifecycle: _Lifecycle):
    return CapabilityTransactionCoordinator(
        "android-emulator",
        ApplicationLifecycleCapabilityAdapter(DevicePlatform.ANDROID, lifecycle),
        identifier_factory=lambda: "lifecycle-transaction",
    )


def test_background_resume_is_supported_with_a_whole_duration() -> None:
    lifecycle = _Lifecycle()
    request = AppStateCapabilitySpec(action="background_resume", duration_seconds=2)

    preparation = _coordinator(lifecycle).prepare(request)

    assert preparation.support.state is CapabilitySupportState.SUPPORTED
    assert preparation.transaction is not None


def test_fractional_background_duration_is_unsupported_before_mutation() -> None:
    lifecycle = _Lifecycle()
    request = AppStateCapabilitySpec(action="background_resume", duration_seconds=1.5)

    preparation = _coordinator(lifecycle).prepare(request)

    assert preparation.support.state is CapabilitySupportState.UNSUPPORTED
    assert preparation.transaction is None
    assert lifecycle.background_durations == []


def test_nonforeground_original_state_is_unavailable_before_mutation() -> None:
    lifecycle = _Lifecycle(ApplicationState.NOT_RUNNING)
    request = AppStateCapabilitySpec(action="background_resume")

    preparation = _coordinator(lifecycle).prepare(request)

    assert preparation.support.state is CapabilitySupportState.UNAVAILABLE
    assert preparation.transaction is None
    assert lifecycle.background_durations == []


def test_background_resume_is_applied_verified_and_restored() -> None:
    lifecycle = _Lifecycle()
    request = AppStateCapabilitySpec(action="background_resume", duration_seconds=2)
    transaction = _coordinator(lifecycle).prepare(request).transaction
    assert transaction is not None

    with transaction:
        assert lifecycle.current_state is ApplicationState.FOREGROUND

    assert lifecycle.background_durations == [2]
    assert lifecycle.activation_count == 0
    assert transaction.receipt is not None
    assert transaction.receipt.application.observed_state is not None
    assert transaction.receipt.application.observed_state.value == "foreground"
    assert transaction.receipt.state is CapabilityTransactionState.RESTORED


def test_work_that_stops_the_app_is_restored_to_foreground() -> None:
    lifecycle = _Lifecycle()
    transaction = (
        _coordinator(lifecycle)
        .prepare(AppStateCapabilitySpec(action="background_resume"))
        .transaction
    )
    assert transaction is not None

    with transaction:
        lifecycle.current_state = ApplicationState.NOT_RUNNING

    assert lifecycle.activation_count == 1
    assert lifecycle.current_state is ApplicationState.FOREGROUND
    assert transaction.receipt is not None
    assert transaction.receipt.restoration.verified
