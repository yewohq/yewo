"""Tests for bounded application lifecycle transitions."""

from typing import Any

import pytest

from yewo.devices.application_lifecycle import (
    ApplicationActivityStatus,
    ApplicationLifecycle,
    ApplicationLifecycleDriver,
    ApplicationState,
)
from yewo.devices.errors import DeviceInteractionError
from support import ClockStub


APPLICATION_ID = "com.example.app"
EXTERNAL_APPLICATION_ID = "com.example.launcher"


class _DriverStub:
    def __init__(self, states: list[int]) -> None:
        self.states = states
        self.state = states[0]
        self.is_installed = True
        self.current_application_id: str | None = EXTERNAL_APPLICATION_ID
        self.lifecycle_calls: list[str] = []

    def execute_script(self, script: str, params: dict[str, Any]) -> Any:
        del script, params
        return None

    def is_app_installed(self, application_id: str) -> bool:
        assert application_id == APPLICATION_ID
        return self.is_installed

    def query_app_state(self, application_id: str) -> int:
        assert application_id == APPLICATION_ID
        if len(self.states) > 1:
            self.state = self.states.pop(0)
        else:
            self.state = self.states[0]
        if self.state == ApplicationState.FOREGROUND:
            self.current_application_id = application_id
        return self.state

    def activate_app(self, application_id: str) -> object:
        self.lifecycle_calls.append(f"activate:{application_id}")
        return self

    def terminate_app(self, application_id: str) -> bool:
        self.lifecycle_calls.append(f"terminate:{application_id}")
        self.state = ApplicationState.NOT_RUNNING
        return True

    def background_app(self, seconds: int) -> object:
        self.lifecycle_calls.append(f"background:{seconds}")
        return self


class _ForegroundReader:
    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        assert isinstance(driver, _DriverStub)
        return driver.current_application_id


def _lifecycle(
    driver: _DriverStub,
    clock: ClockStub,
) -> ApplicationLifecycle:
    return ApplicationLifecycle(
        driver,
        APPLICATION_ID,
        _ForegroundReader(),
        clock,
        transition_timeout_seconds=0.3,
        poll_interval_seconds=0.1,
    )


def test_activation_waits_for_verified_foreground_identity() -> None:
    clock = ClockStub()
    driver = _DriverStub([1, 3, 4])

    _lifecycle(driver, clock).activate()

    assert driver.lifecycle_calls == ["activate:com.example.app"]
    assert clock.now == pytest.approx(0.2)


def test_activation_times_out_without_repeating_the_command() -> None:
    clock = ClockStub()
    driver = _DriverStub([1])

    with pytest.raises(DeviceInteractionError, match="Could not activate") as failure:
        _lifecycle(driver, clock).activate()

    assert "not_running" in str(failure.value)
    assert EXTERNAL_APPLICATION_ID in str(failure.value)
    assert driver.lifecycle_calls == ["activate:com.example.app"]
    assert clock.now == pytest.approx(0.3)


def test_recovery_activates_an_installed_background_application() -> None:
    clock = ClockStub()
    driver = _DriverStub([3, 4])

    recovered = _lifecycle(driver, clock).recover()

    assert recovered is True
    assert driver.lifecycle_calls == ["activate:com.example.app"]
    assert clock.now == 0


def test_recovery_returns_false_after_a_bounded_timeout() -> None:
    clock = ClockStub()
    driver = _DriverStub([3])

    recovered = _lifecycle(driver, clock).recover()

    assert recovered is False
    assert driver.lifecycle_calls == ["activate:com.example.app"]
    assert clock.now == pytest.approx(0.3)


def test_recovery_does_not_activate_an_absent_application() -> None:
    clock = ClockStub()
    driver = _DriverStub([0])
    driver.is_installed = False

    recovered = _lifecycle(driver, clock).recover()

    assert recovered is False
    assert driver.lifecycle_calls == []
    assert clock.now == 0


def test_relaunch_waits_for_verified_foreground_identity() -> None:
    clock = ClockStub()
    driver = _DriverStub([1, 1, 4])

    _lifecycle(driver, clock).relaunch()

    assert driver.lifecycle_calls == [
        "terminate:com.example.app",
        "activate:com.example.app",
    ]
    assert clock.now == 0


def test_relaunch_does_not_activate_when_termination_is_unverified() -> None:
    clock = ClockStub()
    driver = _DriverStub([4])

    with pytest.raises(DeviceInteractionError, match="Could not terminate"):
        _lifecycle(driver, clock).relaunch()

    assert driver.lifecycle_calls == ["terminate:com.example.app"]
    assert clock.now == pytest.approx(0.3)


def test_terminate_rejects_an_absent_application_without_dispatch() -> None:
    clock = ClockStub()
    driver = _DriverStub([ApplicationState.NOT_INSTALLED])

    with pytest.raises(DeviceInteractionError, match="not installed"):
        _lifecycle(driver, clock).terminate()

    assert driver.lifecycle_calls == []
    assert clock.now == 0


def test_missing_foreground_identity_preserves_state_only_evidence() -> None:
    clock = ClockStub()
    driver = _DriverStub([4])

    class MissingForegroundReader:
        def active_application_id(
            self,
            driver: ApplicationLifecycleDriver,
        ) -> None:
            del driver
            return None

    lifecycle = ApplicationLifecycle(
        driver,
        APPLICATION_ID,
        MissingForegroundReader(),
        clock,
        transition_timeout_seconds=0.3,
        poll_interval_seconds=0.1,
    )

    evidence = lifecycle.activity_evidence()

    assert evidence.status is ApplicationActivityStatus.STATE_ONLY_ACTIVE
    assert lifecycle.is_active() is None
    assert lifecycle.wait_until_active() is True


@pytest.mark.parametrize("duration", [0, 31])
def test_background_duration_fails_before_dispatch(duration: int) -> None:
    clock = ClockStub()
    driver = _DriverStub([4])

    with pytest.raises(DeviceInteractionError, match="outside the allowed range"):
        _lifecycle(driver, clock).background_and_resume(duration)

    assert driver.lifecycle_calls == []


def test_background_waits_for_the_application_to_resume() -> None:
    class BackgroundDriver(_DriverStub):
        def background_app(self, seconds: int) -> object:
            self.lifecycle_calls.append(f"background:{seconds}")
            self.states = [3, 4]
            self.state = 3
            self.current_application_id = EXTERNAL_APPLICATION_ID
            return self

    clock = ClockStub()
    driver = BackgroundDriver([4])

    _lifecycle(driver, clock).background_and_resume(1)

    assert driver.lifecycle_calls == ["background:1"]
    assert clock.now == pytest.approx(0.1)


def test_background_resume_times_out_without_repeating_the_command() -> None:
    class BackgroundDriver(_DriverStub):
        def background_app(self, seconds: int) -> object:
            self.lifecycle_calls.append(f"background:{seconds}")
            self.state = 3
            self.current_application_id = EXTERNAL_APPLICATION_ID
            return self

    clock = ClockStub()
    driver = BackgroundDriver([3])

    with pytest.raises(DeviceInteractionError, match="Could not resume"):
        _lifecycle(driver, clock).background_and_resume(1)

    assert driver.lifecycle_calls == ["background:1"]
    assert clock.now == pytest.approx(0.3)


def test_external_ui_requires_a_known_foreground_identity() -> None:
    clock = ClockStub()
    driver = _DriverStub([4])
    lifecycle = _lifecycle(driver, clock)

    assert lifecycle.is_external_ui_visible() is True
    driver.current_application_id = None
    assert lifecycle.is_external_ui_visible() is None
