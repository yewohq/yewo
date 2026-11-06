"""Tests for opening and owning an Appium driver session."""

from appium.options.android import UiAutomator2Options
from appium.options.common.base import AppiumOptions
import pytest

from yewo.automation import driver_session
from yewo.automation import (
    AppiumServerLease,
    DriverClient,
    DriverFactory,
    DriverSessionError,
    DriverSessionManager,
)
from yewo.resources import ResourceOwnership


class _DriverStub:
    def __init__(self, *, session_id: str | None = "session-123") -> None:
        self.session_id = session_id
        self.capabilities: dict[str, object] = {"platformName": "Android"}
        self.quit_calls = 0
        self.quit_error: Exception | None = None
        self.installed = True
        self.removed: list[str] = []

    def quit(self) -> None:
        self.quit_calls += 1
        if self.quit_error is not None:
            error = self.quit_error
            self.quit_error = None
            raise error

    def is_app_installed(self, application_id: str) -> bool:
        del application_id
        return self.installed

    def remove_app(self, application_id: str) -> object:
        self.removed.append(application_id)
        return self


class _DriverFactoryStub:
    def __init__(self, driver: DriverClient) -> None:
        self.driver = driver
        self.calls: list[tuple[str, AppiumOptions]] = []

    def __call__(
        self,
        server_url: str,
        options: AppiumOptions,
    ) -> DriverClient:
        self.calls.append((server_url, options))
        return self.driver


def _server() -> AppiumServerLease:
    return AppiumServerLease(
        "http://127.0.0.1:4723/",
        ResourceOwnership.EXTERNAL,
    )


def test_manager_opens_and_owns_only_the_driver_session() -> None:
    driver = _DriverStub(session_id=" session-123 ")
    factory = _DriverFactoryStub(driver)
    typed_factory: DriverFactory = factory
    manager = DriverSessionManager(typed_factory)
    server = _server()
    options = UiAutomator2Options()

    lease = manager.open(server, options)
    lease.close()
    lease.close()

    assert lease.driver is driver
    assert lease.session_id == "session-123"
    assert lease.server_url == "http://127.0.0.1:4723"
    assert lease.ownership is ResourceOwnership.RUN
    assert driver.quit_calls == 1
    assert not server.closed
    assert factory.calls == [("http://127.0.0.1:4723", options)]


def test_default_driver_uses_a_bounded_appium_command_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}
    driver = _DriverStub()

    def remote(**arguments):
        captured.update(arguments)
        return driver

    monkeypatch.setattr(driver_session.webdriver, "Remote", remote)

    created = driver_session._create_driver(
        "http://127.0.0.1:4723",
        UiAutomator2Options(),
    )

    assert created is driver
    client_config = captured["client_config"]
    assert isinstance(client_config, driver_session.AppiumClientConfig)
    assert client_config.timeout == driver_session.APPIUM_COMMAND_TIMEOUT_SECONDS


def test_manager_rejects_a_closed_server_lease() -> None:
    driver = _DriverStub()
    factory = _DriverFactoryStub(driver)
    server = _server()
    server.close()

    with pytest.raises(DriverSessionError, match="closed Appium server"):
        DriverSessionManager(factory).open(server, UiAutomator2Options())

    assert factory.calls == []


def test_manager_wraps_driver_factory_failure() -> None:
    def fail(server_url, options):
        del server_url, options
        raise ConnectionError("Appium did not answer")

    with pytest.raises(DriverSessionError, match="Could not open") as raised:
        DriverSessionManager(fail).open(_server(), UiAutomator2Options())

    assert isinstance(raised.value.__cause__, ConnectionError)


def test_manager_discards_a_driver_without_a_session_id() -> None:
    driver = _DriverStub(session_id=None)

    with pytest.raises(DriverSessionError, match="without a session ID"):
        DriverSessionManager(_DriverFactoryStub(driver)).open(
            _server(),
            UiAutomator2Options(),
        )

    assert driver.quit_calls == 1


def test_failed_quit_can_be_retried() -> None:
    driver = _DriverStub()
    driver.quit_error = RuntimeError("remote quit failed")
    lease = DriverSessionManager(_DriverFactoryStub(driver)).open(
        _server(),
        UiAutomator2Options(),
    )

    with pytest.raises(DriverSessionError, match="Could not close"):
        lease.close()

    assert not lease.closed
    lease.close()
    assert lease.closed
    assert driver.quit_calls == 2
