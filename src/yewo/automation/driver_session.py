"""Open and own one Appium driver session."""

from __future__ import annotations

from contextlib import suppress
from typing import Protocol

from appium import webdriver
from appium.options.common.base import AppiumOptions

from yewo.automation.appium_server import AppiumServerLease
from yewo.automation.errors import DriverSessionError
from yewo.resources import ResourceLease, ResourceOwnership


class DriverClient(Protocol):
    """The Appium driver values required at the session boundary."""

    @property
    def session_id(self) -> str | None:
        """Return the remote session identifier."""
        ...

    def quit(self) -> None:
        """Close the remote driver session."""
        ...

    def is_app_installed(self, application_id: str, /) -> bool:
        """Return whether the target application is installed."""
        ...

    def remove_app(self, application_id: str, /) -> object:
        """Remove one installed application."""
        ...


class DriverFactory(Protocol):
    """Create one driver against a healthy Appium endpoint."""

    def __call__(
        self,
        server_url: str,
        options: AppiumOptions,
    ) -> DriverClient: ...


def _create_driver(server_url: str, options: AppiumOptions) -> DriverClient:
    return webdriver.Remote(command_executor=server_url, options=options)


class DriverSessionLease(ResourceLease):
    """A run-owned Appium driver session."""

    def __init__(
        self,
        driver: DriverClient,
        session_id: str,
        server_url: str,
    ) -> None:
        super().__init__(ResourceOwnership.RUN, lambda: _quit_driver(driver))
        self.driver = driver
        self.session_id = session_id
        self.server_url = server_url


def _quit_driver(driver: DriverClient) -> None:
    try:
        driver.quit()
    except Exception as error:  # noqa: BLE001 - external driver client boundary
        raise DriverSessionError("Could not close the Appium driver session.") from error


class DriverSessionManager:
    """Open driver sessions through an injectable client factory."""

    def __init__(self, factory: DriverFactory = _create_driver) -> None:
        self._factory = factory

    def open(
        self,
        server: AppiumServerLease,
        options: AppiumOptions,
    ) -> DriverSessionLease:
        """Open and return a driver lease without taking server ownership."""
        if server.closed:
            raise DriverSessionError("A closed Appium server lease cannot open a driver.")
        try:
            driver = self._factory(server.server_url, options)
        except Exception as error:  # noqa: BLE001 - injected driver factory boundary
            raise DriverSessionError("Could not open an Appium driver session.") from error
        session_id = getattr(driver, "session_id", None)
        if not isinstance(session_id, str) or not session_id.strip():
            with suppress(Exception):
                driver.quit()
            raise DriverSessionError("Appium returned a driver without a session ID.")
        return DriverSessionLease(
            driver,
            session_id.strip(),
            server.server_url,
        )
