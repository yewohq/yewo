"""Automation session startup and virtual pasteboard policy."""

from pathlib import Path

import pytest

from automation.test_session import (
    _DriverStub,
    _application,
    _device,
    _manager,
    _open,
    _request,
    _server,
)
from yewo.applications.records import ApplicationInstallMode
from yewo.automation.errors import AutomationSessionError
from yewo.devices.inventory import DeviceKind


def test_strict_reuse_requires_positive_presence_evidence(tmp_path: Path) -> None:
    events: list[str] = []

    with pytest.raises(AutomationSessionError, match="positive evidence"):
        _open(
            _manager(_DriverStub(events), installed_before=None),
            _device(events),
            _server(events),
            _request(ApplicationInstallMode.REUSE),
            _application().model_copy(update={"artifact_path": None}),
            tmp_path,
        )

    assert events == ["server", "device"]


def test_missing_application_after_driver_open_rolls_back_every_lease(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    with pytest.raises(AutomationSessionError, match="not installed after"):
        _open(
            _manager(_DriverStub(events, is_installed=False), installed_before=False),
            _device(events),
            _server(events),
            _request(),
            _application(),
            tmp_path,
        )

    assert events == ["clipboard:", "driver", "server", "device"]


def test_virtual_pasteboard_failure_rolls_back_every_lease(tmp_path: Path) -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    driver.clipboard_error = RuntimeError("clipboard unavailable")

    with pytest.raises(AutomationSessionError, match="virtual device pasteboard"):
        _open(
            _manager(driver, installed_before=False),
            _device(events),
            _server(events),
            _request(),
            _application(),
            tmp_path,
        )

    assert events == ["driver", "server", "device"]


def test_physical_device_pasteboard_is_not_modified(tmp_path: Path) -> None:
    events: list[str] = []
    session = _open(
        _manager(_DriverStub(events), installed_before=True),
        _device(events, kind=DeviceKind.PHYSICAL),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    session.close()

    assert all(not event.startswith("clipboard:") for event in events)
