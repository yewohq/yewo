"""Tests for Appium server acquisition and ownership."""

from collections.abc import Mapping, Sequence
from collections import deque
from io import BytesIO
from pathlib import Path
import subprocess

import pytest

from yewo.automation import (
    AppiumRuntime,
    AppiumServerConfig,
    AppiumServerError,
    AppiumServerManager,
    AppiumStartupTimeoutError,
)
from yewo.automation import appium_server
from yewo.resources import ResourceOwnership


class _StatusProbeStub:
    def __init__(self, answers: Sequence[bool]) -> None:
        self._answers = deque(answers)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, status_url: str, *, timeout_seconds: float) -> bool:
        self.calls.append((status_url, timeout_seconds))
        answer = self._answers[0]
        if len(self._answers) > 1:
            answer = self._answers.popleft()
        return answer


class _ProcessStub:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.pid = 4321

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _ProcessStarterStub:
    def __init__(self, process: _ProcessStub) -> None:
        self.process = process
        self.calls: list[tuple[tuple[str, ...], Mapping[str, str]]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> _ProcessStub:
        self.calls.append((tuple(command), environment))
        return self.process


class _ClockStub:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _runtime(tmp_path: Path) -> AppiumRuntime:
    return AppiumRuntime(
        node_executable=tmp_path / "node",
        appium_script=tmp_path / "appium.js",
        appium_version="3.1.0",
        environment={"PATH": "/runtime/bin"},
    )


def test_manager_attaches_to_a_ready_external_server(tmp_path: Path) -> None:
    probe = _StatusProbeStub((True,))
    starter = _ProcessStarterStub(_ProcessStub())
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=starter,
        status_probe=probe,
    )

    lease = manager.acquire(AppiumServerConfig())
    lease.close()

    assert lease.ownership is ResourceOwnership.EXTERNAL
    assert lease.log_path is None
    assert starter.calls == []
    assert probe.calls == [("http://127.0.0.1:4723/status", 0.75)]


def test_manager_starts_waits_for_and_stops_a_local_server(tmp_path: Path) -> None:
    probe = _StatusProbeStub((False, False, True))
    process = _ProcessStub()
    starter = _ProcessStarterStub(process)
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=starter,
        status_probe=probe,
        clock=_ClockStub(),
    )
    config = AppiumServerConfig(
        log_path=tmp_path / "logs" / "appium.log",
        server_url="http://localhost:4725/wd/hub",
        allow_insecure_features=("uiautomator2:emulator_console",),
    )

    lease = manager.acquire(config)
    lease.close()
    lease.close()

    command, environment = starter.calls[0]
    assert command == (
        str(tmp_path / "node"),
        str(tmp_path / "appium.js"),
        "--address",
        "localhost",
        "--port",
        "4725",
        "--log",
        str(config.log_path),
        "--log-level",
        "warn",
        "--log-no-colors",
        "--base-path",
        "/wd/hub",
        "--allow-insecure",
        "uiautomator2:emulator_console",
    )
    assert environment == {"PATH": "/runtime/bin"}
    assert config.log_path is not None
    assert config.log_path.parent.is_dir()
    assert lease.ownership is ResourceOwnership.RUN
    assert lease.log_path == config.log_path
    assert process.terminated


def test_manager_reports_early_process_exit(tmp_path: Path) -> None:
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=_ProcessStarterStub(_ProcessStub(returncode=1)),
        status_probe=_StatusProbeStub((False, False)),
        clock=_ClockStub(),
    )

    with pytest.raises(AppiumServerError, match="stopped during startup"):
        manager.acquire(AppiumServerConfig(log_path=tmp_path / "appium.log"))


def test_manager_timeout_stops_its_process(tmp_path: Path) -> None:
    process = _ProcessStub()
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=_ProcessStarterStub(process),
        status_probe=_StatusProbeStub((False,)),
        clock=_ClockStub(),
    )

    with pytest.raises(AppiumStartupTimeoutError, match="within 1 seconds"):
        manager.acquire(
            AppiumServerConfig(
                log_path=tmp_path / "appium.log",
                startup_timeout_seconds=1,
                poll_interval_seconds=0.5,
            )
        )

    assert process.terminated


def test_manager_does_not_start_a_remote_server(tmp_path: Path) -> None:
    starter = _ProcessStarterStub(_ProcessStub())
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=starter,
        status_probe=_StatusProbeStub((False,)),
    )

    with pytest.raises(AppiumServerError, match="only on a local HTTP endpoint"):
        manager.acquire(
            AppiumServerConfig(
                log_path=tmp_path / "appium.log",
                server_url="https://appium.example.test:4723",
            )
        )

    assert starter.calls == []


def test_manager_can_require_an_existing_server(tmp_path: Path) -> None:
    manager = AppiumServerManager(status_probe=_StatusProbeStub((False,)))

    with pytest.raises(AppiumServerError, match="No ready Appium server"):
        manager.acquire(
            AppiumServerConfig(
                start_if_unavailable=False,
            )
        )


def test_manager_requires_a_log_path_before_starting() -> None:
    manager = AppiumServerManager(status_probe=_StatusProbeStub((False,)))

    with pytest.raises(AppiumServerError, match="log path is required"):
        manager.acquire(AppiumServerConfig())


class _StatusResponse(BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'{"value":{"ready":true}}', True),
        (b'{"value":{"ready":false}}', False),
        (b'{"status":0}', False),
    ],
)
def test_status_probe_requires_explicit_readiness(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        appium_server,
        "urlopen",
        lambda *args, **kwargs: _StatusResponse(payload),
    )

    assert (
        appium_server._probe_status(
            "http://127.0.0.1:4723/status",
            timeout_seconds=0.5,
        )
        is expected
    )


def test_process_stop_escalates_when_terminate_hangs(tmp_path: Path) -> None:
    class StubbornProcess(_ProcessStub):
        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("appium", timeout or 0)
            return -9

    process = StubbornProcess()
    manager = AppiumServerManager(
        runtime_resolver=lambda: _runtime(tmp_path),
        process_starter=_ProcessStarterStub(process),
        status_probe=_StatusProbeStub((False, True)),
        clock=_ClockStub(),
    )

    lease = manager.acquire(AppiumServerConfig(log_path=tmp_path / "appium.log"))
    lease.close()

    assert process.terminated
    assert process.killed
