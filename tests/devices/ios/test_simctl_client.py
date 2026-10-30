"""Tests for the ported simctl command client."""

import pytest

from devices._runner import CommandRunnerStub
from yewo.devices.errors import DeviceCommandError
from yewo.devices.ios import SimctlClient


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (lambda client: client.boot("UDID"), ("boot", "UDID")),
        (
            lambda client: client.set_location("UDID", 51.5, -0.1),
            ("location", "UDID", "set", "51.5,-0.1"),
        ),
        (lambda client: client.open_url("UDID", "yewo://test"), ("openurl", "UDID", "yewo://test")),
        (
            lambda client: client.set_appearance("UDID", "dark"),
            ("ui", "UDID", "appearance", "dark"),
        ),
        (
            lambda client: client.set_content_size("UDID", "large"),
            ("ui", "UDID", "content_size", "large"),
        ),
        (
            lambda client: client.set_privacy("UDID", "grant", "camera", "com.example.app"),
            ("privacy", "UDID", "grant", "camera", "com.example.app"),
        ),
        (
            lambda client: client.terminate_app("UDID", "com.example.app"),
            ("terminate", "UDID", "com.example.app"),
        ),
        (lambda client: client.shutdown("UDID"), ("shutdown", "UDID")),
        (lambda client: client.add_media("UDID", "photo.png"), ("addmedia", "UDID", "photo.png")),
        (
            lambda client: client.notify_low_memory("UDID"),
            ("notify_post", "UDID", "com.apple.CoreSimulator.SimDevice.MemoryNotification"),
        ),
        (
            lambda client: client.push_notification("UDID", "com.example.app", "payload.apns"),
            ("push", "UDID", "com.example.app", "payload.apns"),
        ),
    ],
)
def test_simctl_operation_builds_one_platform_command(operation, arguments) -> None:
    runner = CommandRunnerStub()

    operation(SimctlClient(runner))

    assert runner.calls == [(("xcrun", "simctl", *arguments), 30)]


def test_simctl_failure_preserves_operation() -> None:
    runner = CommandRunnerStub(stderr="Unable to boot device", returncode=1)

    with pytest.raises(DeviceCommandError, match="shut down the simulator"):
        SimctlClient(runner).shutdown("UDID")


def test_boot_wait_uses_the_startup_timeout() -> None:
    runner = CommandRunnerStub()

    SimctlClient(runner).wait_until_booted("UDID")

    assert runner.calls == [(("xcrun", "simctl", "bootstatus", "UDID", "-b"), 120)]


def test_simulator_probe_returns_no_answer_when_simctl_fails() -> None:
    runner = CommandRunnerStub(returncode=1)

    assert SimctlClient(runner).is_simulator("UDID") is None


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (lambda client: client.appearance("UDID"), ("ui", "UDID", "appearance")),
        (lambda client: client.content_size("UDID"), ("ui", "UDID", "content_size")),
        (lambda client: client.is_simulator("UDID"), ("list", "devices")),
        (
            lambda client: client.installed_applications("UDID"),
            ("listapps", "UDID", "--json"),
        ),
    ],
)
def test_simctl_read_builds_one_platform_command(operation, arguments) -> None:
    runner = CommandRunnerStub(stdout="{}")

    operation(SimctlClient(runner))

    assert runner.calls == [(("xcrun", "simctl", *arguments), 20)]
