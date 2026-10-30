"""The simctl command surface used by iOS simulator components."""

from __future__ import annotations

import json

from yewo.devices._commands import (
    CommandRunner,
    capture_optional,
    execute_command,
    run_command,
)


APPEARANCES = ("light", "dark")
CONTENT_SIZES = (
    "extra-small",
    "small",
    "medium",
    "large",
    "extra-large",
    "extra-extra-large",
    "extra-extra-extra-large",
    "accessibility-medium",
    "accessibility-large",
    "accessibility-extra-large",
    "accessibility-extra-extra-large",
    "accessibility-extra-extra-extra-large",
)


class SimctlClient:
    """Build and execute all simctl commands in one boundary."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        xcrun_executable: str = "xcrun",
    ) -> None:
        self._runner = runner
        self._xcrun_executable = xcrun_executable

    def run(
        self,
        *args: str,
        purpose: str = "run a simulator command",
        timeout_seconds: float = 30,
    ):
        """Run one required simctl command."""
        return execute_command(
            self._runner,
            [self._xcrun_executable, "simctl", *args],
            timeout_seconds=timeout_seconds,
            purpose=purpose,
        )

    def ask(self, *args: str, timeout_seconds: float = 20) -> str | None:
        """Read from one simctl command, or return no answer."""
        return capture_optional(
            self._runner,
            [self._xcrun_executable, "simctl", *args],
            timeout_seconds=timeout_seconds,
        )

    def device_listing(self) -> str:
        """Return the required JSON simulator inventory."""
        return (
            self.run(
                "list",
                "devices",
                "available",
                "--json",
                purpose="list iOS simulators",
            ).stdout
            or ""
        )

    def boot(self, connection_id: str) -> None:
        """Start one simulator without opening the Simulator application."""
        self.run("boot", connection_id, purpose="start the simulator")

    def wait_until_booted(self, connection_id: str, *, timeout_seconds: float = 120) -> None:
        """Wait until CoreSimulator reports that startup completed."""
        self.run(
            "bootstatus",
            connection_id,
            "-b",
            purpose="wait for the simulator to start",
            timeout_seconds=timeout_seconds,
        )

    def set_location(self, connection_id: str, latitude: float, longitude: float) -> None:
        self.run(
            "location",
            connection_id,
            "set",
            f"{latitude},{longitude}",
            purpose="set the simulated location",
        )

    def set_appearance(self, connection_id: str, appearance: str) -> None:
        self.run("ui", connection_id, "appearance", appearance, purpose="set appearance")

    def appearance(self, connection_id: str) -> str | None:
        answer = self.ask("ui", connection_id, "appearance")
        return answer if answer in APPEARANCES else None

    def set_content_size(self, connection_id: str, size: str) -> None:
        self.run("ui", connection_id, "content_size", size, purpose="set content size")

    def content_size(self, connection_id: str) -> str | None:
        answer = self.ask("ui", connection_id, "content_size")
        return answer if answer in CONTENT_SIZES else None

    def set_privacy(self, connection_id: str, action: str, service: str, bundle_id: str) -> None:
        self.run(
            "privacy",
            connection_id,
            action,
            service,
            bundle_id,
            purpose=f"{action} a privacy permission",
        )

    def open_url(self, connection_id: str, url: str) -> None:
        self.run("openurl", connection_id, url, purpose="open a URL")

    def terminate_app(self, connection_id: str, bundle_id: str) -> None:
        self.run("terminate", connection_id, bundle_id, purpose="terminate the application")

    def shutdown(self, connection_id: str) -> None:
        self.run("shutdown", connection_id, purpose="shut down the simulator")

    def add_media(self, connection_id: str, local_path: str) -> None:
        self.run("addmedia", connection_id, local_path, purpose="add media")

    def notify_low_memory(self, connection_id: str) -> None:
        self.run(
            "notify_post",
            connection_id,
            "com.apple.CoreSimulator.SimDevice.MemoryNotification",
            purpose="post a memory warning",
        )

    def push_notification(self, connection_id: str, bundle_id: str, payload_path: str) -> None:
        self.run(
            "push",
            connection_id,
            bundle_id,
            payload_path,
            purpose="send a push notification",
        )

    def is_simulator(self, connection_id: str) -> bool | None:
        """Return whether simctl lists the device ID."""
        listing = self.ask("list", "devices")
        return None if listing is None else connection_id in listing

    def installed_applications(self, connection_id: str) -> set[str] | None:
        """Return installed simulator bundle IDs, or no answer."""
        output = self.ask("listapps", connection_id, "--json")
        if output is None:
            return None
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        identifiers = {key for key in payload if isinstance(key, str) and key}
        for value in payload.values():
            if isinstance(value, dict):
                identifier = value.get("CFBundleIdentifier")
                if isinstance(identifier, str) and identifier:
                    identifiers.add(identifier)
        return identifiers
