"""The `xcrun simctl` surface, in one place.

The iOS counterpart to `AdbControl`. Every simctl argv is built here. The values a subcommand accepts then sit
beside the call that sends them, rather than at each place one is used.

simctl drives simulators only. A physical iPhone needs `devicectl`, which offers almost none of this. A
capability that reaches this class is already one a real device cannot have.

Two kinds of call, as in `AdbControl`. `run` must succeed and raises when it
does not. `ask` is a read and answers None. The distinction matters more here
than on Android, because several simctl subcommands report a refusal on stdout
and still exit 0.
"""

import logging
from typing import Optional

from yewo.errors import DeviceUnavailable
from yewo.utils.subprocess_runner import SubprocessRunError, run_command

logger = logging.getLogger(__name__)

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
"""The preferred content size categories, the last five of which are the
accessibility range. Text size is how clipping and overflow defects surface."""


class SimctlControl:
    """Named simctl subcommands against one simulator."""

    @staticmethod
    def run(*args: str, doing: str = ""):
        """One simctl subcommand. Must succeed.

        Raises:
            DeviceUnavailable: simctl failed, or could not be run.
        """
        command = ["xcrun", "simctl", *args]
        try:
            return run_command(command, check=True)
        except SubprocessRunError as exc:
            attempt = f"Failed to {doing}" if doing else "simctl failed"
            raise DeviceUnavailable(f"{attempt}: {' '.join(command)}") from exc

    @staticmethod
    def ask(*args: str) -> Optional[str]:
        """One simctl read, or None when it will not answer."""
        command = ["xcrun", "simctl", *args]
        try:
            result = run_command(command, timeout=20, log_command=False)
        except (SubprocessRunError, OSError) as exc:
            logger.debug("Could not read from simctl (%s): %s", " ".join(args), exc)
            return None
        output = (result.stdout or "").strip()
        return output or None

    @staticmethod
    def set_location(udid: str, latitude: float, longitude: float) -> None:
        SimctlControl.run(
            "location", udid, "set", f"{latitude},{longitude}",
            doing="set the simulated location",
        )

    @staticmethod
    def set_appearance(udid: str, appearance: str) -> None:
        SimctlControl.run(
            "ui", udid, "appearance", appearance, doing="set the appearance"
        )

    @staticmethod
    def appearance(udid: str) -> Optional[str]:
        answer = SimctlControl.ask("ui", udid, "appearance")
        return answer if answer in APPEARANCES else None

    @staticmethod
    def set_content_size(udid: str, size: str) -> None:
        SimctlControl.run(
            "ui", udid, "content_size", size, doing="set the content size"
        )

    @staticmethod
    def content_size(udid: str) -> Optional[str]:
        answer = SimctlControl.ask("ui", udid, "content_size")
        return answer if answer in CONTENT_SIZES else None

    @staticmethod
    def set_privacy(udid: str, action: str, service: str, bundle_id: str) -> None:
        """Grants, revokes or resets one privacy service.

        An unknown service prints "Operation not permitted" and still exits 0,
        so only a service name the caller already matched reaches this.
        """
        SimctlControl.run(
            "privacy", udid, action, service, bundle_id,
            doing=f"{action} the {service} permission",
        )

    @staticmethod
    def open_url(udid: str, url: str) -> None:
        SimctlControl.run("openurl", udid, url, doing="open a URL")

    @staticmethod
    def terminate(udid: str, bundle_id: str) -> None:
        SimctlControl.run("terminate", udid, bundle_id, doing="terminate the app")

    @staticmethod
    def shutdown(udid: str) -> None:
        SimctlControl.run("shutdown", udid, doing="shut down the simulator")

    @staticmethod
    def add_media(udid: str, local_path: str) -> None:
        SimctlControl.run("addmedia", udid, local_path, doing="add media")

    @staticmethod
    def notify_low_memory(udid: str) -> None:
        SimctlControl.run(
            "notify_post", udid,
            "com.apple.CoreSimulator.SimDevice.MemoryNotification",
            doing="post a memory warning",
        )

    @staticmethod
    def push_notification(udid: str, bundle_id: str, payload_path: str) -> None:
        SimctlControl.run(
            "push", udid, bundle_id, payload_path,
            doing="send a push notification",
        )

    @staticmethod
    def is_simulator(udid: str) -> Optional[bool]:
        """Whether this udid names a simulator, or None when simctl cannot say.

        simctl enumerates simulators and nothing else, so a udid it does not
        list is a physical device. This is the iOS counterpart of reading an AVD name off an Android
        serial. It decides which capabilities a target can have at all,
        and almost none of them reach a real iPhone.
        """
        listed = SimctlControl.ask("list", "devices")
        if listed is None:
            return None
        return udid in listed
