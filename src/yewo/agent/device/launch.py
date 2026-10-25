"""Waiting for a freshly launched app to settle on its first real screen.

Launching is not instant on any toolkit. The window exists before anything is
drawn into it, and many apps then show a splash before the first real screen.
A screenshot of either hands the planner a screen the app is not on. An empty
one cannot be told from a crash, and a splash gets interacted with.

Nothing here inspects the UI toolkit. A launch is over when the screen has
something on it and is no longer changing wholesale, however it was drawn.
Some screens animate forever, such as spinners, carousels and video, and never
go completely still. So content that has held for a while also counts.
"""

import io
import logging
import time
from enum import Enum
from typing import Optional, Tuple

import imagehash
from PIL import Image, ImageStat

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.platform import Platform

from yewo.agent.perception.screen import NAVIGATION_THRESHOLD
from yewo.errors import DeviceUnavailable, ObservationUnavailable
from yewo.defaults import (
    BLANK_SCREEN_STDDEV,
    LAUNCH_CONTENT_GRACE,
    LAUNCH_PAINT_TIMEOUT,
    LAUNCH_POLL_INTERVAL,
    LAUNCH_STABLE_READINGS,
)

logger = logging.getLogger(__name__)


def _sample(device_control) -> Tuple[Optional[bool], Optional[object]]:
    """Takes one screenshot and reads it twice: is it empty, and what is on it.

    Only the body between the status and navigation bars counts. The OS draws
    those, and they are there whether or not the app has painted.
    """
    try:
        img = Image.open(io.BytesIO(device_control.get_screenshot())).convert("L")
    except (ObservationUnavailable, OSError):  # PIL raises OSError on bad bytes
        return None, None
    width, height = img.size
    body = img.crop((0, int(height * 0.08), width, int(height * 0.92)))
    blank = ImageStat.Stat(body).stddev[0] < BLANK_SCREEN_STDDEV
    return blank, imagehash.phash(body, hash_size=16)


def screen_is_blank(device_control) -> Optional[bool]:
    """Reports whether the screen body carries no detail, or None if unreadable."""
    return _sample(device_control)[0]


def await_first_paint(device_control) -> None:
    """Polls until the app is showing the screen it launched into.

    Two ways out, because two things look like an unfinished launch. A screen
    that stops changing is done. A screen that keeps changing but has held
    content for LAUNCH_CONTENT_GRACE is also done. Without the second, any app
    whose first screen animates would cost the full timeout every scenario.

    Only wholesale change counts as unfinished. A spinner or a carousel moves
    part of the screen. A splash giving way to the first screen replaces all of
    it. Perception draws the same distinction for navigation.
    """
    deadline = time.monotonic() + LAUNCH_PAINT_TIMEOUT
    previous = None
    stable = 0
    content_since = None
    while time.monotonic() < deadline:
        blank, phash = _sample(device_control)
        if blank is None:
            return  # Unreadable screens are the caller's problem, not a reason to wait.

        if blank:
            previous, stable, content_since = None, 0, None
        else:
            content_since = content_since or time.monotonic()
            settled = previous is not None and (phash - previous) < NAVIGATION_THRESHOLD
            stable = stable + 1 if settled else 1
            previous = phash
            if stable >= LAUNCH_STABLE_READINGS:
                return
            if time.monotonic() - content_since >= LAUNCH_CONTENT_GRACE:
                logger.info("Launch screen is still animating — proceeding.")
                return
        time.sleep(LAUNCH_POLL_INTERVAL)
    logger.warning(
        "App had not settled %.1fs after launch — continuing anyway.",
        LAUNCH_PAINT_TIMEOUT,
    )


def _clear_task_restart(device_control, package_id: str) -> bool:
    """Relaunches an Android app with a clean back stack. False if unavailable."""
    adb_runner = getattr(device_control, "adb_control", None) or AdbControl
    serial = getattr(device_control, "serial", None) or adb_runner.resolve_emulator_serial(
        device_control.device_name
    )
    try:
        activity = adb_runner.resolve_launch_activity(
            device_control.app_path, package_id, serial
        )
    except DeviceUnavailable as apk_err:
        logger.warning("Failed to get APK activity: %s", apk_err)
        return False
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(
        ["shell", "am", "start", "-n", f"{package_id}/{activity}", "--activity-clear-task"]
    )
    try:
        adb_runner.run_adb_command(cmd)
        return True
    except DeviceUnavailable as launch_err:
        logger.warning("ADB clear-task failed: %s. Falling back.", launch_err)
        return False


class SetupMode(str, Enum):
    """The state a scenario needs before its first step.

    Required state is part of what a test is. Stated in a prompt instead, it
    is an assumption, and a prompt enforces nothing.
    """

    CLEAR_DATA = "clear_data"
    """Wipe the app's data. The scenario then sees a first install."""

    RELAUNCH = "relaunch"
    """Restart the app and keep its data, including any session."""

    PRESERVE = "preserve"
    """Leave the app as the previous scenario left it."""

    def describe(self) -> str:
        """What the planner is told about the state it starts in.

        States what was done, not what will be on screen. A named screen is a
        claim about one app, and a wrong one tells the planner to disbelieve
        the screen in front of it.
        """
        return {
            SetupMode.CLEAR_DATA: (
                "The app's data was cleared before this scenario, so it is in the "
                "state a first install would be in. Any session, permission, or "
                "saved app data from earlier scenarios is gone."
            ),
            SetupMode.RELAUNCH: (
                "The app was restarted before this scenario. Its stored data is "
                "unchanged, so a session or saved app data from an earlier scenario "
                "may still be in place."
            ),
            SetupMode.PRESERVE: (
                "The app was left exactly as the previous scenario left it. It was "
                "not restarted."
            ),
        }[self]


def clear_app_data(device_control) -> bool:
    """Wipes the app's stored data so it opens as a first install would.

    Android only. Returns whether the data went. A caller that asked for a
    cleared app and did not get one is testing a different scenario from the
    one it described. So it needs to know.
    """
    if device_control.platform != Platform.ANDROID:
        logger.warning(
            "Clearing app data is an Android operation. Leaving state as it is."
        )
        return False
    try:
        AdbControl.clear_app_data(device_control.serial, device_control.package_id)
        return True
    except (DeviceUnavailable, OSError) as exc:
        logger.warning("Could not clear app data: %s", exc)
        return False


def restart_app(device_control) -> None:
    """Restarts the app under test and waits for it to draw."""
    try:
        if not hasattr(device_control, "package_id"):
            return
        package_id = device_control.package_id
        logger.info("Restarting application: %s", package_id)

        if device_control.platform == Platform.ANDROID and _clear_task_restart(
            device_control, package_id
        ):
            await_first_paint(device_control)
            return

        device_control.driver.terminate_app(package_id)
        time.sleep(1.5)
        device_control.driver.activate_app(package_id)
        await_first_paint(device_control)
    except Exception as exc:
        raise DeviceUnavailable(f"Failed to restart app: {exc}") from exc
