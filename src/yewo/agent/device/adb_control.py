import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from yewo.agent.device import manifest
from yewo.agent.device.manifest import LAUNCHER_CATEGORY
from yewo.errors import DeviceUnavailable
from yewo.utils.subprocess_runner import run_command, SubprocessRunError

logger = logging.getLogger(__name__)


class AdbControl:
    """Provides shell interfaces and wrapper utilities for working with Android devices via ADB.

    Three ways to reach a device, and the choice between them is the whole
    contract. `adb` and `shell` must succeed and raise when they do not. `ask`
    is a read, and answers None when the device will not say: a question adb
    cannot answer is not the same as a command that failed.

    Every one takes the serial and builds the argv, because a call site that
    forgets `-s` does not fail. adb picks the only attached device, which is
    correct until a run has two.
    """

    @staticmethod
    def adb(serial: str, *args: str, doing: str = ""):
        """One adb command against one device. Must succeed."""
        return AdbControl.run_adb_command(["adb", "-s", serial, *args], doing=doing)

    @staticmethod
    def shell(serial: str, *args: str, doing: str = ""):
        """One adb shell command against one device. Must succeed."""
        return AdbControl.adb(serial, "shell", *args, doing=doing)

    @staticmethod
    def _capture(command: list[str], timeout: float = 20) -> Optional[str]:
        """stdout from one command, or None when it will not run.

        Reports nothing rather than raising. Callers use this where a failure
        is a legitimate answer, so a raise here would turn "the device says no"
        into "the run is over".
        """
        try:
            result = run_command(command, timeout=timeout, log_command=False)
        except (SubprocessRunError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("Could not run %s: %s", " ".join(command), exc)
            return None
        output = (result.stdout or "").strip()
        return output or None

    @staticmethod
    def ask(serial: str, *args: str, timeout: float = 20) -> Optional[str]:
        """One shell read from one device, or None when it will not answer."""
        return AdbControl._capture(["adb", "-s", serial, "shell", *args], timeout)

    @staticmethod
    def ask_adb(serial: str, *args: str, timeout: float = 20) -> Optional[str]:
        """One non-shell adb read, such as an emulator console query."""
        return AdbControl._capture(["adb", "-s", serial, *args], timeout)

    @staticmethod
    def run_adb_command(command, doing: str = ""):
        """Runs one adb command, or reports what the device could not do.

        Args:
            command: A list of command-line arguments to execute.
            doing: What the command was for. The failure message includes it,
                because the argv does not say why Yewo ran the command.

        Returns:
            The CompletedProcess result from the run command.

        Raises:
            DeviceUnavailable: If the command failed, or Yewo could not run it.
        """
        try:
            return run_command(command, check=True)
        except SubprocessRunError as exc:
            attempt = f"Failed to {doing}" if doing else "Failed to run adb command"
            raise DeviceUnavailable(f"{attempt}: {' '.join(command)}") from exc

    @staticmethod
    def find_build_tool(name: str):
        """Finds a build-tools binary in the local Android SDK installation directory.

        Args:
            name: The executable to look for, such as "aapt" or "aapt2".

        Returns:
            The string path to the resolved executable.
        """
        try:
            ANDROID_HOME = os.getenv("ANDROID_HOME")
            if ANDROID_HOME is None:
                raise DeviceUnavailable("ANDROID_HOME is not set")
            command = ["find", f"{ANDROID_HOME}/build-tools", "-name", name]
            result = run_command(command, check=True, capture_output=True, text=True)
            if not result.stdout.strip():
                raise DeviceUnavailable(f"No {name} under {ANDROID_HOME}/build-tools")
            return result.stdout.split("\n")[0]
        except SubprocessRunError as exc:
            raise DeviceUnavailable(f"Failed to find {name}: {exc.stderr}") from exc

    @staticmethod
    def find_aapt():
        """Finds the path to the aapt binary in the local Android SDK installation directory.

        Returns:
            The string path to the resolved aapt executable.
        """
        return AdbControl.find_build_tool("aapt")

    @staticmethod
    def get_apk_info(apk_path):
        """Inspects badging info inside a built APK file to get its Package ID and launchable activity.

        Args:
            apk_path: The filesystem path to the target APK.

        Returns:
            A tuple of (package_id, launchable_activity_name) parsed from the badging details.
        """
        try:
            aapt = AdbControl.find_aapt()
            logger.info("Using aapt at: %s", aapt)
            result = run_command(
                [aapt, "dump", "badging", apk_path], check=True, capture_output=True, text=True
            )
            stdout = result.stdout
            package_match = re.search(r"package: name='([^']+)'", stdout)
            activity_match = re.search(r"launchable-activity: name='([^']+)'", stdout)
            package_id = package_match.group(1) if package_match else None
            activity = activity_match.group(1) if activity_match else None
            return package_id, activity
        except SubprocessRunError as exc:
            raise DeviceUnavailable(f"Failed to read {apk_path}: {exc.stderr}") from exc

    @staticmethod
    def dump_manifest(apk_path: str) -> Optional[str]:
        """The APK's manifest as an aapt2 xmltree dump, or None when unreadable."""
        try:
            aapt2 = AdbControl.find_build_tool("aapt2")
            result = run_command(
                [aapt2, "dump", "xmltree", "--file", "AndroidManifest.xml", apk_path],
                check=True,
                capture_output=True,
                text=True,
                log_command=False,
            )
            return result.stdout
        except (SubprocessRunError, DeviceUnavailable) as exc:
            logger.debug("Could not dump the manifest of %s: %s", apk_path, exc)
            return None

    @staticmethod
    def launch_activity_on_device(serial: str, package_id: str) -> Optional[str]:
        """What the launcher would start for a package, or None if it cannot say.

        This is the only source that accounts for aliases the app enabled or
        disabled at runtime, so it outranks anything read from the APK. It
        needs the package installed.
        """
        resolved = AdbControl.ask(
            serial, "cmd", "package", "resolve-activity", "--brief",
            "-c", LAUNCHER_CATEGORY, package_id,
        )
        lines = [line.strip() for line in (resolved or "").splitlines() if line.strip()]
        if not lines:
            return None
        component = lines[-1]
        if "/" not in component:
            return None
        package, _, activity = component.partition("/")
        if package != package_id or not activity:
            return None
        return f"{package_id}{activity}" if activity.startswith(".") else activity

    @staticmethod
    def resolve_launch_activity(
        apk_path: str, package_id: str, serial: Optional[str] = None
    ) -> str:
        """The activity to name in a launch intent.

        An intent without a component matches every launcher on the device,
        which Android refuses rather than guessing. So a missing activity has
        to stop the run here, where the APK is named, instead of surfacing as
        an adb error about intents.

        Raises:
            DeviceUnavailable: If no source could name one activity.
        """
        if serial:
            on_device = AdbControl.launch_activity_on_device(serial, package_id)
            if on_device:
                return on_device

        xmltree = AdbControl.dump_manifest(apk_path)
        if xmltree:
            declared = manifest.enabled_launcher(xmltree, package_id)
            if declared:
                return declared

        _, badged = AdbControl.get_apk_info(apk_path)
        if badged:
            return badged

        raise DeviceUnavailable(
            f"Could not find the launcher activity of {package_id} in {apk_path}. "
            "The device, the manifest and aapt badging all failed to name one."
        )

    @staticmethod
    def attached_devices() -> dict[str, str]:
        """Every device adb can see, and the state it reports for each.

        `resolve_emulator_serial` only matches a device in the "device" state,
        so an emulator that has gone offline looks exactly like no emulator at
        all. Reading the states apart from that tells the two cases apart.
        """
        listed = AdbControl._capture(["adb", "devices"])
        states = {}
        for line in (listed or "").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        return states

    @staticmethod
    def list_avds() -> Optional[list[str]]:
        """Every emulator Yewo could start, or None when it cannot ask.

        None and an empty list mean different things: one is "the question
        could not be answered", the other is "there are none".
        """
        emulator = shutil.which("emulator")
        if emulator is None:
            home = os.getenv("ANDROID_HOME")
            candidate = Path(home) / "emulator" / "emulator" if home else None
            if candidate is None or not candidate.is_file():
                return None
            emulator = str(candidate)
        try:
            result = run_command([emulator, "-list-avds"], timeout=30, log_command=False)
        except SubprocessRunError as exc:
            logger.debug("Could not list AVDs: %s", exc)
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    @staticmethod
    def boot_completed(serial: str) -> bool:
        """Whether Android has finished booting on a device."""
        return AdbControl.ask(serial, "getprop", "sys.boot_completed") == "1"

    @staticmethod
    def uninstall_app(serial: str, package_id: str) -> bool:
        """Removes one package, reporting whether it is gone.

        Uses adb rather than the driver, because the sessions worth cleaning
        up after are usually the ones that died with the driver.
        """
        try:
            run_command(["adb", "-s", serial, "uninstall", package_id], timeout=60)
        except SubprocessRunError as exc:
            logger.warning("Could not uninstall %s: %s", package_id, exc)
            return False
        logger.info("Uninstalled %s from %s", package_id, serial)
        return True

    @staticmethod
    def toggle_tap_highlights(serial: str, enabled: bool):
        """Enables or disables visual touch indicators/highlights on the target screen.

        Args:
            serial: The target device serial number.
            enabled: True to show touch highlights, False to turn them off.
        """
        val = "1" if enabled else "0"
        AdbControl.shell(
            serial, "settings", "put", "system", "show_touches", val,
            doing="toggle tap highlights",
        )

    @staticmethod
    def toggle_pointer_location(serial: str, enabled: bool):
        """Enables or disables pointer coordinates tracing overlay at the top of the screen.

        Args:
            serial: The target device serial number.
            enabled: True to show trace details, False to turn them off.
        """
        val = "1" if enabled else "0"
        AdbControl.shell(
            serial, "settings", "put", "system", "pointer_location", val,
            doing="toggle pointer location",
        )
        logger.info("Pointer location %s", 'enabled' if enabled else 'disabled')

    @staticmethod
    def enable_feedback(serial: str, pointer_location: bool, tap_highlights: bool):
        """Configures device gesture feedback preferences in Developer Settings.

        Args:
            serial: The target device serial number.
            pointer_location: True to show pointer coordinates, False otherwise.
            tap_highlights: True to show circle touches, False otherwise.
        """
        AdbControl.toggle_pointer_location(serial, pointer_location)
        AdbControl.toggle_tap_highlights(serial, tap_highlights)

    @staticmethod
    def disable_feedback(serial: str):
        """Disables all system pointer tracking and touch highlights on the device."""
        AdbControl.toggle_pointer_location(serial, False)
        AdbControl.toggle_tap_highlights(serial, False)

    @staticmethod
    def avd_name(serial: str) -> Optional[str]:
        """The AVD a running emulator was started from.

        'adb emu avd name' needs the emulator console auth token and prints
        nothing without it, so read the boot property instead.
        """
        answers = (
            AdbControl.ask(serial, "getprop", "ro.boot.qemu.avd_name", timeout=10),
            AdbControl.ask_adb(serial, "emu", "avd", "name", timeout=10),
        )
        for answer in answers:
            first = (answer or "").splitlines()
            if first and first[0].strip():
                return first[0].strip()
        return None

    @staticmethod
    def installed_packages(serial: str) -> set[str]:
        """Package names currently installed on a device, empty when adb cannot answer."""
        listed = AdbControl.ask(serial, "pm", "list", "packages")
        return {
            line.strip().removeprefix("package:")
            for line in (listed or "").splitlines()
            if line.strip()
        }

    @staticmethod
    def resolve_emulator_serial(device_name: str) -> str | None:
        """Resolves an AVD profile name or serial to an active emulator serial number."""
        listed = AdbControl._capture(["adb", "devices"])
        for serial in re.findall(r"^([^\s]+)\s+device", listed or "", re.MULTILINE):
            if serial == device_name:
                return serial
            # An emulator serial says nothing about which AVD it is running.
            if serial.startswith("emulator-") and AdbControl.avd_name(serial) == device_name:
                return serial
        return None

    @staticmethod
    def kill_emulator(device_name: str, serial: str = None):
        """Stops a running emulator matching the device name (AVD profile name or serial)."""
        target_serial = serial or AdbControl.resolve_emulator_serial(device_name)
        if target_serial:
            logger.info("Found matching emulator serial '%s' for '%s'. Terminating.", target_serial, device_name)
            AdbControl.adb(target_serial, "emu", "kill", doing="kill emulator")
        else:
            logger.warning("No active emulator serial found matching '%s'. Skipping termination.", device_name)

    @staticmethod
    def mock_location(serial: str, lat: float, lng: float):
        """Mocks GPS location coordinates on the emulator."""
        AdbControl.adb(
            serial, "emu", "geo", "fix", str(lng), str(lat), doing="mock location"
        )

    @staticmethod
    def set_airplane_mode(serial: str, enabled: bool):
        """Enables or disables airplane mode via connectivity service."""
        action = "enable" if enabled else "disable"
        AdbControl.shell(
            serial, "cmd", "connectivity", "airplane-mode", action,
            doing="set airplane mode",
        )

    @staticmethod
    def set_battery_status(serial: str, level: int, charging: bool):
        """Overrides battery level and AC power charging status."""
        ac_val = "1" if charging else "0"
        AdbControl.shell(
            serial, "dumpsys", "battery", "set", "level", str(level),
            doing="set battery status",
        )
        AdbControl.shell(
            serial, "dumpsys", "battery", "set", "ac", ac_val,
            doing="set battery status",
        )

    @staticmethod
    def reset_battery(serial: str):
        """Hands the battery back to the real one after an override.

        There is no 'set it to what it was': the framework either reports the
        hardware or reports the override, so reset is the only honest undo.
        """
        AdbControl.shell(serial, "dumpsys", "battery", "reset", doing="reset battery")

    @staticmethod
    def set_timezone(serial: str, timezone: str):
        """Sets the system timezone via settings global service."""
        AdbControl.shell(
            serial, "settings", "put", "global", "timezone", timezone,
            doing="set timezone",
        )

    @staticmethod
    def set_app_locale(serial: str, package_id: str, locale_str: str):
        """Configures language locale settings for the target application package."""
        AdbControl.shell(
            serial, "cmd", "locale", "set-app-locales", package_id,
            "--locales", locale_str, doing="set app locale",
        )

    @staticmethod
    def set_app_permission(serial: str, package_id: str, permission: str, grant: bool):
        """Grants or revokes a specific system permission for the target package."""
        action_cmd = "grant" if grant else "revoke"
        AdbControl.shell(
            serial, "pm", action_cmd, package_id, permission,
            doing="set app permission",
        )

    @staticmethod
    def get_app_pid(serial: str, package_id: str) -> str:
        """Gets the active process ID (PID) of the target package."""
        return AdbControl.ask(serial, "pidof", package_id) or ""

    @staticmethod
    def simulate_low_memory(serial: str, pid: str):
        """Triggers a low memory simulation (RUNNING_CRITICAL) on the target PID."""
        AdbControl.shell(
            serial, "am", "send-trim-memory", pid, "RUNNING_CRITICAL",
            doing="simulate low memory",
        )

    @staticmethod
    def simulate_incoming_call(serial: str, phone: str):
        """Simulates an incoming GSM call on the emulator."""
        AdbControl.adb(
            serial, "emu", "gsm", "call", str(phone), doing="simulate incoming call"
        )

    @staticmethod
    def open_deeplink(serial: str, url: str, package_id: str = None):
        """Opens a deep link URL intent via Activity Manager."""
        args = ["am", "start", "-a", "android.intent.action.VIEW", "-d", url]
        if package_id:
            args.append(package_id)
        AdbControl.shell(serial, *args, doing="open deep link")

    @staticmethod
    def force_stop_app(serial: str, package_id: str):
        """Force stops the target application package."""
        AdbControl.shell(serial, "am", "force-stop", package_id, doing="force stop app")

    @staticmethod
    def clear_app_data(serial: str, package_id: str):
        """Clears all storage and user data for the target application package."""
        AdbControl.shell(serial, "pm", "clear", package_id, doing="clear app data")

    @staticmethod
    def push_file(serial: str, local_path: str, remote_path: str):
        """Pushes a local file to the emulator filesystem."""
        AdbControl.adb(serial, "push", local_path, remote_path, doing="push file")

    @staticmethod
    def scan_media_file(serial: str, remote_path: str):
        """Broadcasts a media scan request for the specified path."""
        AdbControl.shell(
            serial, "am", "broadcast",
            "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
            "-d", f"file://{remote_path}", doing="scan media file",
        )

    @staticmethod
    def start_app_via_monkey(serial: str, package_id: str):
        """Starts the target app using the monkey tool."""
        AdbControl.shell(
            serial, "monkey", "-p", package_id, "1", doing="start app via monkey"
        )

    @staticmethod
    def set_accessibility_service(serial: str, service: str, enabled: bool):
        """Adds or removes one accessibility service, leaving the others alone.

        The setting holds every enabled service as one colon-separated list.
        Writing the one service replaces that list, and writing an empty list
        to switch one off turns off every screen reader the owner had running.
        So the list is read, the one service is added to it or taken out of it,
        and the result is written back.
        """
        listed = AdbControl.enabled_accessibility_services(serial)
        services = [name for name in (listed or "").split(":") if name]

        if enabled and service not in services:
            services.append(service)
        elif not enabled and service in services:
            services.remove(service)

        logger.info(
            "%s accessibility service %s",
            "Registering" if enabled else "Deregistering",
            service,
        )
        AdbControl.set_accessibility_services(serial, ":".join(services))

    @staticmethod
    def enabled_accessibility_services(serial: str) -> Optional[str]:
        """Every enabled accessibility service, as the setting stores them."""
        listed = AdbControl.ask(
            serial, "settings", "get", "secure", "enabled_accessibility_services"
        )
        return "" if listed in (None, "null") else listed

    @staticmethod
    def set_accessibility_services(serial: str, services: str):
        """Writes the whole list back, and the master switch to match it.

        An empty list needs the quoted empty string, because `settings put`
        takes no argument otherwise and leaves the old value in place.
        """
        AdbControl.shell(
            serial, "settings", "put", "secure",
            "enabled_accessibility_services", services or '""',
            doing="set the accessibility services",
        )
        AdbControl.shell(
            serial, "settings", "put", "secure",
            "accessibility_enabled", "1" if services else "0",
            doing="set the accessibility master switch",
        )
