"""The ADB command surface used by Android device components."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from yewo.devices.android._validation import (
    ACTIVITY_PATTERN as _ACTIVITY_PATTERN,
    LOCALE_PATTERN as _LOCALE_PATTERN,
    TIMEZONE_PATTERN as _TIMEZONE_PATTERN,
    quote_remote_arguments as _quote_remote_arguments,
    validate_component_name as _validate_component_name,
    validate_deep_link as _validate_deep_link,
    validate_package_id as _validate_package_id,
    validate_pattern as _validate_pattern,
)
from yewo.devices._commands import (
    CommandRunner,
    capture_optional,
    execute_command,
    run_command,
)
from yewo.devices.errors import DeviceCommandError


@dataclass(frozen=True)
class VisualFeedbackSettings:
    """Android overlays that alter pixels observed by an agent."""

    pointer_location: bool | None
    tap_highlights: bool | None


class AdbClient:
    """Build and execute all ADB commands in one boundary."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        adb_executable: str = "adb",
        emulator_executable: str | None = None,
    ) -> None:
        self._runner = runner
        self._adb_executable = adb_executable
        self._emulator_executable = emulator_executable

    @property
    def executable(self) -> str:
        """Return the configured ADB executable."""
        return self._adb_executable

    def run(
        self,
        *args: str,
        purpose: str = "run an ADB command",
        timeout_seconds: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        """Run a required ADB server command."""
        return execute_command(
            self._runner,
            [self._adb_executable, *args],
            timeout_seconds=timeout_seconds,
            purpose=purpose,
        )

    def adb(
        self,
        connection_id: str,
        *args: str,
        purpose: str = "run an ADB command",
        timeout_seconds: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        """Run a required ADB command against one device."""
        return self.run(
            "-s",
            connection_id,
            *args,
            purpose=purpose,
            timeout_seconds=timeout_seconds,
        )

    def shell(
        self,
        connection_id: str,
        *args: str,
        purpose: str = "run an Android shell command",
        timeout_seconds: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        """Run a required shell command against one device."""
        return self.adb(
            connection_id,
            "shell",
            *_quote_remote_arguments(args),
            purpose=purpose,
            timeout_seconds=timeout_seconds,
        )

    def ask(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str | None:
        """Read from one Android shell, or return no answer."""
        return capture_optional(
            self._runner,
            [
                self._adb_executable,
                "-s",
                connection_id,
                "shell",
                *_quote_remote_arguments(args),
            ],
            timeout_seconds=timeout_seconds,
        )

    def ask_adb(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str | None:
        """Read from one non-shell ADB command, or return no answer."""
        return capture_optional(
            self._runner,
            [self._adb_executable, "-s", connection_id, *args],
            timeout_seconds=timeout_seconds,
        )

    def device_listing(self, *, long: bool = True) -> str:
        """Return the required ADB device listing."""
        args = ("devices", "-l") if long else ("devices",)
        return self.run(*args, purpose="list Android devices").stdout or ""

    def attached_devices(self) -> dict[str, str]:
        """Return every connection ID and its native ADB state."""
        states: dict[str, str] = {}
        for line in self.device_listing(long=False).splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = (
                    "no permissions" if parts[1:3] == ["no", "permissions"] else parts[1]
                )
        return states

    def emulator_executable(self) -> str | None:
        """Return the configured Android emulator executable when available."""
        if self._emulator_executable:
            return self._emulator_executable
        found = shutil.which("emulator")
        if found:
            return found
        android_home = os.getenv("ANDROID_HOME")
        candidate = Path(android_home) / "emulator" / "emulator" if android_home else None
        return str(candidate) if candidate is not None and candidate.is_file() else None

    def list_avds(self) -> tuple[str, ...] | None:
        """Return configured AVD names, or no answer when the tool is absent."""
        emulator = self.emulator_executable()
        if emulator is None:
            return None
        output = capture_optional(
            self._runner,
            [emulator, "-list-avds"],
            timeout_seconds=30,
        )
        if output is None:
            return None
        return tuple(line.strip() for line in output.splitlines() if line.strip())

    def boot_completed(self, connection_id: str) -> bool:
        """Return whether Android finished booting."""
        return self.ask(connection_id, "getprop", "sys.boot_completed") == "1"

    def system_services_ready(self, connection_id: str) -> bool:
        """Return whether Appium's required Android services are available."""
        outputs = (self.ask(connection_id, "service", "check", item) for item in ("activity", "settings"))
        return all(
            output is not None and output.strip().lower().endswith(": found") for output in outputs
        )

    def read_show_ime_with_hard_keyboard(self, connection_id: str) -> bool | None:
        """Read whether Android renders its soft keyboard with a hardware keyboard."""
        result = self.shell(
            connection_id,
            "settings",
            "get",
            "secure",
            "show_ime_with_hard_keyboard",
            purpose="read the soft-keyboard display setting",
        )
        value = (result.stdout or "").strip()
        if value in {"", "null"}:
            return None
        if value in {"0", "1"}:
            return value == "1"
        raise DeviceCommandError("Android returned an invalid soft-keyboard display setting.")

    def set_show_ime_with_hard_keyboard(
        self,
        connection_id: str,
        enabled: bool | None,
    ) -> None:
        """Set the soft-keyboard display setting or restore its absent state."""
        operation = "delete" if enabled is None else "put"
        arguments = () if enabled is None else ("1" if enabled else "0",)
        self.shell(
            connection_id,
            "settings",
            operation,
            "secure",
            "show_ime_with_hard_keyboard",
            *arguments,
            purpose="set the soft-keyboard display setting",
        )

    def read_stylus_handwriting_enabled(self, connection_id: str) -> bool | None:
        """Read whether Android may replace the keyboard with handwriting UI."""
        result = self.shell(
            connection_id,
            "settings",
            "get",
            "secure",
            "stylus_handwriting_enabled",
            purpose="read the stylus-handwriting setting",
        )
        value = (result.stdout or "").strip()
        if value in {"", "null"}:
            return None
        if value in {"0", "1"}:
            return value == "1"
        raise DeviceCommandError("Android returned an invalid stylus-handwriting setting.")

    def set_stylus_handwriting_enabled(
        self,
        connection_id: str,
        enabled: bool | None,
    ) -> None:
        """Set the stylus-handwriting setting or restore its absent state."""
        operation = "delete" if enabled is None else "put"
        arguments = () if enabled is None else ("1" if enabled else "0",)
        self.shell(
            connection_id,
            "settings",
            operation,
            "secure",
            "stylus_handwriting_enabled",
            *arguments,
            purpose="set the stylus-handwriting setting",
        )

    def avd_name(self, connection_id: str) -> str | None:
        """Return the stable AVD name of a running emulator."""
        property_name = self.ask(
            connection_id,
            "getprop",
            "ro.boot.qemu.avd_name",
            timeout_seconds=10,
        )
        property_lines = (property_name or "").splitlines()
        if property_lines and property_lines[0].strip():
            return property_lines[0].strip()
        console_name = self.ask_adb(
            connection_id,
            "emu",
            "avd",
            "name",
            timeout_seconds=10,
        )
        console_lines = (console_name or "").splitlines()
        return console_lines[0].strip() if console_lines and console_lines[0].strip() else None

    def resolve_emulator_connection(self, target_id: str) -> str | None:
        """Resolve an AVD name or active connection ID."""
        for connection_id, state in self.attached_devices().items():
            if state != "device":
                continue
            if connection_id == target_id:
                return connection_id
            if connection_id.startswith("emulator-") and self.avd_name(connection_id) == target_id:
                return connection_id
        return None

    def installed_packages(self, connection_id: str) -> set[str] | None:
        """Return installed package IDs, or no answer when inspection fails."""
        output = self.ask(connection_id, "pm", "list", "packages")
        if output is None:
            return None
        return {
            line.strip().removeprefix("package:")
            for line in (output or "").splitlines()
            if line.strip()
        }

    def uninstall_app(self, connection_id: str, package_id: str) -> bool:
        """Uninstall one package and report whether ADB accepted the command."""
        output = capture_optional(
            self._runner,
            [self._adb_executable, "-s", connection_id, "uninstall", package_id],
            timeout_seconds=60,
        )
        return output is not None

    def set_tap_highlights(self, connection_id: str, enabled: bool) -> None:
        self.shell(
            connection_id,
            "settings",
            "put",
            "system",
            "show_touches",
            "1" if enabled else "0",
            purpose="set tap highlights",
        )

    def set_pointer_location(self, connection_id: str, enabled: bool) -> None:
        self.shell(
            connection_id,
            "settings",
            "put",
            "system",
            "pointer_location",
            "1" if enabled else "0",
            purpose="set pointer location",
        )

    def read_visual_feedback(self, connection_id: str) -> VisualFeedbackSettings:
        """Read visual feedback settings before changing them."""

        def read_boolean(setting: str) -> bool | None:
            value = self.ask(connection_id, "settings", "get", "system", setting)
            return True if value == "1" else False if value == "0" else None

        return VisualFeedbackSettings(
            pointer_location=read_boolean("pointer_location"),
            tap_highlights=read_boolean("show_touches"),
        )

    def set_visual_feedback(self, connection_id: str, settings: VisualFeedbackSettings) -> None:
        """Apply each known visual feedback setting."""
        if settings.pointer_location is not None:
            self.set_pointer_location(connection_id, settings.pointer_location)
        if settings.tap_highlights is not None:
            self.set_tap_highlights(connection_id, settings.tap_highlights)

    def kill_emulator(self, connection_id: str) -> None:
        self.adb(connection_id, "emu", "kill", purpose="stop the emulator")

    def set_location(self, connection_id: str, latitude: float, longitude: float) -> None:
        self.adb(
            connection_id,
            "emu",
            "geo",
            "fix",
            str(longitude),
            str(latitude),
            purpose="set the simulated location",
        )

    def set_airplane_mode(self, connection_id: str, enabled: bool) -> None:
        self.shell(
            connection_id,
            "cmd",
            "connectivity",
            "airplane-mode",
            "enable" if enabled else "disable",
            purpose="set airplane mode",
        )

    def set_battery(self, connection_id: str, level: int, charging: bool) -> None:
        if level < 0 or level > 100:
            raise DeviceCommandError("The battery level must be between 0 and 100.")
        self.shell(
            connection_id,
            "dumpsys",
            "battery",
            "set",
            "level",
            str(level),
            purpose="set the battery level",
        )
        self.shell(
            connection_id,
            "dumpsys",
            "battery",
            "set",
            "ac",
            "1" if charging else "0",
            purpose="set the charging state",
        )

    def reset_battery(self, connection_id: str) -> None:
        self.shell(
            connection_id,
            "dumpsys",
            "battery",
            "reset",
            purpose="reset the battery state",
        )

    def set_timezone(self, connection_id: str, timezone: str) -> None:
        _validate_pattern(timezone, _TIMEZONE_PATTERN, "timezone")
        self.shell(
            connection_id,
            "settings",
            "put",
            "global",
            "timezone",
            timezone,
            purpose="set the timezone",
        )

    def set_app_locale(self, connection_id: str, package_id: str, locale: str) -> None:
        _validate_package_id(package_id)
        _validate_pattern(locale, _LOCALE_PATTERN, "locale")
        self.shell(
            connection_id,
            "cmd",
            "locale",
            "set-app-locales",
            package_id,
            "--locales",
            locale,
            purpose="set the application locale",
        )

    def set_app_permission(
        self, connection_id: str, package_id: str, permission: str, granted: bool
    ) -> None:
        _validate_package_id(package_id)
        _validate_package_id(permission, name="permission")
        self.shell(
            connection_id,
            "pm",
            "grant" if granted else "revoke",
            package_id,
            permission,
            purpose="set an application permission",
        )

    def app_process_id(self, connection_id: str, package_id: str) -> str | None:
        _validate_package_id(package_id)
        return self.ask(connection_id, "pidof", package_id)

    def send_low_memory(self, connection_id: str, process_id: str) -> None:
        if not process_id.isdecimal():
            raise DeviceCommandError("The Android process ID must contain only digits.")
        self.shell(
            connection_id,
            "am",
            "send-trim-memory",
            process_id,
            "RUNNING_CRITICAL",
            purpose="send a low-memory event",
        )

    def simulate_incoming_call(self, connection_id: str, phone_number: str) -> None:
        self.adb(
            connection_id,
            "emu",
            "gsm",
            "call",
            phone_number,
            purpose="simulate an incoming call",
        )

    def open_deep_link(self, connection_id: str, url: str, package_id: str | None = None) -> None:
        _validate_deep_link(url)
        args = ["am", "start", "-a", "android.intent.action.VIEW", "-d", url]
        if package_id:
            _validate_package_id(package_id)
            args.extend(("-p", package_id))
        self.shell(connection_id, *args, purpose="open a deep link")

    def force_stop_app(self, connection_id: str, package_id: str) -> None:
        _validate_package_id(package_id)
        self.shell(
            connection_id,
            "am",
            "force-stop",
            package_id,
            purpose="stop the application",
        )

    def clear_app_data(self, connection_id: str, package_id: str) -> None:
        _validate_package_id(package_id)
        self.shell(
            connection_id,
            "pm",
            "clear",
            package_id,
            purpose="clear application data",
        )

    def start_app(
        self,
        connection_id: str,
        package_id: str,
        launcher_activity: str,
    ) -> None:
        _validate_package_id(package_id)
        _validate_pattern(launcher_activity, _ACTIVITY_PATTERN, "launcher activity")
        self.shell(
            connection_id,
            "am",
            "start",
            "-n",
            f"{package_id}/{launcher_activity}",
            purpose="start the application",
        )

    def enabled_accessibility_services(self, connection_id: str) -> str | None:
        value = self.ask(
            connection_id,
            "settings",
            "get",
            "secure",
            "enabled_accessibility_services",
        )
        return "" if value == "null" else value

    def set_accessibility_services(self, connection_id: str, services: str) -> None:
        for service in (item for item in services.split(":") if item):
            _validate_component_name(service)
        self.shell(
            connection_id,
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            services,
            purpose="set accessibility services",
        )
        self.shell(
            connection_id,
            "settings",
            "put",
            "secure",
            "accessibility_enabled",
            "1" if services else "0",
            purpose="set accessibility",
        )

    def set_accessibility_service(self, connection_id: str, service: str, enabled: bool) -> None:
        _validate_component_name(service)
        services = [
            item
            for item in (self.enabled_accessibility_services(connection_id) or "").split(":")
            if item
        ]
        if enabled and service not in services:
            services.append(service)
        elif not enabled and service in services:
            services.remove(service)
        self.set_accessibility_services(connection_id, ":".join(services))
