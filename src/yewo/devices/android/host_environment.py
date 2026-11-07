"""Host controls that isolate an Android emulator from the macOS pasteboard."""

from collections.abc import Callable
from contextlib import suppress

from yewo.devices._commands import (
    CommandRunner,
    capture_optional,
    execute_command,
    run_command,
)
from yewo.devices.errors import DeviceLifecycleError


_DOMAIN = "com.android.Emulator"
_CLIPBOARD_SHARING_KEY = "set.clipboardSharing"


class MacOsAndroidEmulatorEnvironment:
    """Disable emulator pasteboard sharing and restore the host preference."""

    def __init__(self, runner: CommandRunner = run_command) -> None:
        self._runner = runner

    def isolate_clipboard(self) -> Callable[[], None]:
        """Disable host sharing and return one restoration operation."""
        original = capture_optional(
            self._runner,
            ["defaults", "read", _DOMAIN, _CLIPBOARD_SHARING_KEY],
        )
        if original not in {None, "0", "1"}:
            raise DeviceLifecycleError(
                "The Android emulator clipboard-sharing preference is invalid."
            )

        def restore() -> None:
            if original is None:
                execute_command(
                    self._runner,
                    ["defaults", "delete", _DOMAIN, _CLIPBOARD_SHARING_KEY],
                    purpose="restore Android emulator clipboard sharing",
                )
                return
            self._write(original == "1")

        try:
            self._write(False)
            if (
                capture_optional(
                    self._runner,
                    ["defaults", "read", _DOMAIN, _CLIPBOARD_SHARING_KEY],
                )
                != "0"
            ):
                raise DeviceLifecycleError("Could not verify Android emulator clipboard isolation.")
        except BaseException:
            with suppress(Exception):
                restore()
            raise

        return restore

    def _write(self, enabled: bool) -> None:
        execute_command(
            self._runner,
            [
                "defaults",
                "write",
                _DOMAIN,
                _CLIPBOARD_SHARING_KEY,
                "-bool",
                "true" if enabled else "false",
            ],
            purpose="set Android emulator clipboard sharing",
        )
