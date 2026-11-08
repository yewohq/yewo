"""Device tooling that only one host family provides."""

from __future__ import annotations

import sys


def supports_apple_device_tooling() -> bool:
    """Return whether this host provides Xcode simulator and device tooling."""
    return sys.platform == "darwin"


def supports_android_emulator_clipboard_control() -> bool:
    """Return whether Yewo can isolate the emulator clipboard on this host.

    Only macOS exposes the emulator preference Yewo reads and restores. On
    other hosts a run proceeds without clipboard isolation.
    """
    return sys.platform == "darwin"
