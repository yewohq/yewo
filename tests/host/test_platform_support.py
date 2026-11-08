"""Tests for device tooling that only one host family provides."""

from __future__ import annotations

import pytest

from host.windows_simulation import load_host_module_as


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("darwin", True), ("linux", False), ("win32", False)],
)
def test_apple_device_tooling_exists_only_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected: bool,
) -> None:
    module = load_host_module_as(monkeypatch, "platform_support", platform_name)

    assert module.supports_apple_device_tooling() is expected


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("darwin", True), ("linux", False), ("win32", False)],
)
def test_emulator_clipboard_control_exists_only_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected: bool,
) -> None:
    module = load_host_module_as(monkeypatch, "platform_support", platform_name)

    assert module.supports_android_emulator_clipboard_control() is expected
