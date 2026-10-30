"""Tests for the ported ADB command client."""

import pytest

from devices._runner import CommandRunnerStub
from yewo.devices.android import AdbClient, VisualFeedbackSettings
from yewo.devices.errors import DeviceCommandError


def test_shell_binds_every_command_to_one_connection() -> None:
    runner = CommandRunnerStub()

    AdbClient(runner).set_airplane_mode("emulator-5554", True)

    assert runner.calls == [
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "cmd",
                "connectivity",
                "airplane-mode",
                "enable",
            ),
            30,
        )
    ]


def test_required_command_preserves_its_purpose() -> None:
    runner = CommandRunnerStub(stderr="device offline", returncode=1)

    with pytest.raises(DeviceCommandError, match="set airplane mode"):
        AdbClient(runner).set_airplane_mode("emulator-5554", True)


def test_visual_feedback_reads_both_settings() -> None:
    runner = CommandRunnerStub(responses=(("1\n", "", 0), ("0\n", "", 0)))

    settings = AdbClient(runner).read_visual_feedback("emulator-5554")

    assert settings == VisualFeedbackSettings(pointer_location=True, tap_highlights=False)


def test_avd_name_uses_console_only_when_the_property_is_empty() -> None:
    runner = CommandRunnerStub(responses=(("", "", 0), ("Pixel_9\nOK\n", "", 0)))

    name = AdbClient(runner).avd_name("emulator-5554")

    assert name == "Pixel_9"
    assert [call[0] for call in runner.calls] == [
        ("adb", "-s", "emulator-5554", "shell", "getprop", "ro.boot.qemu.avd_name"),
        ("adb", "-s", "emulator-5554", "emu", "avd", "name"),
    ]


def test_accessibility_update_preserves_other_services() -> None:
    runner = CommandRunnerStub(responses=(("example/Existing\n", "", 0), ("", "", 0), ("", "", 0)))

    AdbClient(runner).set_accessibility_service("emulator-5554", "example/New", enabled=True)

    commands = [call[0] for call in runner.calls]
    assert commands[1][-2:] == ("enabled_accessibility_services", "example/Existing:example/New")
    assert commands[2][-2:] == ("accessibility_enabled", "1")


def test_deep_link_binds_the_target_package_with_package_flag() -> None:
    runner = CommandRunnerStub()

    AdbClient(runner).open_deep_link(
        "emulator-5554",
        "example://item/1",
        "com.example.app",
    )

    assert runner.calls == [
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                "example://item/1",
                "-p",
                "com.example.app",
            ),
            30,
        )
    ]


@pytest.mark.parametrize(
    ("operation", "expected_commands"),
    [
        (
            lambda client: client.set_location("emulator-5554", 51.5, -0.1),
            [("emu", "geo", "fix", "-0.1", "51.5")],
        ),
        (
            lambda client: client.kill_emulator("emulator-5554"),
            [("emu", "kill")],
        ),
        (
            lambda client: client.set_pointer_location("emulator-5554", True),
            [("shell", "settings", "put", "system", "pointer_location", "1")],
        ),
        (
            lambda client: client.set_tap_highlights("emulator-5554", False),
            [("shell", "settings", "put", "system", "show_touches", "0")],
        ),
        (
            lambda client: client.set_visual_feedback(
                "emulator-5554",
                VisualFeedbackSettings(pointer_location=True, tap_highlights=False),
            ),
            [
                ("shell", "settings", "put", "system", "pointer_location", "1"),
                ("shell", "settings", "put", "system", "show_touches", "0"),
            ],
        ),
        (
            lambda client: client.set_battery("emulator-5554", 20, False),
            [
                ("shell", "dumpsys", "battery", "set", "level", "20"),
                ("shell", "dumpsys", "battery", "set", "ac", "0"),
            ],
        ),
        (
            lambda client: client.reset_battery("emulator-5554"),
            [("shell", "dumpsys", "battery", "reset")],
        ),
        (
            lambda client: client.set_timezone("emulator-5554", "Europe/London"),
            [("shell", "settings", "put", "global", "timezone", "Europe/London")],
        ),
        (
            lambda client: client.set_app_locale("emulator-5554", "com.example.app", "en-GB"),
            [
                (
                    "shell",
                    "cmd",
                    "locale",
                    "set-app-locales",
                    "com.example.app",
                    "--locales",
                    "en-GB",
                )
            ],
        ),
        (
            lambda client: client.set_app_permission(
                "emulator-5554",
                "com.example.app",
                "android.permission.CAMERA",
                True,
            ),
            [
                (
                    "shell",
                    "pm",
                    "grant",
                    "com.example.app",
                    "android.permission.CAMERA",
                )
            ],
        ),
        (
            lambda client: client.send_low_memory("emulator-5554", "123"),
            [
                (
                    "shell",
                    "am",
                    "send-trim-memory",
                    "123",
                    "RUNNING_CRITICAL",
                )
            ],
        ),
        (
            lambda client: client.app_process_id("emulator-5554", "com.example.app"),
            [("shell", "pidof", "com.example.app")],
        ),
        (
            lambda client: client.simulate_incoming_call("emulator-5554", "+44123"),
            [("emu", "gsm", "call", "+44123")],
        ),
        (
            lambda client: client.force_stop_app("emulator-5554", "com.example.app"),
            [("shell", "am", "force-stop", "com.example.app")],
        ),
        (
            lambda client: client.clear_app_data("emulator-5554", "com.example.app"),
            [("shell", "pm", "clear", "com.example.app")],
        ),
        (
            lambda client: client.push_file("emulator-5554", "photo.png", "/sdcard/photo.png"),
            [("push", "photo.png", "/sdcard/photo.png")],
        ),
        (
            lambda client: client.scan_media_file("emulator-5554", "/sdcard/photo.png"),
            [
                (
                    "shell",
                    "am",
                    "broadcast",
                    "-a",
                    "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
                    "-d",
                    "file:///sdcard/photo.png",
                )
            ],
        ),
        (
            lambda client: client.start_app(
                "emulator-5554",
                "com.example.app",
                "com.example.app.MainActivity",
            ),
            [
                (
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "com.example.app/com.example.app.MainActivity",
                )
            ],
        ),
        (
            lambda client: client.set_accessibility_services(
                "emulator-5554", "com.example/Service"
            ),
            [
                (
                    "shell",
                    "settings",
                    "put",
                    "secure",
                    "enabled_accessibility_services",
                    "com.example/Service",
                ),
                (
                    "shell",
                    "settings",
                    "put",
                    "secure",
                    "accessibility_enabled",
                    "1",
                ),
            ],
        ),
    ],
)
def test_android_capability_builds_bound_commands(operation, expected_commands) -> None:
    runner = CommandRunnerStub()

    operation(AdbClient(runner))

    assert [call[0][3:] for call in runner.calls] == expected_commands


@pytest.mark.parametrize(
    ("operation", "expected_command"),
    [
        (
            lambda client: client.boot_completed("emulator-5554"),
            ("shell", "getprop", "sys.boot_completed"),
        ),
        (
            lambda client: client.installed_packages("emulator-5554"),
            ("shell", "pm", "list", "packages"),
        ),
        (
            lambda client: client.uninstall_app("emulator-5554", "com.example.app"),
            ("uninstall", "com.example.app"),
        ),
        (
            lambda client: client.enabled_accessibility_services("emulator-5554"),
            (
                "shell",
                "settings",
                "get",
                "secure",
                "enabled_accessibility_services",
            ),
        ),
    ],
)
def test_android_capability_read_is_bound_to_one_device(
    operation,
    expected_command,
) -> None:
    runner = CommandRunnerStub(stdout="value")

    operation(AdbClient(runner))

    assert runner.calls[0][0][3:] == expected_command


def test_shell_quotes_values_for_the_remote_android_shell() -> None:
    runner = CommandRunnerStub()
    client = AdbClient(runner)

    client.open_deep_link(
        "emulator-5554",
        "example://item/1;echo injected",
        "com.example.app",
    )
    client.scan_media_file("emulator-5554", "/sdcard/My Photo.png")
    client.set_accessibility_services("emulator-5554", "")

    assert runner.calls[0][0][-2:] == (
        "-p",
        "com.example.app",
    )
    assert runner.calls[0][0][-3] == "'example://item/1;echo injected'"
    assert runner.calls[1][0][-1] == "'file:///sdcard/My Photo.png'"
    assert runner.calls[2][0][-1] == "''"


@pytest.mark.parametrize(
    "operation",
    [
        lambda client: client.set_battery("emulator-5554", -1, False),
        lambda client: client.set_battery("emulator-5554", 101, False),
        lambda client: client.set_timezone("emulator-5554", "Europe/London;reboot"),
        lambda client: client.set_app_locale(
            "emulator-5554", "com.example.app;reboot", "en-GB"
        ),
        lambda client: client.set_app_permission(
            "emulator-5554",
            "com.example.app",
            "android.permission.CAMERA;reboot",
            True,
        ),
        lambda client: client.send_low_memory("emulator-5554", "123;reboot"),
        lambda client: client.start_app(
            "emulator-5554",
            "com.example.app",
            "MainActivity;reboot",
        ),
    ],
)
def test_invalid_android_capability_value_fails_before_dispatch(operation) -> None:
    runner = CommandRunnerStub()

    with pytest.raises(DeviceCommandError):
        operation(AdbClient(runner))

    assert runner.calls == []
