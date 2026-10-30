"""Tests for validated cross-platform device interaction."""

from typing import Any

import pytest

from devices._interaction_stubs import DriverStub as _DriverStub
from devices._interaction_stubs import application as _application
from yewo.devices import DeviceKind, DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.errors import DeviceInteractionError
from yewo.devices.interaction import AndroidKey, DeviceInteraction, ScrollDirection
from yewo.devices.ios.interaction import IosInteractionBackend


def test_android_tap_validates_and_scales_screenshot_coordinates() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.tap(100, 200)

    assert driver.scripts == [("mobile: clickGesture", {"x": 50, "y": 100})]


def test_edge_screenshot_point_stays_inside_the_driver_space() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.tap(199, 399)

    assert driver.scripts == [("mobile: clickGesture", {"x": 99, "y": 199})]


def test_interaction_exposes_calibrated_platform_context() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.screenshot()
    calibration = interaction.screen_calibration()

    assert calibration.scale_x == 0.5
    assert calibration.scale_y == 0.5
    assert interaction.application_context() == "com.example.MainActivity"
    assert interaction.device_orientation() == "portrait"


def test_out_of_bounds_point_fails_before_dispatch() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError, match="outside"):
        interaction.tap(200, 10)

    assert driver.scripts == []


@pytest.mark.parametrize(
    ("count", "interval_seconds"),
    [(0, 0.1), (11, 0.1), (1, -0.1), (1, 1.1)],
)
def test_tap_limits_fail_before_dispatch(
    count: int,
    interval_seconds: float,
) -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError):
        interaction.tap(20, 20, count=count, interval_seconds=interval_seconds)

    assert driver.scripts == []


def test_scroll_translates_semantics_to_a_bounded_drag() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    interaction.scroll(ScrollDirection.UP, left=20, top=40, right=180, bottom=360)

    script, params = driver.scripts[-1]
    assert script == "mobile: dragFromToForDuration"
    assert params == {
        "fromX": 50,
        "fromY": 148,
        "toX": 50,
        "toY": 52,
        "duration": 0.5,
    }


def test_text_operations_use_the_focused_element() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    value = interaction.type_text(20, 30, "hello")
    interaction.clear_text(20, 30)

    assert value == "hello"
    assert driver.element.cleared
    assert interaction.focused_field_state() is True


def test_android_type_text_never_requests_the_ios_only_value_attribute() -> None:
    """UiAutomator2 raises NoSuchAttributeException for 'value'; only XCUITest supports it."""
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.type_text(20, 30, "hello")

    assert "value" not in driver.element.requested_attributes


def test_ios_type_text_reads_the_value_attribute() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    value = interaction.type_text(20, 30, "hello")

    assert value == "hello"
    assert "value" in driver.element.requested_attributes


def test_keyboard_dismissal_is_verified_without_back_navigation() -> None:
    driver = _DriverStub()
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    assert interaction.dismiss_keyboard()
    assert driver.back_calls == 0


def test_platform_keys_reject_the_wrong_vocabulary_before_dispatch() -> None:
    android_driver = _DriverStub()
    android = DeviceInteraction(
        android_driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )
    ios_driver = _DriverStub()
    ios = DeviceInteraction(
        ios_driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError, match="Android keys"):
        android.press_key("home")
    with pytest.raises(DeviceInteractionError, match="iOS button name"):
        invalid_ios_key: Any = -1
        ios.press_key(invalid_ios_key)

    assert android_driver.keycodes == []
    assert ios_driver.scripts == []


def test_android_key_uses_the_supported_vocabulary() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.press_key(AndroidKey.HOME)

    assert driver.keycodes == [AndroidKey.HOME.value]


def test_deep_link_uses_the_observed_application_id() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.open_deep_link("example://item/1")

    assert driver.scripts[-1] == (
        "mobile: deepLink",
        {
            "url": "example://item/1",
            "package": "com.example.app",
        },
    )


def test_ios_deep_link_uses_only_the_bundle_identifier() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    interaction.open_deep_link("example://item/1")

    assert driver.scripts[-1] == (
        "mobile: deepLink",
        {"url": "example://item/1", "bundleId": "com.example.app"},
    )


def test_clear_application_data_uses_platform_parameter() -> None:
    android_driver = _DriverStub()
    android = DeviceInteraction(
        android_driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )
    ios_driver = _DriverStub()
    ios = DeviceInteraction(
        ios_driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
        device_kind=DeviceKind.VIRTUAL,
    )

    android.clear_application_data()
    ios.clear_application_data()

    assert android_driver.scripts[-1] == (
        "mobile: clearApp",
        {"appId": "com.example.app"},
    )
    assert ios_driver.scripts[-1] == (
        "mobile: clearApp",
        {"bundleId": "com.example.app"},
    )


def test_physical_ios_clear_fails_before_dispatch() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
        device_kind=DeviceKind.PHYSICAL,
    )

    with pytest.raises(DeviceInteractionError, match="only on an iOS simulator"):
        interaction.clear_application_data()

    assert driver.scripts == []


def test_native_alert_detection_distinguishes_absence() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    assert interaction.system_prompt_visible() is False
    driver.switch_to.has_alert = True
    assert interaction.system_prompt_visible() is True
