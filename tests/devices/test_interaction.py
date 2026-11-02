"""Tests for validated cross-platform device interaction."""

from typing import Any

import pytest

from devices._interaction_stubs import DriverStub as _DriverStub
from devices._interaction_stubs import application as _application
from yewo.devices import DeviceKind, DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.errors import DeviceInputRejectedError, DeviceInteractionError
from yewo.devices.interaction import AndroidKey, DeviceInteraction, ScrollDirection
from yewo.devices.ios.interaction import IosInteractionBackend


class _Clock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


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


@pytest.mark.parametrize("distance_fraction", [0, -0.1, 0.91])
def test_scroll_rejects_an_unbounded_distance(distance_fraction: float) -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError, match="Scroll distance"):
        interaction.scroll(ScrollDirection.UP, distance_fraction=distance_fraction)

    assert driver.scripts == []


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

    assert value is True
    assert driver.element.cleared
    assert driver.keycodes == [29, 67]
    assert interaction.focused_field_state() is True


class _NumericInputReader:
    def ask(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str:
        del connection_id, args, timeout_seconds
        return "mCurAttribute=EditorInfo{packageName=com.example, inputType=0x2, imeOptions=5}"


def test_numeric_focused_field_rejects_text_before_delivery() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(_NumericInputReader(), "emulator-5554"),
    )

    with pytest.raises(DeviceInputRejectedError, match="numeric input"):
        interaction.type_focused_text("medium")

    assert driver.element.value == ""


def test_text_operation_uses_the_in_bounds_appium_element_without_active_focus() -> None:
    driver = _DriverStub()
    driver.switch_to.active_element_failures = 100
    driver.focused_xpath_available = False
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    assert interaction.type_text(20, 30, "hello") is True
    assert driver.scripts == []
    assert driver.element.clicked


def test_text_operation_rejects_an_appium_element_that_did_not_gain_focus() -> None:
    driver = _DriverStub()
    driver.element.focused = False
    driver.switch_to.active_element_failures = 100
    driver.focused_xpath_available = False
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        focus_timeout_seconds=0.1,
        focus_poll_seconds=0.05,
        clock=_Clock(),
    )

    with pytest.raises(DeviceInteractionError, match="did not accept keyboard focus"):
        interaction.type_text(20, 30, "must not land")

    assert driver.element.clicked
    assert driver.element.value == ""


def test_text_operation_rejects_multiple_appium_fields_at_the_target() -> None:
    driver = _DriverStub()
    overlapping = type(driver.element)()
    driver.text_elements.append(overlapping)
    driver.switch_to.active_element_failures = 100
    driver.focused_xpath_available = False
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        focus_timeout_seconds=0.1,
        focus_poll_seconds=0.05,
        clock=_Clock(),
    )

    with pytest.raises(DeviceInteractionError, match="target field"):
        interaction.type_text(20, 30, "must not land")

    assert driver.element.value == ""
    assert overlapping.value == ""


def test_visible_keyboard_prevents_coordinate_fallback_without_an_editable_target() -> None:
    driver = _DriverStub()
    driver.text_elements = []
    driver.switch_to.active_element_failures = 100
    driver.focused_xpath_available = False
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError, match="no editable target field"):
        interaction.type_text(20, 30, "must not reach the keyboard")

    assert driver.scripts == []
    assert driver.keycodes == []


def test_focus_rejects_an_unchanged_field_away_from_the_target() -> None:
    driver = _DriverStub()
    driver.element.rect = {"x": 0, "y": 0, "width": 20, "height": 20}
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        focus_timeout_seconds=0.1,
        focus_poll_seconds=0.05,
        clock=_Clock(),
    )

    with pytest.raises(DeviceInteractionError, match="target field"):
        interaction.type_text(160, 320, "wrong field")

    assert driver.element.value == ""


def test_focused_field_reading_contains_no_field_value() -> None:
    driver = _DriverStub()
    driver.element.value = "private value"
    driver.keyboard = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    reading = interaction.focused_field_reading()

    assert reading.is_focused is True
    assert reading.driver_bounds == (0, 0, 50, 100)
    assert reading.keyboard_visible is True
    assert "private value" not in repr(reading)


def test_type_text_accepts_append_behavior_without_returning_field_content() -> None:
    driver = _DriverStub()
    driver.element.value = "private value"
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    assert interaction.type_text(20, 30, " appended") is True


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

    assert value is True
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


@pytest.mark.parametrize("duration_seconds", [0, 0.05, 30.1])
def test_wait_rejects_duration_outside_the_device_bounds(duration_seconds: float) -> None:
    interaction = DeviceInteraction(
        _DriverStub(),
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        clock=_Clock(),
    )

    with pytest.raises(DeviceInteractionError, match="Wait duration"):
        interaction.wait(duration_seconds)


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
