"""Focus and text entry through the validated interaction boundary."""

import pytest

from devices._interaction_stubs import DriverStub as _DriverStub
from devices._interaction_stubs import application as _application
from yewo.devices import DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.errors import DeviceInputRejectedError, DeviceInteractionError
from yewo.devices.interaction import DeviceInteraction
from yewo.devices.ios.interaction import IosInteractionBackend
from support import ClockStub


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


def test_secure_field_reports_context_without_comparing_obscured_text() -> None:
    driver = _DriverStub()
    driver.element.password = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    matched = interaction.type_text(20, 30, "private value")
    reading = interaction.focused_field_reading()

    assert matched is None
    assert driver.element.value == "private value"
    assert reading.field_context is not None
    assert reading.field_context.is_secure is True
    assert reading.field_context.field_type.value == "password"


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

    with pytest.raises(DeviceInputRejectedError, match="whole digits only"):
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


def test_focus_clicks_an_already_focused_field_when_keyboard_is_hidden() -> None:
    driver = _DriverStub()
    driver.keyboard = False
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    interaction.focus(20, 30)

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
        clock=ClockStub(),
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
        clock=ClockStub(),
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
        clock=ClockStub(),
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
