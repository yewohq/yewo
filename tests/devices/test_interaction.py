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
from support import ClockStub


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
    # Scrolling up returns to content above, so the finger travels downward.
    assert params == {
        "fromX": 50,
        "fromY": 52,
        "toX": 50,
        "toY": 148,
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


@pytest.mark.parametrize("duration_seconds", [0, 0.05, 30.1])
def test_wait_rejects_duration_outside_the_device_bounds(duration_seconds: float) -> None:
    interaction = DeviceInteraction(
        _DriverStub(),
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        clock=ClockStub(),
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


@pytest.mark.parametrize(
    "direction", [ScrollDirection.DOWN, ScrollDirection.UP, ScrollDirection.RIGHT]
)
def test_android_scroll_asks_uiautomator_whether_more_content_remains(
    direction: ScrollDirection,
) -> None:
    driver = _DriverStub()
    driver.script_results["mobile: scrollGesture"] = True
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    can_scroll_more = interaction.scroll(direction, left=0, top=0, right=99, bottom=199)

    assert can_scroll_more is True
    script, params = driver.scripts[-1]
    assert script == "mobile: scrollGesture"
    # The region is converted like every other gesture region. This stub reports a 200x400
    # screenshot against a 100x200 gesture viewport, so an unconverted region is four times the
    # area the device offers and UiAutomator either mis-targets it or rejects it.
    # The platform shares Yewo's content-relative direction, so it passes through unchanged.
    assert params == {
        "left": 0,
        "top": 0,
        "width": 50,
        "height": 100,
        "direction": direction.value,
        "percent": 0.6,
    }


def test_android_scroll_without_a_platform_answer_is_not_repeated_as_a_drag() -> None:
    # UiAutomator performs the gesture whatever it replies. Reading a missing answer as a
    # declined gesture scrolled the region twice.
    driver = _DriverStub()
    driver.script_results["mobile: scrollGesture"] = None
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    assert interaction.scroll(ScrollDirection.DOWN) is None

    assert [script for script, _ in driver.scripts] == ["mobile: scrollGesture"]


def test_ios_scroll_falls_back_to_a_drag_because_the_platform_declines() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    assert interaction.scroll(ScrollDirection.DOWN) is None
    assert driver.scripts[-1][0] == "mobile: dragFromToForDuration"


def test_refresh_pulls_downward_from_the_top_of_the_region() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.IOS,
        _application(DevicePlatform.IOS),
        IosInteractionBackend(),
    )

    interaction.refresh(left=0, top=0, right=99, bottom=199)

    script, params = driver.scripts[-1]
    assert script == "mobile: dragFromToForDuration"
    # A pull starts near the top edge and travels down, the opposite of scrolling down.
    assert params["fromY"] < params["toY"]
    assert params["fromY"] == 10
    assert params["toY"] == 70
    assert params["fromX"] == params["toX"] == 24


def test_refresh_rejects_a_region_outside_the_screen() -> None:
    driver = _DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        _application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    with pytest.raises(DeviceInteractionError, match="outside"):
        interaction.refresh(left=0, top=0, right=10_000, bottom=10)

    assert driver.scripts == []
