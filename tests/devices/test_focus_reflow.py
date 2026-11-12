"""Focused-field behavior during keyboard layout changes."""

from devices._interaction_stubs import DriverStub
from devices._interaction_stubs import application
from yewo.devices import DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.interaction import DeviceInteraction
from support import ClockStub


def test_focus_accepts_the_clicked_field_after_keyboard_reflow() -> None:
    driver = DriverStub()
    driver.element.focused = False
    clock = ClockStub()

    def focus_and_reflow() -> None:
        driver.element.clicked = True
        driver.element.focused = True
        driver.element.rect = {"x": 0, "y": 120, "width": 50, "height": 100}
        driver.keyboard = True

    driver.element.click = focus_and_reflow
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        focus_timeout_seconds=8,
        focus_poll_seconds=0.05,
        clock=clock,
    )

    focused = interaction.focus(20, 30)

    assert focused is driver.element
    assert driver.element.clicked
    assert clock.now == 0
