"""Executor integration with the real Android and iOS interaction adapters."""

from typing import Any

import pytest

from devices._interaction_stubs import DriverStub, application
from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionExecutor,
    ActionKind,
    ActionSource,
    AuthorizedAction,
    LiteralText,
    ScrollDistance,
)
from yewo.artifacts import ArtifactRetention
from yewo.devices import DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.interaction import DeviceInteraction, ScrollDirection
from yewo.devices.ios.interaction import IosInteractionBackend
from yewo.grounding import PixelPoint


def authorized(action: ActionKind, identifier: str, **updates: object) -> AuthorizedAction:
    values: dict[str, object] = {
        "authorization_id": identifier,
        "proposal_id": f"proposal-{identifier}",
        "source_observation_id": "obs_1",
        "source_screenshot_sha256": "a" * 64,
        "application_id": "com.example.app",
        "action": action,
        "source": ActionSource.PLANNER,
        "side_effect": ACTION_DEFINITIONS[action].side_effect,
        "expected_outcome": "The expected visible change occurs.",
        "retention": ArtifactRetention.RUN,
    }
    values.update(updates)
    return AuthorizedAction.model_validate(values)


@pytest.mark.parametrize(
    ("platform", "backend", "expected_scripts"),
    [
        (
            DevicePlatform.ANDROID,
            AndroidInteractionBackend(),
            [
                "mobile: clickGesture",
                "mobile: doubleClickGesture",
                "mobile: longClickGesture",
                "mobile: dragGesture",
                # UiAutomator performs the scroll itself, so the drag fallback does not run.
                "mobile: scrollGesture",
            ],
        ),
        (
            DevicePlatform.IOS,
            IosInteractionBackend(),
            [
                "mobile: tap",
                "mobile: doubleTap",
                "mobile: touchAndHold",
                "mobile: dragFromToForDuration",
                "mobile: dragFromToForDuration",
            ],
        ),
    ],
)
def test_gesture_dispatch_reaches_each_platform_adapter(
    platform: DevicePlatform,
    backend: Any,
    expected_scripts: list[str],
) -> None:
    driver = DriverStub()
    interaction = DeviceInteraction(driver, platform, application(platform), backend)
    executor = ActionExecutor(interaction)
    point = PixelPoint(x=40, y=80)

    actions = (
        authorized(ActionKind.TAP, "auth-tap", target=point),
        authorized(ActionKind.DOUBLE_TAP, "auth-double", target=point),
        authorized(
            ActionKind.LONG_PRESS,
            "auth-long",
            target=point,
            duration_seconds=1,
        ),
        authorized(
            ActionKind.DRAG,
            "auth-drag",
            target=point,
            destination=PixelPoint(x=80, y=160),
            duration_seconds=1,
        ),
        authorized(
            ActionKind.SCROLL,
            "auth-scroll",
            direction=ScrollDirection.UP,
            distance=ScrollDistance.SHORT,
        ),
    )
    for action in actions:
        executor.dispatch(action)

    gesture_scripts = [script for script, _ in driver.scripts if script in expected_scripts]
    assert gesture_scripts == expected_scripts


@pytest.mark.parametrize("platform", [DevicePlatform.ANDROID, DevicePlatform.IOS])
def test_text_and_keyboard_dispatch_use_platform_interaction(platform: DevicePlatform) -> None:
    driver = DriverStub()
    backend = (
        AndroidInteractionBackend()
        if platform is DevicePlatform.ANDROID
        else IosInteractionBackend()
    )
    interaction = DeviceInteraction(driver, platform, application(platform), backend)
    executor = ActionExecutor(interaction)
    point = PixelPoint(x=40, y=80)

    focused = executor.dispatch(authorized(ActionKind.FOCUS, "auth-focus", target=point))
    typed = executor.dispatch(
        authorized(
            ActionKind.TYPE,
            "auth-type",
            text=LiteralText(value="hello"),
        )
    )
    executor.dispatch(authorized(ActionKind.CLEAR, "auth-clear"))
    driver.keyboard = True
    dismissed = executor.dispatch(authorized(ActionKind.DISMISS_KEYBOARD, "auth-keyboard"))

    assert focused.disposition.value == "dispatched"
    assert typed.disposition.value == "dispatched"
    assert driver.element.cleared
    assert dismissed.disposition.value == "dispatched"
    assert driver.keyboard is False


def test_back_dispatch_uses_android_navigation() -> None:
    driver = DriverStub()
    interaction = DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
    )

    ActionExecutor(interaction).dispatch(authorized(ActionKind.BACK, "auth-back"))

    assert driver.back_calls == 1
