"""Single-dispatch executor tests."""

from typing import Any

import pytest
from pydantic import ValidationError

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionExecutor,
    ActionKind,
    ActionProposal,
    ActionSource,
    AuthorizationReplayError,
    AuthorizedAction,
    DeclaredValueReference,
    DispatchDisposition,
    LiteralText,
    ScrollDistance,
    TextDeliveryStatus,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.devices.errors import DeviceActionRejectedError
from yewo.devices.focus import FocusedFieldReading
from yewo.devices.interaction import ScrollDirection
from yewo.grounding import PixelPoint
from yewo.observations import PixelRegion


class RecordingDevice:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.rejection: str | None = None

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self.rejection is not None:
            raise DeviceActionRejectedError(self.rejection)
        self.calls.append((name, args, kwargs))

    def tap(self, x: int, y: int, *, count: int = 1, interval_seconds: float = 0.1) -> None:
        self._record("tap", x, y, count=count, interval_seconds=interval_seconds)

    def focus(self, x: int, y: int) -> object:
        self._record("focus", x, y)
        return object()

    def type_focused_text(self, text: str) -> bool | None:
        self._record("type_focused_text", text)
        return True

    def clear_focused_text(self) -> bool | None:
        self._record("clear_focused_text")
        return True

    def double_tap(self, x: int, y: int) -> None:
        self._record("double_tap", x, y)

    def long_press(self, x: int, y: int, duration_seconds: float) -> None:
        self._record("long_press", x, y, duration_seconds)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 1,
    ) -> None:
        self._record("drag", start_x, start_y, end_x, end_y, duration_seconds=duration_seconds)

    def scroll(
        self,
        direction: ScrollDirection,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
        distance_fraction: float = 0.6,
    ) -> None:
        self._record(
            "scroll",
            direction,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            distance_fraction=distance_fraction,
        )

    def back(self) -> None:
        self._record("back")

    def dismiss_keyboard(self) -> bool:
        self._record("dismiss_keyboard")
        return True

    def wait(self, duration_seconds: float) -> None:
        self._record("wait", duration_seconds)

    def focused_field_reading(self) -> FocusedFieldReading:
        return FocusedFieldReading(
            is_focused=True,
            driver_bounds=(0, 0, 100, 100),
            keyboard_visible=True,
        )


def authorization(action: ActionKind, **updates: object) -> AuthorizedAction:
    values: dict[str, object] = {
        "authorization_id": "authorization-1",
        "proposal_id": "proposal-1",
        "source_observation_id": "obs_1",
        "source_screenshot_sha256": "a" * 64,
        "application_id": "com.example.app",
        "action": action,
        "source": ActionSource.PLANNER,
        "side_effect": ACTION_DEFINITIONS[action].side_effect,
        "expected_outcome": "The expected visible change occurs.",
        "sensitivity": ArtifactSensitivity.DEVICE_CONTENT,
        "retention": ArtifactRetention.RUN,
    }
    values.update(updates)
    return AuthorizedAction.model_validate(values)


def test_targeted_type_focuses_then_enters_text_in_one_dispatch() -> None:
    device = RecordingDevice()
    executor = ActionExecutor(device)

    executor.dispatch(
        authorization(
            ActionKind.TYPE,
            target=PixelPoint(x=10, y=20),
            text=LiteralText(value="42"),
        )
    )

    assert [call[0] for call in device.calls] == ["focus", "type_focused_text"]
    assert device.calls[0][1] == (10, 20)
    assert device.calls[1][1] == ("42",)


@pytest.mark.parametrize(
    ("authorized", "method"),
    [
        (authorization(ActionKind.TAP, target=PixelPoint(x=10, y=20)), "tap"),
        (authorization(ActionKind.FOCUS, target=PixelPoint(x=10, y=20)), "focus"),
        (
            authorization(
                ActionKind.TYPE,
                text=LiteralText(value="hello"),
            ),
            "type_focused_text",
        ),
        (authorization(ActionKind.CLEAR), "clear_focused_text"),
        (
            authorization(ActionKind.DOUBLE_TAP, target=PixelPoint(x=10, y=20)),
            "double_tap",
        ),
        (
            authorization(
                ActionKind.LONG_PRESS,
                target=PixelPoint(x=10, y=20),
                duration_seconds=1.5,
            ),
            "long_press",
        ),
        (
            authorization(
                ActionKind.DRAG,
                target=PixelPoint(x=10, y=20),
                destination=PixelPoint(x=30, y=40),
                duration_seconds=1.5,
            ),
            "drag",
        ),
        (
            authorization(
                ActionKind.SCROLL,
                region=PixelRegion(left=10, top=20, right=90, bottom=180),
                direction=ScrollDirection.DOWN,
                distance=ScrollDistance.LONG,
            ),
            "scroll",
        ),
        (authorization(ActionKind.BACK), "back"),
        (authorization(ActionKind.DISMISS_KEYBOARD), "dismiss_keyboard"),
        (authorization(ActionKind.WAIT, duration_seconds=0.5), "wait"),
    ],
)
def test_dispatches_exactly_one_mapped_device_operation(
    authorized: AuthorizedAction, method: str
) -> None:
    device = RecordingDevice()

    receipt = ActionExecutor(device, lambda: "receipt-1").dispatch(authorized)

    assert receipt.disposition is DispatchDisposition.DISPATCHED
    assert [call[0] for call in device.calls] == [method]


def test_scroll_maps_region_and_distance() -> None:
    device = RecordingDevice()
    action = authorization(
        ActionKind.SCROLL,
        region=PixelRegion(left=10, top=20, right=90, bottom=180),
        direction=ScrollDirection.UP,
        distance=ScrollDistance.LONG,
    )

    ActionExecutor(device).dispatch(action)

    assert device.calls == [
        (
            "scroll",
            (ScrollDirection.UP,),
            {"left": 10, "top": 20, "right": 89, "bottom": 179, "distance_fraction": 0.85},
        )
    ]


def test_repeated_tap_is_one_bounded_device_instruction() -> None:
    device = RecordingDevice()
    action = authorization(
        ActionKind.TAP,
        source=ActionSource.DETERMINISTIC_TEST,
        target=PixelPoint(x=10, y=20),
        tap_count=4,
        tap_interval_seconds=0.1,
    )

    ActionExecutor(device).dispatch(action)

    assert device.calls == [("tap", (10, 20), {"count": 4, "interval_seconds": 0.1})]


def test_declared_value_is_resolved_only_during_dispatch() -> None:
    secret = "correct horse battery staple"
    device = RecordingDevice()
    action = authorization(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="account_password"),
        sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
    )

    receipt = ActionExecutor(
        device,
        declared_value_resolver=lambda name: secret if name == "account_password" else "",
    ).dispatch(action)

    assert device.calls == [("type_focused_text", (secret,), {})]
    assert secret not in action.model_dump_json()
    assert secret not in receipt.model_dump_json()


def test_text_delivery_comparison_returns_only_secret_free_status() -> None:
    secret = "correct horse battery staple"
    device = RecordingDevice()
    action = authorization(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="account_password"),
        sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
    )
    executor = ActionExecutor(device, declared_value_resolver=lambda unused: secret)

    executor.dispatch(action)
    delivery = executor.verify_text_delivery(action)

    assert delivery is TextDeliveryStatus.MATCHED
    assert secret not in delivery.value
    assert executor.verify_text_delivery(action) is TextDeliveryStatus.UNAVAILABLE


def test_pending_text_can_be_discarded_without_device_comparison() -> None:
    device = RecordingDevice()
    action = authorization(
        ActionKind.TYPE,
        text=LiteralText(value="temporary text"),
    )
    executor = ActionExecutor(device)

    executor.dispatch(action)
    executor.discard_text_delivery(action)

    assert executor.verify_text_delivery(action) is TextDeliveryStatus.UNAVAILABLE
    assert [call[0] for call in device.calls] == ["type_focused_text"]


def test_missing_declared_value_resolver_returns_safe_refusal() -> None:
    action = authorization(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="account_password"),
        sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
    )

    receipt = ActionExecutor(RecordingDevice()).dispatch(action)

    assert receipt.disposition is DispatchDisposition.DEVICE_REFUSED
    assert "account_password" not in receipt.model_dump_json()


def test_refusal_receipt_does_not_retry_or_retain_driver_detail() -> None:
    device = RecordingDevice()
    device.rejection = "private field value"
    executor = ActionExecutor(device, lambda: "receipt-1")
    action = authorization(ActionKind.TAP, target=PixelPoint(x=10, y=20))

    receipt = executor.dispatch(action)

    assert receipt.disposition is DispatchDisposition.DEVICE_REFUSED
    assert receipt.reason == "The device rejected the authorized action."
    assert "private field value" not in receipt.model_dump_json()
    with pytest.raises(AuthorizationReplayError):
        executor.dispatch(action)


def test_executor_rejects_proposal_without_authorization() -> None:
    executor = ActionExecutor(RecordingDevice())
    raw_proposal = ActionProposal(
        proposal_id="proposal-1",
        source_observation_id="obs_1",
        action=ActionKind.BACK,
        expected_outcome="The previous screen appears.",
        reason="The previous screen is the required destination.",
    )

    with pytest.raises(TypeError, match="AuthorizedAction"):
        executor.dispatch(raw_proposal)  # type: ignore[arg-type]


def test_malformed_authorization_fails_validation() -> None:
    with pytest.raises(ValidationError):
        authorization(ActionKind.TAP)
