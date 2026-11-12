"""Action dispatch phase timing tests."""

from typing import cast

import pytest

from yewo.agent.actuation import ActionExecutor, ActionKind
from yewo.devices.errors import DeviceInteractionError
from yewo.observations import ObservationService, ScreenFingerprint
from yewo.observations.errors import ObservationCaptureError
from yewo.agent.runtime import dispatch_timing
from yewo.grounding import PixelPoint

from tests.agent.actuation.test_executor import RecordingDevice, authorization


def test_dispatch_timing_separates_three_pre_settlement_phases(monkeypatch) -> None:
    timestamps = iter((10.0, 10.25, 12.5))
    trace_lines: list[str] = []
    monkeypatch.setattr(dispatch_timing, "trace", trace_lines.append)
    action = authorization(ActionKind.FOCUS, target=PixelPoint(x=10, y=20))

    timed_dispatch = dispatch_timing.dispatch_authorized_action(
        ActionExecutor(RecordingDevice()),
        action,
        4,
        lambda: next(timestamps),
    )
    dispatch_timing.trace_settlement_start(4, timed_dispatch, 12.65)

    assert timed_dispatch.pre_dispatch_evidence_ms == 250
    assert timed_dispatch.device_dispatch_ms == 2_250
    assert trace_lines == [
        "step=4 pre_dispatch_evidence_ms=250.0 device_dispatch_ms=2250.0 "
        "post_dispatch_processing_ms=150.0"
    ]


def test_dispatch_timing_clamps_a_reversed_clock(monkeypatch) -> None:
    timestamps = iter((3.0, 2.0, 1.0))
    trace_lines: list[str] = []
    monkeypatch.setattr(dispatch_timing, "trace", trace_lines.append)
    action = authorization(ActionKind.TAP, target=PixelPoint(x=10, y=20))

    timed_dispatch = dispatch_timing.dispatch_authorized_action(
        ActionExecutor(RecordingDevice()),
        action,
        1,
        lambda: next(timestamps),
    )
    dispatch_timing.trace_settlement_start(1, timed_dispatch, 0.0)

    assert timed_dispatch.pre_dispatch_evidence_ms == 0
    assert timed_dispatch.device_dispatch_ms == 0
    assert trace_lines[0].endswith("post_dispatch_processing_ms=0.0")


class _LostDevice:
    """An observation source whose device drops out during the baseline capture."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def capture_screen_fingerprint(self) -> ScreenFingerprint:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        ObservationCaptureError("Capture failed."),
        DeviceInteractionError("Could not capture a device screenshot."),
    ],
    ids=["capture-error", "device-error"],
)
def test_a_lost_baseline_leaves_the_authorized_action_dispatched(
    error: Exception,
    monkeypatch,
) -> None:
    # The baseline only improves attribution. DeviceInteractionError and ObservationCaptureError
    # are sibling classes, so catching one let a device hiccup escape dispatch, fail the run as an
    # infrastructure failure, and discard an action the policy had already authorized.
    monkeypatch.setattr(dispatch_timing, "trace", lambda line: None)
    timestamps = iter((0.0, 1.0, 2.0))
    device = RecordingDevice()
    action = authorization(ActionKind.TAP, target=PixelPoint(x=10, y=20))

    timed_dispatch = dispatch_timing.dispatch_authorized_action(
        ActionExecutor(device),
        action,
        1,
        lambda: next(timestamps),
        cast(ObservationService, _LostDevice(error)),
    )

    assert timed_dispatch.baseline is None
    assert [name for name, _, _ in device.calls] == ["tap"]
