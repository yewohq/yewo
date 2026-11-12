"""Measure runtime work between authorization and action settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.actuation import (
    ActionExecutor,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.agent.narration import trace, trace_dispatch
from yewo.agent.runtime.settlement import read_focused_field
from yewo.devices.errors import DeviceError
from yewo.devices.focus import FocusedFieldReading
from yewo.observations import ObservationService, ScreenFingerprint
from yewo.observations.errors import ObservationCaptureError
from yewo.run_narration import elapsed_milliseconds


@dataclass(frozen=True)
class TimedDispatch:
    """One receipt and the measured work that produced it."""

    receipt: DispatchReceipt
    before_focus: FocusedFieldReading | None
    baseline: ScreenFingerprint | None
    pre_dispatch_evidence_ms: float
    device_dispatch_ms: float
    dispatch_completed_at: float


def dispatch_authorized_action(
    executor: ActionExecutor,
    authorization: AuthorizedAction,
    step_index: int,
    monotonic: Callable[[], float],
    observations: ObservationService | None = None,
) -> TimedDispatch:
    """Dispatch once and measure evidence collection separately from device work."""
    evidence_started_at = monotonic()
    before_focus = read_focused_field(executor, authorization)
    baseline = _dispatch_baseline(observations, authorization)
    dispatch_started_at = monotonic()
    receipt = executor.dispatch(authorization)
    dispatch_completed_at = monotonic()
    trace_dispatch(step_index, receipt)
    return TimedDispatch(
        receipt=receipt,
        before_focus=before_focus,
        baseline=baseline,
        pre_dispatch_evidence_ms=_duration_ms(evidence_started_at, dispatch_started_at),
        device_dispatch_ms=_duration_ms(dispatch_started_at, dispatch_completed_at),
        dispatch_completed_at=dispatch_completed_at,
    )


def trace_settlement_start(
    step_index: int,
    dispatch: TimedDispatch,
    settlement_started_at: float,
) -> None:
    """Record the complete pre-settlement duration breakdown."""
    trace(
        f"step={step_index} pre_dispatch_evidence_ms={dispatch.pre_dispatch_evidence_ms:.1f} "
        f"device_dispatch_ms={dispatch.device_dispatch_ms:.1f} "
        f"post_dispatch_processing_ms="
        f"{_duration_ms(dispatch.dispatch_completed_at, settlement_started_at):.1f}"
    )


def _duration_ms(started_at: float, completed_at: float) -> float:
    return elapsed_milliseconds(started_at, completed_at)


def _dispatch_baseline(
    observations: ObservationService | None,
    authorization: AuthorizedAction,
) -> ScreenFingerprint | None:
    """Fingerprint the screen as it stands when the action is dispatched.

    Captured after any declared value is registered, so the baseline matches what the run
    this screen stays covered in the baseline.
    """
    if observations is None:
        return None
    try:
        return observations.capture_screen_fingerprint()
    except (ObservationCaptureError, DeviceError):
        # A baseline improves attribution and is never required. Losing the device mid-capture
        # must not discard an action the policy already authorized.
        trace(f"dispatch_baseline=unavailable authorization={authorization.authorization_id}")
        return None
