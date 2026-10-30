"""Tests for deterministic device-foundation records."""

from datetime import datetime, timezone

from pydantic import ValidationError
import pytest

from yewo.devices import DevicePlatform
from yewo.runs.device_foundation import (
    DeviceFoundationActionKind,
    DeviceFoundationActionReceipt,
    DeviceFoundationActionStatus,
    DeviceFoundationResult,
    DeviceFoundationResultState,
    DeviceFoundationScript,
    NoArgumentAction,
    PointerMotionAction,
    PressKeyAction,
    TapAction,
)


CAPTURE_TIME = datetime(2026, 10, 30, tzinfo=timezone.utc)


def test_script_rejects_repeated_action_ids() -> None:
    with pytest.raises(ValidationError, match="cannot repeat"):
        DeviceFoundationScript(
            script_id="script_1",
            platform=DevicePlatform.ANDROID,
            actions=(
                TapAction(action_id="action_1", x=1, y=2),
                TapAction(action_id="action_1", x=3, y=4),
            ),
        )


def test_receipt_rejects_completion_before_start() -> None:
    with pytest.raises(ValidationError, match="before it starts"):
        DeviceFoundationActionReceipt(
            script_id="script_1",
            action_id="action_1",
            action_index=0,
            kind=DeviceFoundationActionKind.TAP,
            status=DeviceFoundationActionStatus.COMPLETED,
            before_observation_id="observation_1",
            after_observation_id="observation_2",
            started_at=CAPTURE_TIME,
            completed_at=CAPTURE_TIME.replace(year=2025),
        )


def test_completed_result_accounts_for_every_declared_action() -> None:
    with pytest.raises(ValidationError, match="every declared action"):
        DeviceFoundationResult(
            script_id="script_1",
            platform=DevicePlatform.ANDROID,
            state=DeviceFoundationResultState.COMPLETED,
            initial_observation_id="observation_1",
            final_observation_id="observation_2",
            declared_action_count=2,
            completed_action_ids=("action_1",),
        )


def test_script_rejects_platform_incompatible_actions_before_device_start() -> None:
    with pytest.raises(ValidationError, match="Back navigation"):
        DeviceFoundationScript(
            script_id="script_1",
            platform=DevicePlatform.IOS,
            actions=(
                NoArgumentAction(
                    action_id="back_1",
                    kind=DeviceFoundationActionKind.BACK,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="at least 0.5"):
        DeviceFoundationScript(
            script_id="script_1",
            platform=DevicePlatform.IOS,
            actions=(
                PointerMotionAction(
                    action_id="swipe_1",
                    kind=DeviceFoundationActionKind.SWIPE,
                    start_x=1,
                    start_y=2,
                    end_x=3,
                    end_y=4,
                    duration_seconds=0.1,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="key name is unsupported"):
        DeviceFoundationScript(
            script_id="script_1",
            platform=DevicePlatform.ANDROID,
            actions=(PressKeyAction(action_id="key_1", key="volume_up"),),
        )
