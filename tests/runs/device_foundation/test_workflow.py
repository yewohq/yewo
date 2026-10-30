"""Tests for deterministic workflow execution and evidence order."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import create_autospec

import pytest

from yewo.applications import ApplicationDescriptor, ApplicationInstallMode, ApplicationRequest
from yewo.automation import AutomationSession
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
)
from yewo.devices.errors import (
    ApplicationFirstPaintError,
    DeviceActionRejectedError,
    DeviceInteractionError,
)
from yewo.devices.interaction import DeviceInteraction, ScrollDirection
from yewo.devices.logs import DeviceLogCaptureError
from yewo.devices.screen import ScreenCalibration
from yewo.runs import (
    RunExecutionContext,
    RunNarrator,
    RunRequest,
    RunStart,
    RunTerminalCategory,
    RunWorkspace,
)
from yewo.runs.device_foundation import (
    BackgroundAction,
    ClearTextAction,
    DeepLinkAction,
    DeviceFoundationActionKind,
    DeviceFoundationResultState,
    DeviceFoundationScript,
    DeviceFoundationWorkflow,
    DoubleTapAction,
    LongPressAction,
    NoArgumentAction,
    PointerMotionAction,
    PressKeyAction,
    ScrollAction,
    TapAction,
    TypeTextAction,
    WaitAction,
)
from yewo.runs.device_foundation.workflow import _dispatch
from yewo.runs.errors import RunWorkflowInterrupted


CAPTURE_TIME = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
SCRIPT_SHA256 = "a" * 64
DEVICE = DeviceDescriptor(
    platform=DevicePlatform.ANDROID,
    target_id="emulator-5554",
    connection_id="emulator-5554",
    display_name="Pixel",
    kind=DeviceKind.VIRTUAL,
    readiness=DeviceReadiness.READY,
    native_state="device",
)
APPLICATION = ApplicationDescriptor(
    platform=DevicePlatform.ANDROID,
    application_id="com.example.app",
)


class _InteractionStub:
    platform = DevicePlatform.ANDROID

    def __init__(
        self,
        *,
        reject_tap: bool = False,
        first_paint_error: Exception | None = None,
        screenshot_size: tuple[int, int] = (1080, 2400),
    ) -> None:
        self.reject_tap = reject_tap
        self.first_paint_error = first_paint_error
        self.screenshot_size = screenshot_size
        self.calls: list[tuple[object, ...]] = []

    def wait_for_first_paint(self) -> bytes:
        self.calls.append(("first_paint",))
        if self.first_paint_error is not None:
            raise self.first_paint_error
        return b"png"

    def tap(self, x: int, y: int, *, count: int, interval_seconds: float) -> None:
        self.calls.append(("tap", x, y, count, interval_seconds))
        if self.reject_tap:
            raise DeviceActionRejectedError("rejected")

    def type_text(self, x: int, y: int, text: str) -> str:
        self.calls.append(("type_text", x, y, text))
        return text

    def dismiss_keyboard(self) -> bool:
        self.calls.append(("dismiss_keyboard",))
        return True

    def screen_calibration(self) -> ScreenCalibration:
        width, height = self.screenshot_size
        return ScreenCalibration(width, height, width, height)


@dataclass(frozen=True)
class _ObservationStub:
    observation_id: str


class _ObservationsStub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = 0

    def capture(self, *, predecessor_observation_id: str | None = None) -> _ObservationStub:
        del predecessor_observation_id
        self.index += 1
        observation_id = f"observation_{self.index}"
        path = self.root / "assets/observations" / f"{observation_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return _ObservationStub(observation_id)


class _Signal:
    def __init__(self, interrupted: bool = False) -> None:
        self.interrupted = interrupted

    def is_set(self) -> bool:
        return self.interrupted


def _context(
    tmp_path: Path,
    drained: list[str],
    *,
    script_sha256: str = SCRIPT_SHA256,
) -> RunExecutionContext:
    request = RunRequest(
        output_root=tmp_path,
        workflow_input_path=Path("script.json"),
        workflow_input_sha256=script_sha256,
        device=DeviceQuery(platform=DevicePlatform.ANDROID, target_id="emulator-5554"),
        application=ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=ApplicationInstallMode.REUSE,
            application_id=APPLICATION.application_id,
        ),
    )
    start = RunStart(run_id="run_1", request=request, started_at=CAPTURE_TIME)
    session = SimpleNamespace(
        device=SimpleNamespace(descriptor=DEVICE),
        application=APPLICATION,
        drain_action_logs=drained.append,
    )
    return RunExecutionContext(
        workspace=RunWorkspace(tmp_path, start, RunNarrator(tmp_path)),
        session=cast(AutomationSession, session),
    )


def _script() -> DeviceFoundationScript:
    return DeviceFoundationScript(
        script_id="script_1",
        platform=DevicePlatform.ANDROID,
        application_id=APPLICATION.application_id,
        screenshot_width_px=1080,
        screenshot_height_px=2400,
        actions=(
            TapAction(action_id="tap_1", x=10, y=20),
            TypeTextAction(action_id="type_1", x=30, y=40, text="secret input"),
            NoArgumentAction(
                action_id="dismiss_1",
                kind=DeviceFoundationActionKind.DISMISS_KEYBOARD,
            ),
        ),
    )


def _workflow(
    tmp_path: Path,
    interaction: _InteractionStub,
    script: DeviceFoundationScript | None = None,
) -> DeviceFoundationWorkflow:
    times = iter(CAPTURE_TIME + timedelta(seconds=index) for index in range(10))
    return DeviceFoundationWorkflow(
        script or _script(),
        script_sha256=SCRIPT_SHA256,
        interaction_factory=lambda context: cast(DeviceInteraction, interaction),
        observation_factory=lambda device, context: _ObservationsStub(tmp_path),
        time_factory=times.__next__,
    )


def test_workflow_persists_ordered_receipts_without_text_payloads(tmp_path: Path) -> None:
    drained: list[str] = []
    interaction = _InteractionStub()

    outcome = _workflow(tmp_path, interaction).execute(
        _context(tmp_path, drained),
        _Signal(),
    )

    assert outcome.category is RunTerminalCategory.COMPLETED
    assert drained == ["tap_1", "type_1", "dismiss_1"]
    receipts = sorted((tmp_path / "assets/actions").glob("*.json"))
    assert len(receipts) == 3
    assert "secret input" not in "".join(path.read_text() for path in receipts)
    result = json.loads((tmp_path / "assets/device-foundation/result.json").read_text())
    assert result["state"] == DeviceFoundationResultState.COMPLETED
    assert result["script_sha256"] == SCRIPT_SHA256
    session_log = (tmp_path / "runtime/yewo-session.log").read_text().splitlines()
    assert all(" device-foundation: " in line for line in session_log)
    assert [line.split(" ", 2)[2] for line in session_log] == [
        "action[000] tap_1 (tap) -> completed",
        "action[001] type_1 (type_text) -> completed",
        "action[002] dismiss_1 (dismiss_keyboard) -> completed",
        "script result: completed (3/3 actions)",
    ]


def test_rejected_action_stops_the_script_after_persisting_evidence(tmp_path: Path) -> None:
    drained: list[str] = []

    outcome = _workflow(tmp_path, _InteractionStub(reject_tap=True)).execute(
        _context(tmp_path, drained),
        _Signal(),
    )

    assert outcome.category is RunTerminalCategory.INCOMPLETE
    assert drained == ["tap_1"]
    result = json.loads((tmp_path / "assets/device-foundation/result.json").read_text())
    assert result["state"] == DeviceFoundationResultState.INCOMPLETE
    assert result["rejected_action_id"] == "tap_1"
    session_log = (tmp_path / "runtime/yewo-session.log").read_text().splitlines()
    assert all(" device-foundation: " in line for line in session_log)
    assert [line.split(" ", 2)[2] for line in session_log] == [
        "action[000] tap_1 (tap) -> rejected",
        "script result: incomplete (0/3 actions)",
    ]


def test_blank_first_paint_is_an_application_failure(tmp_path: Path) -> None:
    interaction = _InteractionStub(
        first_paint_error=ApplicationFirstPaintError("application stayed blank")
    )

    outcome = _workflow(tmp_path, interaction).execute(
        _context(tmp_path, []),
        _Signal(),
    )

    assert outcome.category is RunTerminalCategory.APPLICATION_FAILURE
    assert not (tmp_path / "assets/observations").exists()


def test_script_for_another_application_is_blocked_before_interaction(
    tmp_path: Path,
) -> None:
    interaction = _InteractionStub()
    script = _script().model_copy(update={"application_id": "com.example.other"})

    outcome = _workflow(tmp_path, interaction, script).execute(
        _context(tmp_path, []),
        _Signal(),
    )

    assert outcome.category is RunTerminalCategory.BLOCKED_BY_POLICY
    assert interaction.calls == []


def test_script_for_another_screenshot_geometry_is_blocked(tmp_path: Path) -> None:
    interaction = _InteractionStub(screenshot_size=(720, 1600))

    outcome = _workflow(tmp_path, interaction).execute(
        _context(tmp_path, []),
        _Signal(),
    )

    assert outcome.category is RunTerminalCategory.BLOCKED_BY_POLICY
    assert interaction.calls == [("first_paint",)]


def test_script_source_must_match_the_run_record(tmp_path: Path) -> None:
    interaction = _InteractionStub()

    with pytest.raises(ValueError, match="does not identify"):
        _workflow(tmp_path, interaction).execute(
            _context(tmp_path, [], script_sha256="b" * 64),
            _Signal(),
        )

    assert interaction.calls == []


def test_first_paint_driver_failure_is_not_an_application_failure(tmp_path: Path) -> None:
    interaction = _InteractionStub(
        first_paint_error=DeviceInteractionError("screenshot capture failed")
    )

    with pytest.raises(DeviceInteractionError, match="screenshot capture failed"):
        _workflow(tmp_path, interaction).execute(
            _context(tmp_path, []),
            _Signal(),
        )


def test_log_receipt_persistence_failure_remains_infrastructure_failure(
    tmp_path: Path,
) -> None:
    class FailingDrains(list[str]):
        def append(self, value: str) -> None:
            del value
            raise DeviceLogCaptureError("receipt persistence failed")

    with pytest.raises(DeviceLogCaptureError, match="receipt persistence failed"):
        _workflow(tmp_path, _InteractionStub()).execute(
            _context(tmp_path, FailingDrains()),
            _Signal(),
        )


def test_interruption_before_an_action_does_not_dispatch_it(tmp_path: Path) -> None:
    interaction = _InteractionStub()

    with pytest.raises(RunWorkflowInterrupted):
        _workflow(tmp_path, interaction).execute(
            _context(tmp_path, []),
            _Signal(True),
        )

    assert interaction.calls == [("first_paint",)]
    assert not (tmp_path / "assets/actions").exists()


DISPATCH_CASES = (
    (TapAction(action_id="tap", x=1, y=2), "tap"),
    (DoubleTapAction(action_id="double", x=1, y=2), "double_tap"),
    (
        LongPressAction(action_id="long", x=1, y=2, duration_seconds=1),
        "long_press",
    ),
    (
        PointerMotionAction(
            action_id="drag",
            kind=DeviceFoundationActionKind.DRAG,
            start_x=1,
            start_y=2,
            end_x=3,
            end_y=4,
        ),
        "drag",
    ),
    (
        PointerMotionAction(
            action_id="swipe",
            kind=DeviceFoundationActionKind.SWIPE,
            start_x=1,
            start_y=2,
            end_x=3,
            end_y=4,
        ),
        "swipe",
    ),
    (ScrollAction(action_id="scroll", direction=ScrollDirection.UP), "scroll"),
    (TypeTextAction(action_id="type", x=1, y=2, text="value"), "type_text"),
    (ClearTextAction(action_id="clear", x=1, y=2), "clear_text"),
    (PressKeyAction(action_id="key", key="enter"), "press_key"),
    (WaitAction(action_id="wait", duration_seconds=0), "wait"),
    (
        BackgroundAction(action_id="background", duration_seconds=1),
        "background_and_resume",
    ),
    (DeepLinkAction(action_id="link", url="example://path"), "open_deep_link"),
    *tuple(
        (
            NoArgumentAction(action_id=f"action_{kind.value}", kind=kind),
            {
                DeviceFoundationActionKind.BACK: "back",
                DeviceFoundationActionKind.SUBMIT: "submit",
                DeviceFoundationActionKind.DISMISS_KEYBOARD: "dismiss_keyboard",
                DeviceFoundationActionKind.ACTIVATE: "activate",
                DeviceFoundationActionKind.TERMINATE: "terminate",
                DeviceFoundationActionKind.CLEAR_APPLICATION_DATA: "clear_application_data",
                DeviceFoundationActionKind.RELAUNCH: "relaunch",
                DeviceFoundationActionKind.RECOVER_TO_APPLICATION: "recover_to_application",
            }[kind],
        )
        for kind in (
            DeviceFoundationActionKind.BACK,
            DeviceFoundationActionKind.SUBMIT,
            DeviceFoundationActionKind.DISMISS_KEYBOARD,
            DeviceFoundationActionKind.ACTIVATE,
            DeviceFoundationActionKind.TERMINATE,
            DeviceFoundationActionKind.CLEAR_APPLICATION_DATA,
            DeviceFoundationActionKind.RELAUNCH,
            DeviceFoundationActionKind.RECOVER_TO_APPLICATION,
        )
    ),
)


@pytest.mark.parametrize(("action", "operation_name"), DISPATCH_CASES)
def test_dispatch_covers_each_declared_action(action, operation_name: str) -> None:
    interaction = create_autospec(DeviceInteraction, instance=True)
    interaction.platform = DevicePlatform.ANDROID
    interaction.dismiss_keyboard.return_value = True
    interaction.recover_to_application.return_value = True

    _dispatch(cast(DeviceInteraction, interaction), action)

    assert interaction.method_calls[0][0] == operation_name


def test_dispatch_cases_cover_the_complete_action_vocabulary() -> None:
    assert {action.kind for action, _ in DISPATCH_CASES} == set(DeviceFoundationActionKind)
