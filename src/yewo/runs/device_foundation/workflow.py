"""Execute a deterministic device script through existing device services."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from yewo.devices import DevicePlatform
from yewo.devices.errors import (
    ApplicationFirstPaintError,
    DeviceActionRejectedError,
)
from yewo.devices.interaction import AndroidKey, DeviceInteraction
from yewo.observations import (
    ObservationCapturePolicy,
    ObservationService,
    ObservationStore,
    RegionScreenshotRedactor,
    ScreenshotStore,
)
from yewo.runs import (
    InterruptionSignal,
    RunExecutionContext,
    RunSessionPolicy,
    RunTerminalCategory,
    RunWorkflowOutcome,
)
from yewo.runs.device_foundation.records import (
    BackgroundAction,
    ClearTextAction,
    DeepLinkAction,
    DeviceFoundationAction,
    DeviceFoundationActionKind,
    DeviceFoundationActionReceipt,
    DeviceFoundationActionStatus,
    DeviceFoundationResult,
    DeviceFoundationResultState,
    DeviceFoundationScript,
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
from yewo.runs.device_foundation.store import DeviceFoundationStore
from yewo.runs.errors import RunWorkflowInterrupted


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _ObservedScreen(Protocol):
    @property
    def observation_id(self) -> str: ...


class _ObservationCapture(Protocol):
    def capture(
        self,
        *,
        predecessor_observation_id: str | None = None,
    ) -> _ObservedScreen: ...


InteractionFactory = Callable[[RunExecutionContext], DeviceInteraction]
ObservationFactory = Callable[[DeviceInteraction, RunExecutionContext], _ObservationCapture]


_NARRATION_SOURCE = "device-foundation"


class DeviceFoundationWorkflow:
    """Capture before-and-after evidence for a bounded deterministic script."""

    def __init__(
        self,
        script: DeviceFoundationScript,
        *,
        script_sha256: str,
        observation_policy: ObservationCapturePolicy | None = None,
        interaction_factory: InteractionFactory | None = None,
        observation_factory: ObservationFactory | None = None,
        store_factory: Callable[[Path], DeviceFoundationStore] = DeviceFoundationStore,
        time_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._script = script
        self._script_sha256 = script_sha256
        self._redactor = RegionScreenshotRedactor(script.redaction_regions)
        self._observation_policy = observation_policy
        self._interaction_factory = interaction_factory or _interaction
        self._observation_factory = observation_factory or self._observations
        self._store_factory = store_factory
        self._time_factory = time_factory

    def session_policy(self) -> RunSessionPolicy:
        """Apply script masks before the first recording segment starts."""
        return RunSessionPolicy(
            recording_redaction_regions=self._script.redaction_regions,
        )

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
    ) -> RunWorkflowOutcome:
        """Execute each declared action once and stop at the first rejection."""
        session = context.session
        platform = session.device.descriptor.platform
        if platform is not self._script.platform:
            raise ValueError("The device-foundation script platform does not match the session.")
        if session.application.application_id != self._script.application_id:
            return RunWorkflowOutcome(
                RunTerminalCategory.BLOCKED_BY_POLICY,
                "The deterministic script does not target the active application.",
            )
        if context.workspace.start.request.workflow_input_sha256 != self._script_sha256:
            raise ValueError("The run request does not identify the loaded device script.")
        interaction = self._interaction_factory(context)
        try:
            interaction.wait_for_first_paint()
        except ApplicationFirstPaintError:
            return RunWorkflowOutcome(
                RunTerminalCategory.APPLICATION_FAILURE,
                "The application did not produce a usable first frame.",
            )
        calibration = interaction.screen_calibration()
        if (
            calibration.screenshot_width_px != self._script.screenshot_width_px
            or calibration.screenshot_height_px != self._script.screenshot_height_px
        ):
            return RunWorkflowOutcome(
                RunTerminalCategory.BLOCKED_BY_POLICY,
                "The deterministic script does not match the observed screenshot geometry.",
            )
        observations = self._observation_factory(interaction, context)
        store = self._store_factory(context.workspace.root)
        current = observations.capture()
        initial_observation_id = current.observation_id
        completed_action_ids: list[str] = []
        for index, action in enumerate(self._script.actions):
            if interruption.is_set():
                raise RunWorkflowInterrupted
            started_at = self._time_factory()
            status = DeviceFoundationActionStatus.COMPLETED
            try:
                _dispatch(interaction, action)
            except DeviceActionRejectedError:
                status = DeviceFoundationActionStatus.REJECTED
            after = observations.capture(
                predecessor_observation_id=current.observation_id,
            )
            receipt = DeviceFoundationActionReceipt(
                script_id=self._script.script_id,
                action_id=action.action_id,
                action_index=index,
                kind=action.kind,
                status=status,
                before_observation_id=current.observation_id,
                after_observation_id=after.observation_id,
                started_at=started_at,
                completed_at=self._time_factory(),
            )
            store.persist_action(receipt)
            context.workspace.narrator.record(
                _NARRATION_SOURCE,
                f"action[{index:03d}] {action.action_id} ({action.kind.value}) -> {status.value}",
            )
            session.drain_action_logs(action.action_id)
            current = after
            if status is DeviceFoundationActionStatus.REJECTED:
                result = self._result(
                    initial_observation_id,
                    current.observation_id,
                    tuple(completed_action_ids),
                    action.action_id,
                )
                store.persist_result(result)
                context.workspace.narrator.record(
                    _NARRATION_SOURCE,
                    f"script result: {result.state.value} "
                    f"({len(completed_action_ids)}/{len(self._script.actions)} actions)",
                )
                return RunWorkflowOutcome(
                    RunTerminalCategory.INCOMPLETE,
                    "The deterministic device-foundation script rejected an action.",
                )
            completed_action_ids.append(action.action_id)
        result = self._result(
            initial_observation_id,
            current.observation_id,
            tuple(completed_action_ids),
            None,
        )
        store.persist_result(result)
        context.workspace.narrator.record(
            _NARRATION_SOURCE,
            f"script result: {result.state.value} "
            f"({len(completed_action_ids)}/{len(self._script.actions)} actions)",
        )
        return RunWorkflowOutcome(
            RunTerminalCategory.COMPLETED,
            "The deterministic device-foundation script completed.",
        )

    def _observations(
        self,
        interaction: DeviceInteraction,
        context: RunExecutionContext,
    ) -> ObservationService:
        root = context.workspace.root
        session = context.session
        return ObservationService(
            interaction,
            session.device.descriptor,
            session.application,
            ScreenshotStore(root),
            ObservationStore(root),
            self._redactor,
            policy=self._observation_policy,
        )

    def _result(
        self,
        initial_observation_id: str,
        final_observation_id: str,
        completed_action_ids: tuple[str, ...],
        rejected_action_id: str | None,
    ) -> DeviceFoundationResult:
        return DeviceFoundationResult(
            script_id=self._script.script_id,
            script_sha256=self._script_sha256,
            platform=self._script.platform,
            state=(
                DeviceFoundationResultState.INCOMPLETE
                if rejected_action_id is not None
                else DeviceFoundationResultState.COMPLETED
            ),
            initial_observation_id=initial_observation_id,
            final_observation_id=final_observation_id,
            declared_action_count=len(self._script.actions),
            completed_action_ids=completed_action_ids,
            rejected_action_id=rejected_action_id,
        )


def _interaction(context: RunExecutionContext) -> DeviceInteraction:
    return DeviceInteraction.from_session(context.session)


def _dispatch(interaction: DeviceInteraction, action: DeviceFoundationAction) -> None:
    if isinstance(action, TapAction):
        interaction.tap(
            action.x,
            action.y,
            count=action.count,
            interval_seconds=action.interval_seconds,
        )
    elif isinstance(action, DoubleTapAction):
        interaction.double_tap(action.x, action.y)
    elif isinstance(action, LongPressAction):
        interaction.long_press(action.x, action.y, action.duration_seconds)
    elif isinstance(action, PointerMotionAction):
        operation = (
            interaction.drag
            if action.kind is DeviceFoundationActionKind.DRAG
            else interaction.swipe
        )
        operation(
            action.start_x,
            action.start_y,
            action.end_x,
            action.end_y,
            duration_seconds=action.duration_seconds,
        )
    elif isinstance(action, ScrollAction):
        interaction.scroll(
            action.direction,
            left=action.left,
            top=action.top,
            right=action.right,
            bottom=action.bottom,
        )
    elif isinstance(action, TypeTextAction):
        interaction.type_text(action.x, action.y, action.text)
    elif isinstance(action, ClearTextAction):
        interaction.clear_text(action.x, action.y)
    elif isinstance(action, PressKeyAction):
        _press_key(interaction, action.key)
    elif isinstance(action, WaitAction):
        interaction.wait(action.duration_seconds)
    elif isinstance(action, BackgroundAction):
        interaction.background_and_resume(action.duration_seconds)
    elif isinstance(action, DeepLinkAction):
        interaction.open_deep_link(action.url)
    else:
        _dispatch_no_argument(interaction, action)


def _press_key(interaction: DeviceInteraction, key: str) -> None:
    if interaction.platform is DevicePlatform.IOS:
        interaction.press_key(key)
        return
    interaction.press_key(AndroidKey.parse_name(key))


def _dispatch_no_argument(
    interaction: DeviceInteraction,
    action: NoArgumentAction,
) -> None:
    if action.kind is DeviceFoundationActionKind.BACK:
        interaction.back()
    elif action.kind is DeviceFoundationActionKind.SUBMIT:
        interaction.submit()
    elif action.kind is DeviceFoundationActionKind.DISMISS_KEYBOARD:
        if interaction.dismiss_keyboard() is not True:
            raise DeviceActionRejectedError("The requested verified interaction did not succeed.")
    elif action.kind is DeviceFoundationActionKind.ACTIVATE:
        interaction.activate()
    elif action.kind is DeviceFoundationActionKind.TERMINATE:
        interaction.terminate()
    elif action.kind is DeviceFoundationActionKind.CLEAR_APPLICATION_DATA:
        interaction.clear_application_data()
    elif action.kind is DeviceFoundationActionKind.RELAUNCH:
        interaction.relaunch()
    elif action.kind is DeviceFoundationActionKind.RECOVER_TO_APPLICATION:
        if interaction.recover_to_application() is not True:
            raise DeviceActionRejectedError("The requested verified interaction did not succeed.")
    else:
        raise ValueError("The no-argument action kind is unsupported.")
