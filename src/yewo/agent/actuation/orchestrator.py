import logging
import time
from typing import Optional

from yewo.agent.actuation.actions import (
    MAX_WAIT_SECONDS,
    MIN_WAIT_SECONDS,
    Action,
    ActionPlan,
    ActionType,
)
from yewo.agent.device.control import DeviceControl
from yewo.agent.device.input_adapter import DeviceInputAdapter
from yewo.agent.device.capability_controller import CapabilityController
from yewo.agent.device.receipts import Receipt
from yewo.agent.run.postconditions import TextEntryReceipt

logger = logging.getLogger(__name__)

MIN_LONG_PRESS_SECONDS = 0.1
MAX_LONG_PRESS_SECONDS = 10.0
ALLOWED_KEYCODES = {3, 4, 66, 67}


class ActionOrchestrator:
    def __init__(self, device_control: DeviceControl, scrubber=None):
        self.device_control = device_control
        self.input_adapter = DeviceInputAdapter(device_control)
        self.capability_controller = CapabilityController(device_control)
        self._scrubber = scrubber

    def execute(self, action: Action) -> Optional[Receipt | TextEntryReceipt]:
        """Execute a single Action object. Used by DeviceAgent.

        Returns:
            What the device reported about a capability this action set, or
            None for every other kind of action. The caller keeps it, because
            a scenario that declared a device state has to show the state was
            reached and not only asked for.
        """
        return self._run_action(action)

    def orchestrate(self, action_plan: ActionPlan) -> None:
        """Execute all actions in an ActionPlan. Kept for backward compatibility."""
        for action in action_plan.actions:
            self._run_action(action)

    def _reason(self, action: Action) -> str:
        reason = getattr(action, "reason", "")
        return self._scrubber.scrub(reason) if self._scrubber else reason

    def _validate_point(self, x: int, y: int, label: str) -> None:
        width = getattr(self.device_control, "screen_width", None)
        height = getattr(self.device_control, "screen_height", None)
        if x < 0 or y < 0:
            raise ValueError(f"{label} coordinates must be non-negative: ({x}, {y})")
        if width is not None and x >= width:
            raise ValueError(f"{label} x coordinate {x} is outside screen width {width}")
        if height is not None and y >= height:
            raise ValueError(f"{label} y coordinate {y} is outside screen height {height}")

    def _validate(self, action: Action) -> None:
        coordinates = getattr(action, "coordinates", None)
        if coordinates is not None:
            action_label = (
                action.type.value if isinstance(action.type, ActionType) else action.type
            )
            self._validate_point(coordinates.x, coordinates.y, action_label)
        if action.type == ActionType.DRAG_AND_DROP:
            self._validate_point(action.source_x, action.source_y, "drag source")
            self._validate_point(
                action.destination_x, action.destination_y, "drag destination"
            )
        elif action.type == ActionType.SWIPE:
            self._validate_point(action.start_x, action.start_y, "swipe start")
            self._validate_point(action.end_x, action.end_y, "swipe end")
        elif action.type == ActionType.LONG_PRESS and not (
            MIN_LONG_PRESS_SECONDS <= action.duration <= MAX_LONG_PRESS_SECONDS
        ):
            raise ValueError(
                f"long_press duration must be between {MIN_LONG_PRESS_SECONDS} "
                f"and {MAX_LONG_PRESS_SECONDS} seconds"
            )
        elif action.type == ActionType.KEY_EVENT and action.key not in ALLOWED_KEYCODES:
            raise ValueError(f"Unsupported key code: {action.key}")

    def _run_action(self, action: Action) -> Optional[Receipt | TextEntryReceipt]:
        if action.type == ActionType.TARGET_NOT_LISTED:
            raise ValueError(
                "target_not_listed reports a missing target and has nothing "
                "to execute. The caller stops the step and records it."
            )
        self._validate(action)
        match action.type:
            case ActionType.WAIT:
                seconds = min(max(action.duration, MIN_WAIT_SECONDS), MAX_WAIT_SECONDS)
                logger.info("Waiting %.1fs — %s", seconds, self._reason(action))
                time.sleep(seconds)
            case ActionType.TYPE:
                coordinates = action.coordinates
                logger.info(
                    "Executing action: %s at (%d, %d) — %s",
                    action.type, coordinates.x, coordinates.y, self._reason(action),
                )
                typed_text = (
                    self._scrubber.resolve_model_text(action.text)
                    if self._scrubber
                    else action.text
                )
                return self.input_adapter.enter_text(
                    typed_text, coordinates.x, coordinates.y
                )
            case ActionType.CLEAR:
                coordinates = action.coordinates
                logger.info(
                    "Executing action: %s at (%d, %d) — %s",
                    action.type, coordinates.x, coordinates.y, self._reason(action),
                )
                self.device_control.clear(coordinates.x, coordinates.y)
            case ActionType.TAP:
                coordinates = action.coordinates
                logger.info(
                    "Executing action: %s x%d at (%d, %d) — %s",
                    action.type, action.count, coordinates.x, coordinates.y,
                    self._reason(action),
                )
                # Counted here, where counting is arithmetic. Asked of a planner
                # across steps, it becomes a judgement about how many have
                # happened so far, from a screen that may not show it.
                for tap in range(action.count):
                    if tap and action.interval_ms:
                        time.sleep(action.interval_ms / 1000)
                    self.device_control.tap(
                        coordinates.x, coordinates.y, with_feedback=True
                    )
            case ActionType.DOUBLE_TAP:
                coordinates = action.coordinates
                logger.info(
                    "Executing action: %s at (%d, %d) — %s",
                    action.type, coordinates.x, coordinates.y, self._reason(action),
                )
                self.device_control.double_tap(coordinates.x, coordinates.y, with_feedback=True)
            case ActionType.LONG_PRESS:
                coordinates = action.coordinates
                logger.info(
                    "Executing action: %s at (%d, %d) duration=%s — %s",
                    action.type, coordinates.x, coordinates.y, action.duration, self._reason(action),
                )
                self.device_control.long_press(
                    coordinates.x, coordinates.y, action.duration
                )
            case ActionType.DRAG_AND_DROP:
                logger.info(
                    "Executing action: %s from (%d, %d) to (%d, %d) — %s",
                    action.type,
                    action.source_x, action.source_y,
                    action.destination_x, action.destination_y,
                    self._reason(action),
                )
                self.device_control.drag_and_drop(
                    action.source_x,
                    action.source_y,
                    action.destination_x,
                    action.destination_y,
                )
            case ActionType.KEY_EVENT:
                logger.info(
                    "Executing action: %s key=%s — %s",
                    action.type, action.key, self._reason(action),
                )
                self.device_control.key_event(action.key)
            case ActionType.SWIPE:
                logger.info(
                    "Executing action: %s from (%d, %d) to (%d, %d) — %s",
                    action.type,
                    action.start_x, action.start_y,
                    action.end_x, action.end_y,
                    self._reason(action),
                )
                self.device_control.swipe(
                    action.start_x, action.start_y, action.end_x, action.end_y
                )
            case ActionType.CAPABILITY:
                logger.info(
                    "Executing action: %s capability=%s — %s",
                    action.type, action.capability.kind.value, self._reason(action),
                )
                return self.capability_controller.execute(action.capability)
            case _:
                raise ValueError(f"Unsupported action type: {action.type}")
        time.sleep(1.5)
