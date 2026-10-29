"""
DevicePlanner — core device-control planning layer.
Builds prompts and calls the LLM to decide the next action.
No QA concerns: those live in yewo/testing/qa_planner.py.
"""

import logging
from typing import List, Optional

from yewo.agent.actuation.actions import (
    capability_vocabulary,
)
from yewo.agent.plan.schema import (  # noqa: F401 - re-exported for callers
    DescribeDeviceAction,
    DescribeStepResponse,
    DeviceAction,
    DeviceStepResponse,
    DoneAction,
    MarksDeviceAction,
    MarksStepResponse,
)
from yewo.agent.grounding import (  # noqa: F401 - re-exported for callers
    GROUNDING_COORDINATES,
    GROUNDING_DESCRIBE,
    GROUNDING_MARKS,
)
from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import Message, MultiPartMessage, SystemMessage
from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.device.platform import Platform
from yewo.utils.secrets import SECRET_MASK_GUIDANCE, SecretScrubber

logger = logging.getLogger(__name__)

COMMON_ACTIONS = """
- "drag_and_drop": Drag from source to destination. Fields: `type`, `source_x`, `source_y`, `destination_x`, `destination_y`, `reason`.
{key_event_action}
- "swipe": Swipe from start to end. Fields: `type`, `start_x`, `start_y`, `end_x`, `end_y`, `reason`.
- "wait": Let the screen finish loading or animating. Fields: `type`, `duration` (float seconds), `reason`. Use this instead of giving up when the screen looks empty or half-drawn.
- "capability": System-level control. Send exactly:
  {{"type": "capability", "capability": <one object below>, "reason": "..."}}
  Send no field an object does not list. A field the object does not name is
  refused, and the whole action with it.
  Available capabilities:
{capabilities}
- "done": Goal complete or execution must stop. Fields: `type`, `reason`.
"""
"""The actions every grounding mode offers, which name no target."""

DESCRIBE_ACTIONS = """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap something once, then observe the result. Fields: `type`, `target` (description), `reason`.
- "type": Type text into a field. Fields: `type`, `text`, `target` (description), `reason`.
- "clear": Clear a text field. Fields: `type`, `target` (description), `reason`.
- "double_tap": Double-tap something. Fields: `type`, `target` (description), `reason`.
- "long_press": Long press something. Fields: `type`, `duration` (float seconds), `target` (description), `reason`."""
MARKS_ACTIONS = """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap an element once, then observe the result. Fields: `type`, `mark` (int), `target` (description), `reason`.
- "type": Type text into an element. Fields: `type`, `text`, `mark` (int), `target` (description), `reason`.
- "clear": Clear a text field. Fields: `type`, `mark` (int), `target` (description), `reason`.
- "double_tap": Double-tap an element. Fields: `type`, `mark` (int), `target` (description), `reason`.
- "long_press": Long press an element. Fields: `type`, `duration` (float seconds), `mark` (int), `target` (description), `reason`.
- "show_keyboard_targets": Reveal detected visual keys for direct tapping. Use only when the goal tests the keyboard/keypad UI itself, never to enter ordinary text. Fields: `type`, `reason`.
- "target_not_listed": An app element you need is not numbered on the screenshot. Fields: `type`, `description` (what it looks like), `region` (one of upper_left, upper_center, upper_right, middle_left, middle_center, middle_right, lower_left, lower_center, lower_right), `intent` ("enter_text" if you need to put text into it, "activate" if you need to press it), `anchor_above` and `anchor_below` (the numbered visible controls immediately above and below the missing target), `reason`."""
COORDINATE_ACTIONS = """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap once at coordinates, then observe the result. Fields: `type`, `target` (description), `coordinates` as {"x": int, "y": int}, `reason`.
- "type": Type text into a field. Fields: `type`, `text`, `target` (description), `coordinates` as {"x": int, "y": int}, `reason`.
- "clear": Clear a text field. Fields: `type`, `target` (description), `coordinates` as {"x": int, "y": int}, `reason`.
- "double_tap": Double-tap. Fields: `type`, `target` (description), `coordinates` as {"x": int, "y": int}, `reason`.
- "long_press": Long press. Fields: `type`, `duration` (float seconds), `target` (description), `coordinates` as {"x": int, "y": int}, `reason`."""

DESCRIBE_CONSTRAINTS = """
CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. Nothing on this screen is listed or numbered. Name your target in `target`. Describe it as you would point it out: "the Search input field", "the Save button".
3. Describe the thing itself: the words on it, the symbol on it, its shape, its colour. Say where it sits if that helps — "in the top right".
4. Do not describe it by what is beside it. Naming a neighbour returns both: asking for the tick "opposite the back arrow" gives back the whole header bar instead of the tick.
5. Do not identify it by its position in a sequence. The grounder reads the picture, not a list. Say what makes it different: "the blue Add button with the plus symbol".
6. The target must be visible on the screen in front of you. Do not name something you remember from an earlier screen, and do not name something you expect to appear.
7. "type" focuses the field itself — do not spend a separate "tap" first. Only tap when you need focus without entering text.
8. {keyboard_recovery}
9. If a description was not found, describe the same thing by a different visible feature. Use its symbol, shape, colour, or words. Do not add details about its surroundings.
10. Do not repeat an action from history. Repeat it only when the outcome shows that it failed or was undone. Uncertain visual evidence does not prove failure.
11. A tap is exactly one tap. Observe the resulting screen before choosing another action.
12. If a goal requires an exact number of taps or clicks, count every tap recorded as delivered once, even when the screen stays unchanged. Do not add a replacement tap solely because a value or pixels repeated.
"""
MARKS_CONSTRAINTS = """
CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. Numbers identify visible detected regions. A number does not prove that a region is interactive. Name the number in `mark` and describe that same region in `target`. Never send coordinates — they are not accepted.
3. Use "type" and "clear" only on the actual editable field. A caption, summary value, read-only value, or label with the same words is not the field. If the field is hidden by the keyboard or is not numbered, use "target_not_listed". Never type into a same-named label as a substitute.
4. "type" focuses the element itself — do not spend a separate "tap" first. Only tap when you need focus without entering text.
5. Choose only a number that appears in the list below. If what you need is not numbered, use "target_not_listed" and describe it. Do not pick the nearest number instead, and do not guess.
6. {keyboard_recovery}
7. Do not repeat an action from history. Repeat it only when the outcome shows that it failed or was undone. Uncertain visual evidence does not prove failure.
8. A tap is exactly one tap. Observe the resulting screen before choosing another action.
9. The numbers are redrawn every step. Use the numbers from the CURRENT screenshot and list, never one you remember from an earlier step.
10. If a goal requires an exact number of taps or clicks, count every tap recorded as delivered once, even when the screen stays unchanged. Do not add a replacement tap solely because a value or pixels repeated.
"""
COORDINATE_CONSTRAINTS = """
CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. Every targeted action MUST provide `target`. Describe the control by its own visible text, symbol, shape, or colour.
3. For "type", you MUST provide `text`, `target`, AND `coordinates` of the target field.
4. "type" focuses the field at those coordinates itself — do not spend a separate "tap" first. Only tap when you need focus without entering text.
5. {keyboard_recovery}
6. Do not repeat an action from history. Repeat it only when the outcome shows that it failed or was undone. Uncertain visual evidence does not prove failure.
7. Read coordinates from the CURRENT screenshot. Never reuse coordinates from an older screenshot.
8. Re-read coordinates after any layout change. This includes keyboard changes, navigation, modals, and scrolling.
9. A tap is exactly one tap. Observe the resulting screen before choosing another action.
10. If a goal requires an exact number of taps or clicks, count every tap recorded as delivered once, even when the screen stays unchanged. Do not add a replacement tap solely because a value or pixels repeated.
"""

APP_LAUNCH_CONSTRAINTS = """
APP OPENING RULES:
1. On a device home screen or launcher, do not infer an app's identity from an icon that has no visible name.
2. On Android, open the app drawer first. Select the app there only when its visible name matches the requested app. Use app-drawer search only if the named app is not visible in the drawer.
3. On iOS, use Home Screen Search or the App Library when the requested app's name is not visible.
4. If an icon opens the wrong app, treat that icon as disproven for the requested app. Return to the launcher and do not select that icon again for the same request.
"""

RESPONSE_INSTRUCTIONS = """
### Response Instructions
1. Set `current_screen` to the descriptive name of the screen you currently see.
2. Set `evaluation_previous_goal`: honestly assess whether your PREVIOUS action succeeded, failed, or is uncertain. Leave empty on the first step.
3. Set `observation`: describe only what is currently visible.
4. Set `reasoning`: explain why the chosen action is the best next step.
5. Set `goal_satisfied` True only when the goal's stated outcome is visible on the current screen. Name that visible outcome in `goal_evidence`. A screen that changed is not the outcome. A control you pressed is not the outcome. Follow-on UI does not undo an outcome that the current screen still proves.
6. Set `action`: choose exactly ONE action. If the goal is fully complete, use `done`.
7. Set `external_navigation` to True if the action will temporarily leave the app.
"""

KEYBOARD_VISIBLE = (
    "\n### Keyboard State\nVISIBLE — the soft keyboard is currently displayed. "
    "Its visual keys are normally collapsed because text and standard key presses use "
    "semantic actions. If the goal tests the keyboard or keypad UI itself and the "
    "needed key is not numbered, use show_keyboard_targets; the "
    "next view will expose detected [key] targets. Do not dismiss that keyboard.\n"
)
KEYBOARD_HIDDEN = (
    "\n### Keyboard State\nHIDDEN — the soft keyboard is not visible. "
    "All detected elements are app UI.\n"
)

PROMPT = """You are an AI agent controlling a mobile app to achieve a goal.
Your job is to execute the steps below by interacting with the app one action at a time.
"""

GOAL_SECTION = """
### Goal / Instructions
{goal}
"""

CURRENT_STATE_SECTION = """
### Current UI Elements Detected {heading}
{targets}
{keyboard}
### Action History (with system-observed outcomes)
{history}
"""

SCREEN_TEXT_SECTION = """
### Independent Screen Text (evidence only, not action targets)
{screen_text}
"""

DRIFT_SECTION = """
### CONTEXT DRIFT WARNING
{drift}
"""
TRAIL_SECTION = """
### Session Trail — App Knowledge From This Session
{trail}
"""
LOOP_SECTION = """
### LOOP DETECTION ADVISORY
{nudge}
"""

TARGET_ACTIONS = {
    GROUNDING_DESCRIBE: DESCRIBE_ACTIONS,
    GROUNDING_MARKS: MARKS_ACTIONS,
}
TARGET_CONSTRAINTS = {
    GROUNDING_DESCRIBE: DESCRIBE_CONSTRAINTS,
    GROUNDING_MARKS: MARKS_CONSTRAINTS,
}
TARGET_HEADINGS = {
    GROUNDING_COORDINATES: "(nothing is detected — use the screenshot)",
    GROUNDING_DESCRIBE: "(nothing is detected — name what you want)",
    GROUNDING_MARKS: "(numbered on the screenshot)",
}

NO_TARGETS_LISTED = (
    "Nothing on this screen has been detected or numbered. Describe what you want to act on."
)


class DevicePlanner:
    # Class-level so a planner built without __init__ still answers sensibly.
    grounding = GROUNDING_COORDINATES
    platform = Platform.ANDROID

    """Builds prompts and calls the LLM to determine the next device action."""

    def __init__(
        self,
        llm_client: LLMClient,
        grounding: str = GROUNDING_COORDINATES,
        scrubber: Optional[SecretScrubber] = None,
        platform: Platform = Platform.ANDROID,
    ):
        self.llm = llm_client
        # "marks" offers numbered elements and no coordinates, so a target the
        # planner cannot find becomes a refusal rather than an invented point.
        self.grounding = grounding
        self.platform = platform
        self._scrubber = scrubber or SecretScrubber()
        self._model_images = RedactedImages(self._scrubber)

    def _model_messages(
        self,
        stable_prompt: str,
        current_state: str,
        image: bytes,
        gui_elements,
        ocr_evidence=(),
    ) -> list[Message]:
        """Keep the reusable prefix separate from changing state and pixels.

        Provider prompt caches operate on a prefix. A screenshot made its whole
        multipart message a changing cache block. Text before the image did not
        remain stable when it was in that message.
        """
        stable = SystemMessage(
            content=self._scrubber.for_model(
                f"{stable_prompt.rstrip()}\n\n### Yewo screenshot masks\n{SECRET_MASK_GUIDANCE}"
            )
        )
        current = MultiPartMessage()
        current.add_text(self._scrubber.for_model(current_state))
        current.add_text("Current screenshot (newest):")
        redaction_evidence = [*(gui_elements or []), *ocr_evidence]
        current.add_image(self._model_images.redact_model_image(image, redaction_evidence))
        return [stable, current]

    def plan_step(
        self,
        goal: List[str] | str,
        gui_elements=None,
        history: str = "",
        trail: Optional[str] = None,
        drift_warning: Optional[str] = None,
        loop_nudge: Optional[str] = None,
        current_image_bytes: bytes = b"",
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
        ocr_evidence=(),
    ) -> DeviceStepResponse:
        """Plan the next action given current screen state and accumulated history."""

        if isinstance(goal, str):
            goal_lines = [goal]
        else:
            goal_lines = goal or []

        stable_prompt, current_state = self._build_prompt_sections(
            goal_lines=goal_lines,
            gui_elements=gui_elements,
            history=history,
            trail=trail,
            drift_warning=drift_warning,
            loop_nudge=loop_nudge,
            keyboard_open=keyboard_open,
            marks=marks,
            screen_height=screen_height,
            screen_text=screen_text,
        )

        messages = self._model_messages(
            stable_prompt,
            current_state,
            current_image_bytes,
            gui_elements,
            ocr_evidence,
        )

        logger.info("Querying DevicePlanner for next step...")
        response: DeviceStepResponse = self.llm.generate_response(
            messages, self.step_response_model()
        )

        return response

    def _action_vocabulary(self) -> str:
        """The action schema both planners offer the model.

        Defined once: QAPlanner subclasses this planner, and when the two kept
        their own copies they drifted into contradicting each other.
        """
        targeting = TARGET_ACTIONS.get(self.grounding, COORDINATE_ACTIONS)
        key_event_action = ""
        if self.platform is Platform.ANDROID:
            key_event_action = (
                '- "key_event": Trigger an Android key event. Fields: `type`, '
                '`key` (4=back, 3=home, 66=enter, 67=backspace), `reason`.'
            )
        return targeting + COMMON_ACTIONS.format(
            capabilities=capability_vocabulary(),
            key_event_action=key_event_action,
        )

    def _action_constraints(self) -> str:
        """The rules for choosing an action. Shared with QAPlanner — see above."""
        targeting = TARGET_CONSTRAINTS.get(self.grounding, COORDINATE_CONSTRAINTS)
        if self.platform is Platform.ANDROID:
            recovery = (
                "Type directly into another input field without dismissing the keyboard. "
                "If the keyboard blocks a non-input control, report the target as not "
                "listed when that action is available, or scroll. Yewo uses the platform "
                "driver for keyboard recovery. Use key_event 4 only for deliberate "
                "navigation, never to dismiss the keyboard."
            )
        else:
            recovery = (
                "Type directly into another input field without dismissing the keyboard. "
                "If the keyboard blocks another control, report the target as not listed "
                "when that action is available, or scroll. Yewo uses the platform driver "
                "for keyboard recovery. Do not send Android keycodes."
            )
        return targeting.format(keyboard_recovery=recovery) + APP_LAUNCH_CONSTRAINTS

    def _build_prompt(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
    ) -> str:
        stable_prompt, current_state = self._build_prompt_sections(
            goal_lines=goal_lines,
            gui_elements=gui_elements,
            history=history,
            trail=trail,
            drift_warning=drift_warning,
            loop_nudge=loop_nudge,
            keyboard_open=keyboard_open,
            marks=marks,
            screen_height=screen_height,
            screen_text=screen_text,
        )
        return stable_prompt + current_state

    def _build_prompt_sections(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        keyboard = ""
        if keyboard_open is True:
            keyboard = KEYBOARD_VISIBLE
        elif keyboard_open is False:
            keyboard = KEYBOARD_HIDDEN

        stable_prompt = (
            PROMPT
            + self._action_vocabulary()
            + self._action_constraints()
            + RESPONSE_INSTRUCTIONS
            + GOAL_SECTION.format(goal="\n".join(f"- {step}" for step in goal_lines))
        )
        current_state = CURRENT_STATE_SECTION.format(
            heading=self._target_heading(),
            targets=self.format_targets(gui_elements, marks, screen_height),
            keyboard=keyboard,
            history=history or "No actions executed yet.",
        )
        if screen_text:
            current_state += self._format_screen_text(screen_text)
        if drift_warning:
            current_state += DRIFT_SECTION.format(drift=drift_warning)
        if trail:
            current_state += TRAIL_SECTION.format(trail=trail)
        if loop_nudge:
            current_state += LOOP_SECTION.format(nudge=loop_nudge)
        return stable_prompt, current_state

    ROW_BAND = 60
    DUPLICATE_OVERLAP = 0.9

    @staticmethod
    def _center(el) -> tuple[int, int]:
        box = el.bounding_box
        return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2

    @staticmethod
    def _overlap(a, b) -> float:
        """Intersection over union of two bounding boxes."""
        left = max(a.x_min, b.x_min)
        top = max(a.y_min, b.y_min)
        right = min(a.x_max, b.x_max)
        bottom = min(a.y_max, b.y_max)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        areas = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (
            b.y_max - b.y_min
        )
        union = areas - intersection
        return intersection / union if union else 0.0

    def _deduplicate(self, elements) -> list:
        """Drops repeat detections of the same thing, which only add noise."""
        kept = []
        for el in elements:
            if any(
                el.text == other.text
                and self._overlap(el.bounding_box, other.bounding_box) >= self.DUPLICATE_OVERLAP
                for other in kept
            ):
                continue
            kept.append(el)
        return kept

    ROW_LABEL_LIMIT = 3

    def _row_label(self, el, elements) -> Optional[str]:
        """The texts sharing a row with an element, naming which row it is in.

        List screens repeat the same label on every row, and coordinates alone
        are easy to mismatch by one row. All the row's text is offered rather
        than the nearest line. The closest line is often a subtitle, while the
        identifying name sits a few pixels further away.
        """
        _, cy = self._center(el)
        neighbours = []
        for other in elements:
            if other is el or not other.text or other.text == el.text:
                continue
            ox, oy = self._center(other)
            if abs(oy - cy) <= self.ROW_BAND:
                neighbours.append((abs(oy - cy), ox, other.text))
        if not neighbours:
            return None
        neighbours.sort()
        return ", ".join(f"'{text}'" for _, _, text in neighbours[: self.ROW_LABEL_LIMIT])

    def step_response_model(self):
        """The schema the model answers in, which differs by grounding mode."""
        if self.grounding == GROUNDING_DESCRIBE:
            return DescribeStepResponse
        if self.grounding == GROUNDING_MARKS:
            return MarksStepResponse
        return DeviceStepResponse

    def _target_heading(self) -> str:
        """Says where the targets are, which differs by mode and must not lie."""
        return TARGET_HEADINGS.get(self.grounding, "(YOLO/OCR)")

    @staticmethod
    def _format_screen_text(screen_text: tuple[str, ...]) -> str:
        return SCREEN_TEXT_SECTION.format(
            screen_text="\n".join(f"- {text}" for text in dict.fromkeys(screen_text))
        )

    def format_targets(self, elements, marks=None, screen_height: int = 0) -> str:
        """The element list for the prompt, in whichever form the mode calls for."""
        if self.grounding in (GROUNDING_COORDINATES, GROUNDING_DESCRIBE):
            return NO_TARGETS_LISTED
        if self.grounding == GROUNDING_MARKS:
            from yewo.agent.grounding import marks as marks_module

            return marks_module.describe(marks or [], screen_height)
        return self._format_elements(elements)

    def _format_elements(self, elements) -> str:
        if not elements:
            return "None detected."
        elements = self._deduplicate(elements)
        elements = sorted(
            elements,
            key=lambda el: (self._center(el)[1] // 25, self._center(el)[0]),
        )

        repeated = {
            el.text
            for el in elements
            if el.text and sum(1 for other in elements if other.text == el.text) > 1
        }

        lines = []
        for el in elements:
            cx, cy = self._center(el)
            if el.text:
                line = f"- {el.element} | text: '{el.text}' | center: ({cx}, {cy})"
                if el.text in repeated:
                    row = self._row_label(el, elements)
                    if row:
                        line += f" | in row with: {row}"
            else:
                line = f"- {el.element} | center: ({cx}, {cy})"
            lines.append(line)
        return "\n".join(lines)
