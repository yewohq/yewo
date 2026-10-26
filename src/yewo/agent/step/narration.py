"""Saying what an action did, in the words the planner reads next.

Nothing here touches the device or the agent's state. An action, two perceptual
hashes and the scan metadata are enough to describe what happened. Keeping them
here lets the wording change without a reader opening the loop.
"""

import logging
from typing import Optional

from yewo.agent.actuation.actions import ActionType
from yewo.agent.memory.history import ActionResult
from yewo.agent.perception.elements import ScanInfo

logger = logging.getLogger(__name__)


def action_result(
    action,
    pre_phash: Optional[str],
    post_phash: Optional[str],
    error: Optional[str],
) -> ActionResult:
    if error:
        return ActionResult(
            success=False,
            observation=f"Action raised an error: {error[:200]}",
            error=error,
        )

    screen_changed = (
        (pre_phash != post_phash) if (pre_phash and post_phash) else None
    )
    t = action.type.value if hasattr(action.type, "value") else str(action.type)

    if screen_changed is None:
        return ActionResult(
            success=True, observation=f"{t} executed (screen state unknown)"
        )

    if screen_changed:
        return ActionResult(
            success=True,
            observation=f"{t} executed — screen changed (navigation or state update occurred)",
        )

    if t == ActionType.TYPE.value:
        coords = action.coordinates
        obs = (
            f"type executed at ({coords.x}, {coords.y}) — screen unchanged "
            f"(text may have been entered. A masked field cannot be confirmed by sight)"
        )
    elif t in (ActionType.TAP.value, ActionType.DOUBLE_TAP.value):
        obs = (
            f"{t} delivered to the device. The screen did not change. "
            "An unchanged screen is not evidence that the app ignored this input. "
            "The app may have changed state that is not visible."
        )
    elif t == ActionType.KEY_EVENT.value:
        obs = f"key_event {action.key} sent — screen unchanged"
    elif t == ActionType.SWIPE.value:
        obs = "swipe executed — screen unchanged (content may have scrolled without triggering navigation)"
    elif t == ActionType.CLEAR.value:
        obs = f"clear executed at ({action.coordinates.x}, {action.coordinates.y}) — screen unchanged"
    else:
        obs = f"{t} executed — screen unchanged"

    return ActionResult(success=True, observation=obs)


def reaction(
    scan_info: Optional[ScanInfo],
    prev_element_count: Optional[int],
    keyboard_open: Optional[bool] = None,
) -> str:
    """Describe what the screen actually did after the previous action, using perception metadata."""
    if scan_info is None or scan_info.reason == "first_pass":
        return ""

    if scan_info.reason == "stable":
        return "screen unchanged — coordinates remain valid"

    parts = []

    if scan_info.reason == "navigation":
        dist = scan_info.phash_distance
        parts.append(f"navigation detected (pHash Δ{dist})" if dist else "navigation detected")
    elif scan_info.reason == "layout_shift":
        match_pct = int(scan_info.match_rate * 100) if scan_info.match_rate is not None else "?"
        parts.append(f"layout shifted ({match_pct}% elements matched)")
    elif scan_info.reason == "ocr_refresh":
        parts.append("screen text updated")

    if prev_element_count is not None:
        delta = scan_info.element_count - prev_element_count
        if keyboard_open is not None and abs(delta) > 30:
            if keyboard_open:
                parts.append(f"keyboard opened (+{delta} elements, {prev_element_count}→{scan_info.element_count})")
            else:
                parts.append(f"keyboard dismissed ({delta} elements, {prev_element_count}→{scan_info.element_count})")
        elif delta != 0:
            parts.append(f"element count {prev_element_count}→{scan_info.element_count}")
    else:
        parts.append(f"{scan_info.element_count} elements")

    if scan_info.reason in ("navigation", "layout_shift"):
        parts.append("coordinates invalidated")

    return ", ".join(parts)


def describe(action) -> str:
    """Produce a concise, redacted human-readable action description for history."""

    t = action.type.value if hasattr(action.type, "value") else str(action.type)

    fields = action.model_dump()
    coords = fields.get("coordinates")
    mark = fields.get("mark")
    # An unresolved mark has no point to report, so name the number instead.
    where = (
        f"({coords['x']}, {coords['y']})"
        if coords is not None
        else f"element {mark if mark is not None else '?'}"
    )

    if t == ActionType.TYPE.value:
        desc = f"type {fields.get('text', '')!r} at {where}"
    elif t in (
        ActionType.TAP.value,
        ActionType.DOUBLE_TAP.value,
        ActionType.LONG_PRESS.value,
        ActionType.CLEAR.value,
    ):
        desc = f"{t} at {where}"
    elif t == ActionType.KEY_EVENT.value:
        desc = f"key_event {action.key}"
    elif t == ActionType.SWIPE.value:
        desc = f"swipe ({action.start_x},{action.start_y})→({action.end_x},{action.end_y})"
    elif t == ActionType.DRAG_AND_DROP.value:
        desc = f"drag ({action.source_x},{action.source_y})→({action.destination_x},{action.destination_y})"
    elif t == ActionType.CAPABILITY.value:
        desc = f"capability:{action.capability}"
    else:
        desc = t

    target = fields.get("target")
    if target and t in (
        ActionType.TYPE.value,
        ActionType.TAP.value,
        ActionType.DOUBLE_TAP.value,
        ActionType.LONG_PRESS.value,
        ActionType.CLEAR.value,
    ):
        desc += f" targeting {target!r}"

    reason = fields.get("reason", "")
    if reason:
        desc += f" (reason: {reason})"
    return desc
