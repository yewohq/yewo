"""Everything that can stop an action before the device is touched.

Nothing here guesses. A gap we count is worth more than one we hide.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from yewo.agent.actuation.actions import (
    Action,
    ActionType,
)
from yewo.agent.memory.history import ActionEntry, ActionResult
from yewo.agent.memory.replay import target_label
from yewo.agent.grounding import GROUNDING_DESCRIBE
from yewo.agent.grounding.description import resolve_description
from yewo.agent.perception.elements import BoundingBox
from yewo.agent.grounding.marks import resolve_mark
from yewo.agent.step.staleness import has_target, target_moved
from yewo.errors import ObservationUnavailable
from yewo.agent.step import narration

logger = logging.getLogger(__name__)


def record_unexecuted(history, step: int, action_desc: str, observation: str) -> None:
    """Tells the planner why nothing happened, so it re-plans instead of repeating."""
    history.append(
        ActionEntry(
            step_num=step + 1,
            action_desc=action_desc,
            result=ActionResult(success=False, observation=observation),
        )
    )


def record_quarantined(ctx, step: int, action, snapshot) -> None:
    """Record a resolved action refused by the no-progress policy."""
    description = narration.describe(action)
    observation = (
        "The resolved control is quarantined after repeated attempts without "
        "durable progress. The action was not sent to the device."
    )
    record_unexecuted(ctx.history, step, description, observation)
    ctx.steps.record_blocked_action(
        action,
        description,
        observation,
        target_label(action, snapshot.gui_elements),
    )


def is_target_not_listed(action) -> bool:
    return getattr(action, "type", None) == ActionType.TARGET_NOT_LISTED


SCREEN_MOVED = (
    "Not executed — the screen changed while this action was being chosen, so "
    "its target was already out of date. Read the current elements and choose "
    "again."
)

SCREEN_UNREADABLE = (
    "Not executed — the screen could not be read again to confirm the target is "
    "still there. Look at the screen again before choosing."
)


def stale_target_reason(device_control, snapshot, action) -> Optional[str]:
    """Why the target is no longer trustworthy, or None while it still is.

    A refusal costs one step and one more look at the screen. That is cheaper
    than touching whatever took the target's place. A screen Yewo cannot read
    is no evidence that the target survived, so it counts too.
    """
    if not has_target(action):
        return None

    try:
        fresh = device_control.get_screenshot()
    except ObservationUnavailable as exc:
        logger.warning("Could not re-read the screen before acting: %s", exc)
        return SCREEN_UNREADABLE
    if not fresh:
        logger.warning("The screen re-read before acting came back empty.")
        return SCREEN_UNREADABLE
    if target_moved(snapshot.raw_img_bytes, fresh, action, snapshot.gui_elements):
        return SCREEN_MOVED
    return None


@dataclass(frozen=True)
class Decision:
    """Whether an action may touch the device, and what to say if not."""

    ok: bool = True
    observation: Optional[str] = None
    unmatched: bool = False
    """The planner asked for something no number names. The loop decides whether
    that ends the scenario, because only the loop knows if the screen changed
    since the last time it was asked."""

    target: Optional[str] = None
    """What the planner said it was looking for, when no number named it."""
    stale: bool = False
    """The screen under the target moved between planning and acting. Counted
    per control by the loop, because one control going stale repeatedly is a
    fact about that control, not about the step that met it."""
    action: Optional[Action] = None
    """The action to run instead, once a number has been turned into a point."""
    target_bounds: Optional[BoundingBox] = None
    """The grounded target box, available when describe mode placed it."""


USABLE = Decision()


def stalled_reason(decision) -> str:
    """Why a screen that keeps refusing ends the scenario.

    Named from what the last refusal was, so the report says which wall the
    run hit rather than only that it stopped.
    """
    if decision.target:
        return (
            "Target was not among the detected elements, on a screen already "
            f"searched for it: {decision.target}"
        )
    return (
        "No action reached the screen after repeated attempts. Last refusal: "
        f"{decision.observation}"
    )


def evaluate(
    action, snapshot, device_control, grounding: str = "", grounder=None
) -> Decision:
    """Everything that can stop an action before the device is touched."""
    if is_target_not_listed(action):
        logger.warning(
            "Target not among the %d detected elements: %s",
            len(snapshot.marks),
            action.description,
        )
        return Decision(
            ok=False,
            observation=(
                f"Not executed. No number names this: {action.description}. Only "
                "what is on screen gets a number. A target below the fold has no "
                "number. A target behind the keyboard has no number. Scroll it "
                "into view. If a keyboard is visible, Yewo will dismiss it once "
                "with the platform driver. Then look again. Answer "
                "target_not_listed again if it is still absent."
            ),
            unmatched=True,
            target=action.description,
        )

    if grounding == GROUNDING_DESCRIBE:
        unresolved, resolved, target_bounds = resolve_description(
            action, grounder, snapshot.raw_img_bytes
        )
    else:
        unresolved, resolved = resolve_mark(action, snapshot.marks, grounding)
        target_bounds = None
    if unresolved:
        return Decision(
            ok=False,
            observation=unresolved,
            unmatched=grounding == GROUNDING_DESCRIBE,
        )
    action = resolved or action

    stale = stale_target_reason(device_control, snapshot, action)
    if stale:
        return Decision(ok=False, observation=stale, stale=True, action=action)

    return Decision(action=action, target_bounds=target_bounds)
