"""Grounding by description: the planner names a target, a model places it."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from yewo.agent.actuation.actions import TARGETED_ACTIONS, Action, Coordinates

from yewo.agent.grounding.placement import (
    MAX_SCREEN_FRACTION,
    MIN_PRESENCE,
    Placement,
)
from yewo.agent.perception.elements import BoundingBox

logger = logging.getLogger(__name__)

SCREEN_MISMATCH = "the saved screen description does not match"


@dataclass(frozen=True)
class ReboundTarget:
    """A replay target accepted together with its saved screen contract."""

    placement: Placement
    screen_score: float
    uniqueness_score: float


class DecisionGrounder:
    """A grounder backed by a decision model.

    One request asks two questions: where the target is, and whether it is
    there at all. The frame costs the same either way. The second question
    decides. The first one never does.
    """

    def __init__(self, client, min_presence: float = MIN_PRESENCE):
        self.client = client
        self.min_presence = min_presence

    def ground(self, screenshot: bytes, description: str) -> Optional[Placement]:
        from yewo.llm.jev_client import device_size, to_device

        # Plain mappings, so this seam needs nothing from the vendor package.
        response = self.client.ask(
            {
                "where": {"type": "box", "instructions": description},
                "present": {
                    "type": "noul",
                    "instructions": f"Is this visible in the image: {description}",
                },
            },
            screenshot=screenshot,
        )
        edges = response.answers["where"].box
        box_confidence = response.answers["where"].confidence
        present = response.answers["present"].noul

        fraction = (edges.right - edges.left) * (edges.bottom - edges.top)
        if present < self.min_presence or fraction > MAX_SCREEN_FRACTION:
            logger.info(
                "Could not place %r (present %.2f, covering %.0f%% of the screen)",
                description,
                present,
                fraction * 100,
            )
            return None

        x_min, y_min, x_max, y_max = to_device(
            device_size(screenshot), edges.left, edges.top, edges.right, edges.bottom
        )
        logger.info(
            "Grounded %r at (%d, %d, %d, %d), box confidence %s",
            description,
            x_min,
            y_min,
            x_max,
            y_max,
            box_confidence,
        )
        return Placement(
            box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            present=present,
            box_confidence=box_confidence,
        )

    def rebind(
        self, screenshot: bytes, screen_description: str, target_description: str
    ) -> tuple[Optional[ReboundTarget], Optional[str]]:
        """Place a cached target only when its screen and identity both agree."""
        from yewo.llm.jev_client import device_size, to_device

        response = self.client.ask(
            {
                "right_screen": {
                    "type": "noul",
                    "instructions": (
                        "Does this image show the same app screen and interaction "
                        f"state as this saved description: {screen_description}"
                    ),
                },
                "present": {
                    "type": "noul",
                    "instructions": (f"Is this target visible in the image: {target_description}"),
                },
                "unique": {
                    "type": "noul",
                    "instructions": (
                        "Is exactly one visible UI element a plausible match for: "
                        f"{target_description}"
                    ),
                },
                "where": {"type": "box", "instructions": target_description},
            },
            screenshot=screenshot,
        )
        screen_score = response.answers["right_screen"].noul
        present_score = response.answers["present"].noul
        uniqueness_score = response.answers["unique"].noul
        edges = response.answers["where"].box
        fraction = (edges.right - edges.left) * (edges.bottom - edges.top)
        if screen_score < self.min_presence:
            return None, SCREEN_MISMATCH
        if present_score < self.min_presence:
            return None, "the saved target is not visible"
        if uniqueness_score < self.min_presence:
            return None, "the saved target is not unique"
        if fraction > MAX_SCREEN_FRACTION:
            return None, "the saved target box covers too much of the screen"

        x_min, y_min, x_max, y_max = to_device(
            device_size(screenshot), edges.left, edges.top, edges.right, edges.bottom
        )
        placement = Placement(
            box=BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            ),
            present=present_score,
            box_confidence=response.answers["where"].confidence,
        )
        return ReboundTarget(
            placement=placement,
            screen_score=screen_score,
            uniqueness_score=uniqueness_score,
        ), None

    def matches_screen(self, screenshot: bytes, screen_description: str) -> bool:
        """Whether a targetless cached action starts on its saved screen."""
        response = self.client.ask(
            {
                "right_screen": {
                    "type": "noul",
                    "instructions": (
                        "Does this image show the same app screen and interaction "
                        f"state as this saved description: {screen_description}"
                    ),
                }
            },
            screenshot=screenshot,
        )
        return response.answers["right_screen"].noul >= self.min_presence


NOT_FOUND = (
    "Not executed. Nothing on this screen was found for: {target}. Describe the "
    "same thing by a different feature of itself — the words on it, its symbol, "
    "its shape, its colour. A target below the fold cannot be found. A target "
    "behind the keyboard cannot be found. Scroll it into view, or close the "
    "keyboard with key_event 4, then look again."
)


def resolve_description(
    action, grounder, screenshot: bytes
) -> Tuple[Optional[str], Optional[Action], Optional[BoundingBox]]:
    """Turns a named target into the point to touch.

    Returns (reason the step cannot proceed, action to run instead), which is
    what resolve_mark returns for the same job done by number.
    """
    if action.type not in TARGETED_ACTIONS:
        return None, None, None

    target = getattr(action, "target", None)
    if target is None:
        return (
            (
                "Not executed. This action needs `target`: what to act on, described "
                "as you would point it out to someone looking at the screen."
            ),
            None,
            None,
        )

    placed = grounder.ground(screenshot, target)
    if placed is None:
        return NOT_FOUND.format(target=target), None, None

    x, y = placed.point
    # Every describe variant carries this, set by actions._describe_variant.
    resolved = type(action).coordinate_form(
        coordinates=Coordinates(x=x, y=y), **action.model_dump(exclude={"type"})
    )
    logger.info("%r resolves to (%d, %d).", target, x, y)
    return None, resolved, placed.box
