"""Keyboard ownership policy over model-declared semantic roles.

This module does not detect keyboards and contains no platform or screen-position
rules. The model identifies keyboard regions and keys. This policy keeps them
separate from ordinary app controls and expands visual keys when the planner asks.
"""

import logging
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

from yewo.agent.perception.elements import BoundingBox, GUIElement

logger = logging.getLogger(__name__)

KEYBOARD_REGION_ROLE = "keyboard_region"
KEYBOARD_KEY_ROLE = "keyboard_key"


@dataclass(frozen=True)
class KeyboardObservation:
    """Visual/native keyboard evidence and the elements owned by the keyboard."""

    native_visible: Optional[bool]
    visual_visible: bool
    effective_visible: bool
    regions: tuple[BoundingBox, ...]
    app_elements: tuple[GUIElement, ...]
    owned_elements: tuple[GUIElement, ...]
    key_elements: tuple[GUIElement, ...]

    @property
    def state_disagrees(self) -> bool:
        return self.native_visible is not None and self.native_visible != self.visual_visible


def _has_role(element: GUIElement, role: str) -> bool:
    return role in getattr(element, "semantic_roles", frozenset())


def _center_is_inside(element: GUIElement, region: BoundingBox) -> bool:
    box = element.bounding_box
    center_x = (box.x_min + box.x_max) / 2
    center_y = (box.y_min + box.y_max) / 2
    return region.x_min <= center_x <= region.x_max and region.y_min <= center_y <= region.y_max


def observe(
    elements: Optional[Sequence[GUIElement]], native_visible: Optional[bool] = None
) -> KeyboardObservation:
    """Resolve keyboard ownership from semantic roles, plus native visibility."""
    all_elements = list(elements or [])
    region_elements = [
        element for element in all_elements if _has_role(element, KEYBOARD_REGION_ROLE)
    ]
    regions = tuple(element.bounding_box for element in region_elements)
    explicit_keys = [element for element in all_elements if _has_role(element, KEYBOARD_KEY_ROLE)]

    owned, app = [], []
    for element in all_elements:
        is_region = _has_role(element, KEYBOARD_REGION_ROLE)
        is_key = _has_role(element, KEYBOARD_KEY_ROLE)
        is_contained = any(_center_is_inside(element, region) for region in regions)
        if is_region or is_key or is_contained:
            owned.append(element)
        else:
            app.append(element)

    visual_visible = bool(regions or explicit_keys)
    observation = KeyboardObservation(
        native_visible=native_visible,
        visual_visible=visual_visible,
        effective_visible=native_visible is True or visual_visible,
        regions=regions,
        app_elements=tuple(app),
        owned_elements=tuple(owned),
        key_elements=tuple(explicit_keys),
    )
    if observation.state_disagrees:
        logger.info(
            "Keyboard evidence disagrees: native=%s visual=%s.",
            native_visible,
            visual_visible,
        )
    return observation


def target_elements(
    elements: Optional[Sequence[GUIElement]],
    native_visible: Optional[bool],
    include_keyboard_targets: bool = False,
) -> tuple[List[GUIElement], List[GUIElement]]:
    """Return actionable elements while retaining keyboard ownership separately."""
    observation = observe(elements, native_visible)
    targets = list(observation.app_elements)
    if include_keyboard_targets:
        for key in observation.key_elements:
            annotated = copy(key)
            text = (annotated.text or "").strip()
            annotated.text = f"[key] {text}" if text else "[key]"
            targets.append(annotated)
    return targets, list(observation.owned_elements)


def split_keyboard(
    elements: Optional[Sequence[GUIElement]],
    keyboard_open: Optional[bool],
    screen_height: Optional[int],
) -> tuple[List[GUIElement], List[GUIElement]]:
    """Separate keyboard-owned detections from app controls.

    `screen_height` remains in the public signature for callers that number a
    screenshot, but semantic ownership intentionally does not use it.
    """
    del screen_height
    return target_elements(elements, keyboard_open)


def without_keyboard_keys(
    elements: Optional[Sequence[GUIElement]],
    keyboard_open: Optional[bool],
    screen_height: Optional[int],
) -> List[GUIElement]:
    """Return app controls with model-declared keyboard content removed."""
    return split_keyboard(elements, keyboard_open, screen_height)[0]


def keyboard_band(keys: List[GUIElement], screen_width: int) -> Optional[tuple]:
    """Return the actual bounds of the declared keyboard surface."""
    del screen_width
    if not keys:
        return None
    region_boxes = [
        element.bounding_box for element in keys if _has_role(element, KEYBOARD_REGION_ROLE)
    ]
    boxes = region_boxes or [element.bounding_box for element in keys]
    return (
        min(box.x_min for box in boxes),
        min(box.y_min for box in boxes),
        max(box.x_max for box in boxes),
        max(box.y_max for box in boxes),
    )


def annotate_keyboard_keys(
    elements: Optional[Sequence[GUIElement]],
    keyboard_open: Optional[bool],
    screen_height: Optional[int],
) -> List[GUIElement]:
    """Label semantic keyboard content for planner modes that retain all boxes."""
    del screen_height
    observation = observe(elements, keyboard_open)
    owned_ids = {id(element) for element in observation.owned_elements}
    annotated = []
    for element in elements or []:
        if id(element) in owned_ids:
            element = copy(element)
            text = (element.text or "").strip()
            prefix = "[keyboard]" if _has_role(element, KEYBOARD_REGION_ROLE) else "[key]"
            element.text = f"{prefix} {text}" if text else prefix
        annotated.append(element)
    return annotated
