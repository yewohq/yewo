"""Building the snapshot the rest of the loop decides from.

Perceive, record the transition, then number the targets once the keyboard
state is known.
"""

import io
import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

from PIL import Image

from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.grounding import marks
from yewo.agent.perception.screen import ScreenPerception
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.memory.trail import SessionTrail

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """One reading of the screen, and everything decided from it."""

    gui_elements: list
    marks: list
    raw_img_bytes: bytes
    planner_img_bytes: bytes
    annotated_img: Image.Image
    current_phash: Optional[str]
    timestamp: str
    scan_info: Optional[ScanInfo] = None
    prev_phash: Optional[str] = None
    planner_elements: Optional[list] = None
    planner_screen_height: Optional[int] = None
    refinement: Optional[dict] = None
    view_note: Optional[str] = None


def capture(
    perception: ScreenPerception,
    session_trail: Optional[SessionTrail],
    prev_phash: Optional[str],
) -> PerceptionSnapshot:
    gui_elements, raw_img, annotated_img = perception.perceive()
    current_phash = (
        str(perception.current_hash) if perception.current_hash else None
    )
    # Describe mode detects nothing, so a count of what was detected says
    # only which mode is running.
    if gui_elements:
        logger.info("Perceived %d elements. pHash: %s", len(gui_elements), current_phash)
    else:
        logger.info("Read the screen. pHash: %s", current_phash)

    if session_trail and current_phash:
        if prev_phash and current_phash != prev_phash:
            session_trail.flush_transition(current_phash)
        session_trail.record_screen(current_phash, "", gui_elements)

    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    return PerceptionSnapshot(
        gui_elements=gui_elements,
        marks=[],
        raw_img_bytes=raw_bytes,
        planner_img_bytes=raw_bytes,
        annotated_img=annotated_img,
        current_phash=current_phash,
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        scan_info=perception.last_scan_info,
        prev_phash=prev_phash,
    )


def _crop_box(region: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Return a padded third-of-screen crop for a coarse region name."""
    vertical, horizontal = region.split("_", 1)
    column = {"left": 0, "center": 1, "right": 2}[horizontal]
    row = {"upper": 0, "middle": 1, "lower": 2}[vertical]
    x0, x1 = column * width // 3, (column + 1) * width // 3
    y0, y1 = row * height // 3, (row + 1) * height // 3
    pad_x, pad_y = width // 12, height // 12
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def _full_screen_element(element: GUIElement, crop, scale: int) -> GUIElement:
    """Map one crop detection back to the untouched screenshot coordinates."""
    left, top, _, _ = crop
    box = element.bounding_box
    mapped = element.model_copy(deep=True)
    mapped.bounding_box = BoundingBox(
        x_min=left + round(box.x_min / scale),
        y_min=top + round(box.y_min / scale),
        x_max=left + round(box.x_max / scale),
        y_max=top + round(box.y_max / scale),
    )
    return mapped


def _full_screen_mark(mark, crop, scale: int):
    mapped = _full_screen_element(mark.element, crop, scale)
    point = None
    if mark.point is not None:
        point = (
            crop[0] + round(mark.point[0] / scale),
            crop[1] + round(mark.point[1] / scale),
        )
    return replace(mark, element=mapped, point=point)


def _number_refined_targets(snapshot: PerceptionSnapshot, ctx, perception, region: str) -> None:
    """Detect and number a magnified crop, while keeping action points global."""
    source = Image.open(io.BytesIO(snapshot.raw_img_bytes))
    crop = _crop_box(region, *source.size)
    scale = 2
    close_up = source.crop(crop).resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale),
        Image.Resampling.LANCZOS,
    )
    buf = io.BytesIO()
    close_up.save(buf, format="PNG")
    close_up_bytes = buf.getvalue()
    local_elements, _ = perception.image_descriptor.describe(close_up_bytes)
    local_marks, planner_bytes = marks.number(
        local_elements,
        close_up_bytes,
        keyboard_open=False,
        screen_height=close_up.height,
    )
    snapshot.marks = [
        _full_screen_mark(mark, crop, scale) for mark in local_marks
    ]
    snapshot.planner_img_bytes = planner_bytes
    snapshot.planner_elements = local_elements
    snapshot.planner_screen_height = close_up.height
    snapshot.refinement = {
        "region": region,
        "crop": list(crop),
        "scale": scale,
        "full_screen_size": list(source.size),
        "detected_elements": len(local_elements),
    }
    snapshot.view_note = (
        f"VISUAL REFINEMENT: The screenshot is a {scale}x magnified {region} "
        "crop of the unchanged full screen. Numbered targets map back to the "
        "full screen. Select a number only when the requested target is visible."
    )
    logger.info(
        "Refined %s crop found %d elements; action points remain full-screen.",
        region,
        len(local_elements),
    )


def number_targets(
    snapshot: PerceptionSnapshot, ctx, grounding: str, perception=None
) -> None:
    """Numbers the elements and redraws the screenshot to match.

    Done here rather than at perception time, because only now is the keyboard
    state known. The keyboard's own keys must not carry numbers: they would
    outnumber the app's controls several times over.
    """
    if grounding != GROUNDING_MARKS:
        return
    pending = getattr(ctx, "pending_visual_refinement", None)
    if hasattr(ctx, "pending_visual_refinement"):
        ctx.pending_visual_refinement = None
    if (
        pending is not None
        and pending[0] == snapshot.current_phash
        and perception is not None
    ):
        _number_refined_targets(snapshot, ctx, perception, pending[1])
        return
    snapshot.marks, snapshot.planner_img_bytes = marks.number(
        snapshot.gui_elements,
        snapshot.raw_img_bytes,
        keyboard_open=ctx.keyboard_open,
        screen_height=ctx.screen_height,
        refused=ctx.refused_marks.get(snapshot.current_phash) or [],
    )
    snapshot.planner_elements = snapshot.gui_elements
    snapshot.planner_screen_height = ctx.screen_height
