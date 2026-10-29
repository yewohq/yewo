"""Building the snapshot the rest of the loop decides from.

Perceive, record the transition, then number the targets once the keyboard
state is known.
"""

import io
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from PIL import Image

from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.grounding import grid, marks, region as region_geometry
from yewo.agent.perception.screen import ScreenPerception
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.perception.fusion import PerceptionFrame
from yewo.agent.perception.keyboard import KeyboardObservation
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.run.evidence_identity import new_observation_id
from yewo.agent.run.control_discovery import ControlDiscoveryState
from yewo.agent.memory.screen_identity import is_same_screen

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
    intervention: Optional["InterventionPlan"] = None
    """A magnified region that still detected nothing, and the cells to try.

    Set by numbering, consumed by the step loop, because resolving it needs
    the device and numbering has none."""
    intervention_failure_reason: Optional[str] = None
    """Why an evidence-bounded intervention could not be constructed."""
    view_note: Optional[str] = None
    perception_frame: Optional[PerceptionFrame] = None
    planner_perception_frame: Optional[PerceptionFrame] = None
    keyboard_observation: Optional[KeyboardObservation] = None
    observation_id: str = field(default_factory=new_observation_id)


def screen_size(snapshot: PerceptionSnapshot) -> Optional[tuple[int, int]]:
    """Return the screenshot coordinate space used by actions on this reading."""
    scan_info = getattr(snapshot, "scan_info", None)
    if scan_info is None:
        return None
    return scan_info.screen_width, scan_info.screen_height


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
        session_trail.record_screen(current_phash, gui_elements)

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
        perception_frame=getattr(perception, "current_frame", None),
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


@dataclass(frozen=True)
class InterventionPlan:
    """What to try, where, when a magnified region came back empty."""

    region: str
    crop: tuple[int, int, int, int]
    scale: int
    close_up: bytes
    candidates: tuple
    intent: str
    description: str
    anchor_above: Optional[tuple[int, int, int, int]] = None
    anchor_below: Optional[tuple[int, int, int, int]] = None
    derived: Optional[dict] = None
    redaction: tuple = ()
    """What to cover on the overlay, in the magnified crop's own coordinates.

    Carried with the plan because the snapshot's copy is replaced the moment a
    control is found and the full screen is numbered again."""

    dropped_cells: int = 0
    """Cells the region had room for that the budget would not pay to probe.

    Recorded rather than discarded. A report listing ten tried cells over a
    region that held twelve has to say so, or it reads as full coverage."""


def _number_refined_targets(
    snapshot: PerceptionSnapshot,
    ctx,
    perception,
    region: str,
    description: str = "",
    intent: str = "activate",
    anchor_above=None,
    anchor_below=None,
) -> None:
    """Detect and number a magnified crop, while keeping action points global."""
    source = Image.open(io.BytesIO(snapshot.raw_img_bytes))
    derived = region_geometry.derive_between(
        anchor_above, anchor_below, snapshot.gui_elements, source.size
    )
    if derived is None:
        snapshot.intervention_failure_reason = (
            f"The {region} region could not be bounded by surrounding controls. "
            "No candidate points were probed."
        )
        snapshot.view_note = (
            "VISUAL INTERVENTION UNAVAILABLE: surrounding controls did not "
            "bound the requested region."
        )
        logger.info("No surrounding controls bound the %s region.", region)
        return
    crop = derived.crop
    scale = region_geometry.scale_for(crop)
    close_up = source.crop(crop).resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale),
        Image.Resampling.LANCZOS,
    )
    buf = io.BytesIO()
    close_up.save(buf, format="PNG")
    close_up_bytes = buf.getvalue()
    local_elements, _ = perception.image_descriptor.describe(close_up_bytes)
    snapshot.planner_perception_frame = getattr(perception.image_descriptor, "last_frame", None)
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
        "derivation": derived.as_record() if derived else None,
    }
    snapshot.view_note = (
        f"VISUAL REFINEMENT: The screenshot is a {scale}x magnified {region} "
        "crop of the unchanged full screen. Numbered targets map back to the "
        "full screen. Select a number only when the requested target is visible."
    )
    logger.info(
        "Refined %s crop found %d elements. Action points remain full-screen.",
        region,
        len(local_elements),
    )
    if not snapshot.marks:
        snapshot.intervention = _grid_plan(
            region,
            crop,
            scale,
            close_up_bytes,
            local_elements,
            description,
            intent,
            derived,
            anchor_above,
            anchor_below,
            _close_up_redaction(snapshot, local_elements),
        )


def _close_up_redaction(snapshot: PerceptionSnapshot, local_elements) -> tuple:
    """Secrets to cover on the magnified crop, measured in that crop."""
    frame = snapshot.planner_perception_frame
    ocr = frame.ocr_redaction_evidence() if frame is not None else ()
    return (*local_elements, *ocr)


def _grid_plan(
    region,
    crop,
    scale,
    close_up_bytes,
    local_elements,
    description,
    intent,
    derived,
    anchor_above,
    anchor_below,
    redaction,
) -> "InterventionPlan":
    """Cells to try where magnifying a bounded region still numbered nothing.

    The raw detections go with them. A cell over one of those covers something
    numbering dropped. A cell over none covers something never detected. The
    two are allowed different things.
    """
    local_marks_elements = [
        _full_screen_element(element, crop, scale) for element in local_elements
    ]
    candidates = grid.build(crop, local_marks_elements)
    room = grid.cell_capacity(crop)
    logger.info(
        "Magnified %s region numbered nothing. Offering %d candidate cells.",
        region,
        len(candidates),
    )
    return InterventionPlan(
        region=region,
        crop=crop,
        scale=scale,
        close_up=close_up_bytes,
        candidates=tuple(candidates),
        intent=intent,
        description=description,
        anchor_above=anchor_above,
        anchor_below=anchor_below,
        derived=derived.as_record() if derived else None,
        redaction=tuple(redaction),
        dropped_cells=max(0, room - len(candidates)),
    )


def number_targets(
    snapshot: PerceptionSnapshot, ctx, grounding: str, perception=None
) -> None:
    """Numbers the elements and redraws the screenshot to match.

    Keyboard keys stay collapsed for ordinary app interaction. If the planner
    explicitly asks to test the visual keyboard surface, the next view numbers
    its model-declared keys alongside the app controls.
    """
    if grounding != GROUNDING_MARKS:
        return
    pending = getattr(ctx, "control_discovery", None)
    if (
        pending is not None
        and pending.state in (
            ControlDiscoveryState.REQUESTED,
            ControlDiscoveryState.PROBING,
        )
        and is_same_screen(pending.source_phash, snapshot.current_phash)
        and perception is not None
    ):
        _number_refined_targets(
            snapshot,
            ctx,
            perception,
            pending.region,
            pending.description,
            pending.intent,
            pending.anchor_above,
            pending.anchor_below,
        )
        return
    snapshot.marks, snapshot.planner_img_bytes = marks.number(
        snapshot.gui_elements,
        snapshot.raw_img_bytes,
        keyboard_open=ctx.keyboard_open,
        screen_height=ctx.screen_height,
        refused=ctx.refused_marks.get(snapshot.current_phash) or [],
        include_keyboard_targets=getattr(ctx, "keyboard_targets_expanded", False),
    )
    snapshot.planner_elements = snapshot.gui_elements
    snapshot.planner_screen_height = ctx.screen_height
