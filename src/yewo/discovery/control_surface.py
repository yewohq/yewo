"""Project one perceived screen into the interaction surface discovery needs.

Perception is vision. A detector returns boxes and its own class names, OCR
returns text lines, and fusion attaches the text to the boxes. Nothing here
reads an accessibility tree, so nothing here knows what a platform declared
about a control. What a control offers is resolved from the roles the detector
package declares, and a role the package does not declare is never assumed.

Two consequences are deliberate. A control whose roles do not say it accepts
text is never reported as editable. A capability that needs a field falls short
rather than acting on a guess. And a control below the fold was never detected
at all, so this surface speaks only for the frame it was built from.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum

from yewo.agent.perception.elements import GUIElement, element_roles
from yewo.agent.perception.fusion import PerceptionFrame
from yewo.agent.perception.keyboard import KEYBOARD_KEY_ROLE, KEYBOARD_REGION_ROLE
from yewo.agent.perception.ocr import OCRLine
from yewo.discovery.system_chrome import is_inside_system_chrome


def control_text_digest(text: str) -> str:
    """Identify a control's name across runs without retaining the name."""
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else ""


class ControlInteraction(str, Enum):
    """What a control offers, normalized away from detector spellings."""

    ACTIVATE = "activate"
    EDIT = "edit"
    INERT = "inert"


class ControlLabelSource(str, Enum):
    """Where a control's name came from, including the case where it has none."""

    OWN = "own"
    ASSOCIATED = "associated"
    ABSENT = "absent"


INPUT_ROLES = frozenset(
    {
        "text_input",
        "text_field",
        "textfield",
        "edittext",
        "search_field",
        "searchfield",
        "secure_text_field",
    }
)
"""Roles that mean a control accepts typed text.

Matched against the roles a detector package declares, never against app
vocabulary. A package declaring none of these yields no editable control,
which is the fail-closed reading: unobserved, not absent.
"""

SYSTEM_CHROME_ROLES = frozenset({"statusbar"})
"""Detector classes the platform draws, not the application.

Separate from reading matter because the two answer different questions. A
screen's own heading is reading matter and must survive. The clock in the
status bar is not the application's and must not.
"""

STRUCTURAL_ROLES = frozenset({"line"})
"""Detector classes that describe layout rather than a bounded tap target."""

NON_INTERACTIVE_ROLES = STRUCTURAL_ROLES | SYSTEM_CHROME_ROLES
"""Every class that is not a control."""

KEYBOARD_ROLES = frozenset({KEYBOARD_REGION_ROLE, KEYBOARD_KEY_ROLE})
"""The input surface itself, keyed on the roles `perception.keyboard` owns.

A key is never a discovery frontier. Pressing one types into whatever already
holds focus, so it reaches no capability the frontier does not already own.
Recognition depends on the package declaring these roles, which is the same
contract the keyboard policy itself rests on.
"""

MAXIMUM_ASSOCIATION_GAP_PX = 120
"""Device pixels between a control and the text line above it that may name it.

Text further away than this belongs to something else. Kept beside the
association rule because the rule and the distance it allows are one decision.
"""

CONTAINER_VIEWPORT_FRACTION = 0.6
"""Share of the viewport above which a detection is scenery, not a control.

A detector reports the backdrop it sees as readily as a button. Counting a
backdrop as activatable would inflate every surface by one control that leads
nowhere.
"""


@dataclass(frozen=True)
class SurfaceControl:
    """One control a capability needs, with the evidence to reacquire it.

    Carries no label text. A digest identifies a name across runs. The
    anchors let a control OCR could not name be found again by what sits
    around it.
    """

    role: str
    interaction: ControlInteraction
    label_source: ControlLabelSource
    text_digest: str
    bounds: tuple[int, int, int, int]
    detection_confidence: float
    is_clipped_by_viewport: bool
    anchor_above_digest: str = ""
    anchor_below_digest: str = ""

    @property
    def center(self) -> tuple[int, int]:
        """The point an action targets, from which control identity derives."""
        left, top, right, bottom = self.bounds
        return (left + right) // 2, (top + bottom) // 2

    @property
    def is_reacquirable(self) -> bool:
        """Whether later observation can find this control again by evidence."""
        if self.text_digest:
            return True
        return bool(self.anchor_above_digest or self.anchor_below_digest)


@dataclass(frozen=True)
class ControlSurface:
    """Every control one frame offered, including the ones OCR could not name.

    Describes that frame and no other. A control the detector did not see is
    absent here rather than inert. That happens below the fold and behind an
    overlay. Absence is not evidence that the control does not exist.
    """

    controls: tuple[SurfaceControl, ...]
    viewport: tuple[int, int]
    unassociated_text_digests: tuple[str, ...] = ()

    @property
    def editable_controls(self) -> tuple[SurfaceControl, ...]:
        return tuple(
            control
            for control in self.controls
            if control.interaction is ControlInteraction.EDIT
        )

    @property
    def unnamed_controls(self) -> tuple[SurfaceControl, ...]:
        """Controls no text named. Reported, because silence is not absence."""
        return tuple(
            control
            for control in self.controls
            if control.label_source is ControlLabelSource.ABSENT
        )

    @property
    def unreacquirable_controls(self) -> tuple[SurfaceControl, ...]:
        return tuple(control for control in self.controls if not control.is_reacquirable)


def build_control_surface(
    frame: PerceptionFrame,
    viewport: tuple[int, int],
    platform_chrome_bounds: tuple[tuple[int, int, int, int], ...] = (),
) -> ControlSurface:
    """Project one perceived frame's interaction surface.

    Args:
        frame: One fused observation: detected elements, OCR lines, and the
            associations between them.
        viewport: The device's width and height in pixels.

    Returns:
        Every detection that can be acted on or typed into. Each carries the
        digests that find it again. The surface also carries the text OCR
        read that no detection claimed.
    """
    text_lines = tuple(
        _build_text_line(line) for line in frame.text_lines if line.text.strip()
    )
    detected_chrome_bounds = tuple(
        _element_bounds(element)
        for element in frame.elements
        if element_roles(element) & SYSTEM_CHROME_ROLES
    )
    system_chrome_bounds = (*detected_chrome_bounds, *platform_chrome_bounds)
    controls = []
    for element in frame.elements:
        control = _build_surface_control(
            element,
            text_lines,
            viewport,
            system_chrome_bounds,
        )
        if control is not None:
            controls.append(control)
    return ControlSurface(
        controls=_deduplicate_overlapping_controls(tuple(controls)),
        viewport=viewport,
        unassociated_text_digests=tuple(
            control_text_digest(text) for text in frame.unmatched_text()
        ),
    )


@dataclass(frozen=True)
class _TextLine:
    """One OCR line reduced to the box and text the association rule needs."""

    bounds: tuple[int, int, int, int]
    text: str


def _build_text_line(line: OCRLine) -> _TextLine:
    x_values = [point.x for point in line.box]
    y_values = [point.y for point in line.box]
    return _TextLine(
        bounds=(
            math.floor(min(x_values)),
            math.floor(min(y_values)),
            math.ceil(max(x_values)),
            math.ceil(max(y_values)),
        ),
        text=line.text,
    )


def _build_surface_control(
    element: GUIElement,
    text_lines: tuple[_TextLine, ...],
    viewport: tuple[int, int],
    system_chrome_bounds: tuple[tuple[int, int, int, int], ...],
) -> SurfaceControl | None:
    """Project one detection, or decline it as keyboard, scenery, or chrome."""
    roles = element_roles(element)
    if roles & KEYBOARD_ROLES:
        return None
    bounds = _element_bounds(element)
    if is_inside_system_chrome(bounds, system_chrome_bounds):
        return None
    if _is_container_sized(bounds, viewport):
        return None
    interaction = _resolve_interaction(roles)
    if interaction is ControlInteraction.INERT:
        return None
    above = _select_nearest_line(bounds, text_lines, below=False)
    below = _select_nearest_line(bounds, text_lines, below=True)
    fused_text = " ".join(element.text.split())
    if fused_text:
        label_source = ControlLabelSource.OWN
        text_digest = control_text_digest(fused_text)
    elif above is not None and interaction is ControlInteraction.EDIT:
        label_source = ControlLabelSource.ASSOCIATED
        text_digest = control_text_digest(above.text)
    else:
        label_source = ControlLabelSource.ABSENT
        text_digest = ""
    return SurfaceControl(
        role=_resolve_role(element),
        interaction=interaction,
        label_source=label_source,
        text_digest=text_digest,
        bounds=bounds,
        detection_confidence=element.confidence,
        is_clipped_by_viewport=_is_clipped_by_viewport(bounds, viewport),
        anchor_above_digest=control_text_digest(above.text) if above else "",
        anchor_below_digest=control_text_digest(below.text) if below else "",
    )


def _resolve_interaction(roles: frozenset[str]) -> ControlInteraction:
    """Decide what a detection offers from the roles its package declared.

    Input is tested first so a class declaring both text and input resolves to
    a field rather than to a label.
    """
    if roles & INPUT_ROLES:
        return ControlInteraction.EDIT
    if roles & NON_INTERACTIVE_ROLES:
        return ControlInteraction.INERT
    return ControlInteraction.ACTIVATE


def _resolve_role(element: GUIElement) -> str:
    """The detector's own name for this class, for reporting and not for policy."""
    for name in (element.element, element.type):
        normalized = " ".join(name.split()).casefold()
        if normalized:
            return normalized
    return "control"


def _element_bounds(element: GUIElement) -> tuple[int, int, int, int]:
    box = element.bounding_box
    return box.x_min, box.y_min, box.x_max, box.y_max


def _select_nearest_line(
    bounds: tuple[int, int, int, int],
    text_lines: tuple[_TextLine, ...],
    *,
    below: bool,
) -> _TextLine | None:
    """The closest text line that clears this box and overlaps it horizontally."""
    left, top, right, bottom = bounds
    neighbours = [
        line
        for line in text_lines
        if min(right, line.bounds[2]) > max(left, line.bounds[0])
        and (line.bounds[1] >= bottom if below else line.bounds[3] <= top)
    ]
    if not neighbours:
        return None
    nearest = min(neighbours, key=lambda line: _vertical_gap_px(bounds, line, below=below))
    gap_px = _vertical_gap_px(bounds, nearest, below=below)
    return nearest if gap_px <= MAXIMUM_ASSOCIATION_GAP_PX else None


def _vertical_gap_px(
    bounds: tuple[int, int, int, int],
    line: _TextLine,
    *,
    below: bool,
) -> int:
    return line.bounds[1] - bounds[3] if below else bounds[1] - line.bounds[3]


def _is_container_sized(
    bounds: tuple[int, int, int, int],
    viewport: tuple[int, int],
) -> bool:
    width, height = viewport
    viewport_area_px = max(1, width * height)
    return _area_px(bounds) / viewport_area_px >= CONTAINER_VIEWPORT_FRACTION


def _is_clipped_by_viewport(
    bounds: tuple[int, int, int, int],
    viewport: tuple[int, int],
) -> bool:
    """Whether the box meets a viewport edge, so it may continue off screen."""
    width, height = viewport
    left, top, right, bottom = bounds
    return left <= 0 or top <= 0 or right >= width or bottom >= height


def _area_px(bounds: tuple[int, int, int, int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def _deduplicate_overlapping_controls(
    controls: tuple[SurfaceControl, ...],
) -> tuple[SurfaceControl, ...]:
    kept: list[SurfaceControl] = []
    for control in controls:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if _is_same_labeled_target(existing, control)
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(control)
            continue
        existing = kept[duplicate_index]
        if _control_preference(control) < _control_preference(existing):
            kept[duplicate_index] = control
    return tuple(kept)


def _is_same_labeled_target(first: SurfaceControl, second: SurfaceControl) -> bool:
    if not first.text_digest or first.text_digest != second.text_digest:
        return False
    intersection = (
        max(first.bounds[0], second.bounds[0]),
        max(first.bounds[1], second.bounds[1]),
        min(first.bounds[2], second.bounds[2]),
        min(first.bounds[3], second.bounds[3]),
    )
    smaller_area = min(_area_px(first.bounds), _area_px(second.bounds))
    return smaller_area > 0 and _area_px(intersection) / smaller_area >= 0.8


def _control_preference(control: SurfaceControl) -> tuple[bool, int]:
    return (
        control.interaction is not ControlInteraction.EDIT,
        _area_px(control.bounds),
    )


def select_control_at(
    surface: ControlSurface,
    point: tuple[int, int],
) -> SurfaceControl | None:
    """The smallest control whose box contains one point.

    Smallest wins because detections nest. A point inside a card and inside
    the button on it belongs to the button.
    """
    x, y = point
    containing = [
        control
        for control in surface.controls
        if control.bounds[0] <= x <= control.bounds[2]
        and control.bounds[1] <= y <= control.bounds[3]
    ]
    return min(containing, key=lambda control: _area_px(control.bounds), default=None)
