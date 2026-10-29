"""Normalize live accessibility controls into reusable semantic identities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from xml.etree import ElementTree

ANDROID_BOUNDS = re.compile(r"^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$")


@dataclass(frozen=True)
class SemanticControl:
    role: str
    text_digest: str
    bounds: tuple[int, int, int, int]
    actionable: bool

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.bounds
        return (left + right) // 2, (top + bottom) // 2


def control_text_digest(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else ""


def parse_semantic_controls(page_source: str) -> tuple[SemanticControl, ...]:
    """Read Android or iOS XML without retaining labels or coordinates."""
    root = ElementTree.fromstring(page_source)
    controls = []
    for node in root.iter():
        bounds = _bounds(node.attrib)
        if bounds is None or not _is_actionable(node.attrib):
            continue
        label = _label(node)
        if not label:
            continue
        controls.append(
            SemanticControl(
                role=_role(node.attrib),
                text_digest=control_text_digest(label),
                bounds=bounds,
                actionable=True,
            )
        )
    return tuple(controls)


def semantic_control_at(
    page_source: str,
    point: tuple[int, int],
) -> SemanticControl | None:
    """Select the smallest actionable semantic control containing one audit point."""
    x, y = point
    candidates = [
        control
        for control in parse_semantic_controls(page_source)
        if control.bounds[0] <= x <= control.bounds[2]
        and control.bounds[1] <= y <= control.bounds[3]
    ]
    return min(candidates, key=lambda item: _area(item.bounds), default=None)


def _is_actionable(attributes: dict[str, str]) -> bool:
    enabled = attributes.get("enabled", "true").casefold() != "false"
    flags = (
        attributes.get("clickable"),
        attributes.get("hittable"),
        attributes.get("focusable"),
    )
    return enabled and any(flag and flag.casefold() == "true" for flag in flags)


def _label(node: ElementTree.Element) -> str:
    own = _own_label(node.attrib)
    if own:
        return own
    return next(
        (
            label
            for descendant in node.iter()
            if descendant is not node
            if (label := _own_label(descendant.attrib))
        ),
        "",
    )


def _own_label(attributes: dict[str, str]) -> str:
    for name in ("content-desc", "label", "name", "text", "value", "resource-id"):
        value = " ".join(attributes.get(name, "").split())
        if value:
            return value
    return ""


def _role(attributes: dict[str, str]) -> str:
    raw = attributes.get("class") or attributes.get("type") or "control"
    return raw.rsplit(".", 1)[-1].casefold()


def _bounds(attributes: dict[str, str]) -> tuple[int, int, int, int] | None:
    android = ANDROID_BOUNDS.fullmatch(attributes.get("bounds", ""))
    if android is not None:
        return tuple(int(value) for value in android.groups())  # type: ignore[return-value]
    try:
        x = round(float(attributes["x"]))
        y = round(float(attributes["y"]))
        width = round(float(attributes["width"]))
        height = round(float(attributes["height"]))
    except (KeyError, ValueError):
        return None
    return x, y, x + width, y + height


def _area(bounds: tuple[int, int, int, int]) -> int:
    return max(1, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))


class ControlInteraction(str, Enum):
    """What a control offers, normalized away from platform spellings."""

    ACTIVATE = "activate"
    EDIT = "edit"
    INERT = "inert"


class ControlLabelSource(str, Enum):
    """Where a control's name came from, including the case where it has none."""

    OWN = "own"
    ASSOCIATED = "associated"
    ABSENT = "absent"


EDITABLE_ROLES = frozenset(
    {
        "edittext",
        "autocompletetextview",
        "textinputedittext",
        "searchview",
        "xcuielementtypetextfield",
        "xcuielementtypesecuretextfield",
        "xcuielementtypetextview",
        "xcuielementtypesearchfield",
    }
)
"""Roles that accept typed text. Platform vocabulary, never app vocabulary."""

MAXIMUM_ASSOCIATION_GAP = 120
"""Device pixels between a control and the label above it that may name it.

A label further away than this belongs to something else. Kept here because
the association rule and the distance it allows are one decision.
"""


@dataclass(frozen=True)
class SurfaceControl:
    """One control a capability needs, with the evidence to reacquire it.

    Carries no label text and no absolute coordinates. A digest identifies a
    name across runs, and relative bounds survive a different viewport.
    """

    role: str
    interaction: ControlInteraction
    label_source: ControlLabelSource
    text_digest: str
    bounds: tuple[int, int, int, int]
    focusable: bool
    within_scrollable: bool
    requires_scroll: bool
    anchor_above_digest: str = ""
    anchor_below_digest: str = ""

    @property
    def is_reacquirable(self) -> bool:
        """Whether later observation can find this control again by evidence."""
        if self.text_digest:
            return True
        return bool(self.anchor_above_digest or self.anchor_below_digest)


@dataclass(frozen=True)
class ControlSurface:
    """Every control one observation offered, including the unnamed ones."""

    controls: tuple[SurfaceControl, ...]
    viewport: tuple[int, int]

    @property
    def editable_controls(self) -> tuple[SurfaceControl, ...]:
        return tuple(
            control
            for control in self.controls
            if control.interaction is ControlInteraction.EDIT
        )

    @property
    def unnamed_controls(self) -> tuple[SurfaceControl, ...]:
        """Controls no metadata named. Reported, because silence is not absence."""
        return tuple(
            control
            for control in self.controls
            if control.label_source is ControlLabelSource.ABSENT
        )

    @property
    def unreacquirable_controls(self) -> tuple[SurfaceControl, ...]:
        return tuple(control for control in self.controls if not control.is_reacquirable)


def build_control_surface(page_source: str, viewport: tuple[int, int]) -> ControlSurface:
    """Project one screen's interaction surface, naming what metadata omitted.

    Args:
        page_source: The platform's accessibility XML for one observation.
        viewport: The device's width and height in pixels.

    Returns:
        Every control that can be acted on or typed into, plus the inert
        labels that let an unnamed control be found again.
    """
    root = ElementTree.fromstring(page_source)
    nodes = tuple(_surface_nodes(root))
    labelled = tuple(node for node in nodes if node.own_label)
    controls = tuple(
        _surface_control(node, labelled, viewport)
        for node in nodes
        if node.interaction is not ControlInteraction.INERT
    )
    return ControlSurface(controls=controls, viewport=viewport)


@dataclass(frozen=True)
class _SurfaceNode:
    """One parsed accessibility node, before association decides its name."""

    role: str
    own_label: str
    bounds: tuple[int, int, int, int]
    interaction: ControlInteraction
    focusable: bool
    within_scrollable: bool


def _surface_nodes(root: ElementTree.Element):
    """Walk the tree once, carrying whether an ancestor scrolls."""
    for node, scrollable in _nodes_with_scroll_context(root, False):
        bounds = _bounds(node.attrib)
        if bounds is None:
            continue
        yield _SurfaceNode(
            role=_role(node.attrib),
            own_label=_own_label(node.attrib),
            bounds=bounds,
            interaction=_interaction(node.attrib),
            focusable=_is_true(node.attrib.get("focusable")),
            within_scrollable=scrollable,
        )


def _nodes_with_scroll_context(node: ElementTree.Element, scrollable: bool):
    inherited = scrollable or _is_true(node.attrib.get("scrollable"))
    yield node, inherited
    for child in node:
        yield from _nodes_with_scroll_context(child, inherited)


def _interaction(attributes: dict[str, str]) -> ControlInteraction:
    """Normalize platform flags and roles into what the control offers."""
    if not _is_enabled(attributes):
        return ControlInteraction.INERT
    if _role(attributes) in EDITABLE_ROLES or _is_true(attributes.get("editable")):
        return ControlInteraction.EDIT
    if _is_actionable(attributes):
        return ControlInteraction.ACTIVATE
    return ControlInteraction.INERT


def _surface_control(
    node: _SurfaceNode,
    labelled: tuple[_SurfaceNode, ...],
    viewport: tuple[int, int],
) -> SurfaceControl:
    above = _nearest_label(node, labelled, below=False)
    below = _nearest_label(node, labelled, below=True)
    associated = above if node.interaction is ControlInteraction.EDIT else None
    if node.own_label:
        label_source = ControlLabelSource.OWN
        text_digest = control_text_digest(node.own_label)
    elif associated is not None:
        label_source = ControlLabelSource.ASSOCIATED
        text_digest = control_text_digest(associated.own_label)
    else:
        label_source = ControlLabelSource.ABSENT
        text_digest = ""
    return SurfaceControl(
        role=node.role,
        interaction=node.interaction,
        label_source=label_source,
        text_digest=text_digest,
        bounds=node.bounds,
        focusable=node.focusable,
        within_scrollable=node.within_scrollable,
        requires_scroll=_requires_scroll(node.bounds, viewport),
        anchor_above_digest=control_text_digest(above.own_label) if above else "",
        anchor_below_digest=control_text_digest(below.own_label) if below else "",
    )


def _nearest_label(
    node: _SurfaceNode,
    labelled: tuple[_SurfaceNode, ...],
    *,
    below: bool,
) -> _SurfaceNode | None:
    """The closest named neighbour that overlaps this control horizontally."""
    left, top, right, bottom = node.bounds
    neighbours = [
        candidate
        for candidate in labelled
        if candidate.bounds != node.bounds
        and min(right, candidate.bounds[2]) > max(left, candidate.bounds[0])
        and (
            candidate.bounds[1] >= bottom
            if below
            else candidate.bounds[3] <= top
        )
    ]
    if not neighbours:
        return None
    nearest = min(
        neighbours,
        key=lambda candidate: (
            candidate.bounds[1] - bottom if below else top - candidate.bounds[3]
        ),
    )
    gap = nearest.bounds[1] - bottom if below else top - nearest.bounds[3]
    return nearest if gap <= MAXIMUM_ASSOCIATION_GAP else None


def _requires_scroll(bounds: tuple[int, int, int, int], viewport: tuple[int, int]) -> bool:
    width, height = viewport
    left, top, right, bottom = bounds
    return left < 0 or top < 0 or right > width or bottom > height


def _is_true(value: str | None) -> bool:
    return bool(value) and value.casefold() == "true"


def _is_enabled(attributes: dict[str, str]) -> bool:
    return attributes.get("enabled", "true").casefold() != "false"
