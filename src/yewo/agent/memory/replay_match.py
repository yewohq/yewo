"""Deciding whether the screen in front of a run is the one a trace expects.

Two questions, and they are not one question. Is this the screen the transition
starts from, and which element on it is the one the transition touched. The
first is answered loosely, because a build changes a screen without changing
what it is. The second is answered strictly, because acting on the wrong
control is the failure this phase exists to prevent.

Every answer here is a refusal or a certainty. There is no best guess: a
caller that cannot match falls back to planning, which is slower and always
correct.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from yewo.agent.perception.elements import GUIElement
from yewo.agent.memory.replay_store import (
    ElementAnchor,
    RelativeBounds,
    ScreenSignature,
    ValueNames,
    anchors_of,
)

logger = logging.getLogger(__name__)

BOUNDS_TOLERANCE = 40
"""How far a control may move and still be that control, in thousandths.

Four percent of the screen. A rebuild nudges a control by padding or a font
change. A different control is somewhere else entirely. Wide enough to
survive the first, narrow enough to refuse the second.
"""

SCREEN_AGREEMENT = 0.6
"""How much of the expected screen has to still be there.

A screen is not its pixels. A build adds a banner and removes a hint, and it
is the same screen. Below this it is a different screen, and the
transition that started here does not start here.
"""

DESTINATION_MARKER_AGREEMENT = 0.5
"""The share of newly produced semantic markers that must remain.

Detection and OCR can lose one marker between two readings of the same
screen. Requiring half still proves that the action produced new semantic
content, while one missing detection cannot reject an otherwise matching
destination.
"""


@dataclass(frozen=True)
class Match:
    """One element the trace meant, as it is on the screen now."""

    element: GUIElement
    anchor: ElementAnchor


@dataclass(frozen=True)
class Refusal:
    """Why nothing was matched, in words a run log can carry."""

    reason: str


def _count_by_identity(anchors: Sequence[ElementAnchor]) -> Dict[Tuple[str, str], int]:
    """How many elements of each role and reading a screen holds."""
    counts_by_identity: Dict[Tuple[str, str], int] = {}
    for anchor in anchors:
        counts_by_identity[(anchor.role, anchor.text)] = (
            counts_by_identity.get((anchor.role, anchor.text), 0) + 1
        )
    return counts_by_identity


def screen_agrees(
    expected: ScreenSignature,
    elements: Sequence,
    scan,
    names: ValueNames,
    key: Optional[bytes] = None,
    allow_detector_count_drift: bool = False,
) -> Optional[Refusal]:
    """Whether this is still the screen the transition starts from.

    Counted over the expected anchors rather than the current ones. A build
    that adds ten elements has not removed the screen's identity, and
    counting the other way would refuse it for growing.

    Counted, not set: two controls reading alike are two controls, and a
    screen that lost one of them is not the screen that had both. The twin
    counts hold on their own rather than inside the share, which would
    forgive a loss on a crowded screen.

    Enough of what distinguishes this screen from the one before it must
    remain. A whole-screen share alone would accept a failure that kept the
    surrounding controls and lost what said the action worked.
    """
    if not expected.anchors:
        return Refusal("the trace holds no anchors for this screen")

    # A screen with nothing readable on it is only its boxes, and boxes
    # line up across unrelated screens. The image hash is the one thing
    # left that tells a splash screen from a blank form.
    unreadable = not any(anchor.text for anchor in expected.anchors)
    if unreadable and (not expected.phash or expected.phash != scan.phash):
        return Refusal("this screen has nothing readable on it and a different image")

    present_anchors = anchors_of(elements, scan, names, key)
    if not present_anchors:
        return Refusal("this screen reported no size, so nothing can be placed")

    counts_by_identity = _count_by_identity(present_anchors)
    expected_counts = _count_by_identity(expected.anchors)

    for identity, count in expected_counts.items():
        present_count = counts_by_identity.get(identity, 0)
        if 0 < present_count < count:
            role, _ = identity
            if allow_detector_count_drift and role == "yolo_detection":
                continue
            return Refusal(f"a {role} that was one of {count} is now one of {present_count}")

    if expected.distinguishing:
        marker_match_count = sum(
            counts_by_identity.get(identity, 0) > 0 for identity in expected.distinguishing
        )
        marker_share = marker_match_count / len(expected.distinguishing)
        if marker_share < DESTINATION_MARKER_AGREEMENT:
            return Refusal(
                f"only {marker_match_count} of {len(expected.distinguishing)} "
                "destination markers are on this screen"
            )

    matched_count = sum(
        min(count, counts_by_identity.get(identity, 0))
        for identity, count in expected_counts.items()
    )
    share = matched_count / len(expected.anchors)
    if share < SCREEN_AGREEMENT:
        return Refusal(
            f"only {matched_count} of {len(expected.anchors)} expected elements are here"
        )
    return None


def find_target(
    expected_anchor: ElementAnchor,
    elements: Sequence,
    scan,
    names: ValueNames,
    key: Optional[bytes] = None,
) -> Tuple[Optional[Match], Optional[Refusal]]:
    """The one element this transition touched, or why there is no such one.

    Role and text have to be equal, never near. The twin count has to be
    equal too. A build that turned one "Continue" into two changed what the
    first one meant, and picking either would be a guess.
    """
    present_anchors = anchors_of(elements, scan, names, key)
    if not present_anchors:
        return None, Refusal("this screen reported no size, so nothing can be placed")

    if not expected_anchor.text:
        if expected_anchor.context is None:
            return None, Refusal("an unnamed target has no named context")
        return _find_contextual_target(expected_anchor, elements, present_anchors)

    matching_anchors = [
        (element, anchor)
        for element, anchor in zip(elements, present_anchors, strict=True)
        if anchor.role == expected_anchor.role and anchor.text == expected_anchor.text
    ]
    if not matching_anchors:
        return None, Refusal(
            f"nothing on this screen is {expected_anchor.role} {expected_anchor.text!r}"
        )

    twins = matching_anchors[0][1].twins
    if twins != expected_anchor.twins:
        return None, Refusal(
            f"{expected_anchor.text!r} was one of {expected_anchor.twins} and is now one of {twins}"
        )

    nearby_matches = [
        (element, anchor)
        for element, anchor in matching_anchors
        if anchor.bounds.is_near(expected_anchor.bounds, BOUNDS_TOLERANCE)
    ]
    if len(nearby_matches) == 1:
        return Match(element=nearby_matches[0][0], anchor=nearby_matches[0][1]), None
    if not nearby_matches:
        return None, Refusal(f"{expected_anchor.text!r} is on this screen, but somewhere else")
    return None, Refusal(
        f"{len(nearby_matches)} elements match {expected_anchor.text!r} equally well"
    )


def _find_contextual_target(
    expected: ElementAnchor,
    elements: Sequence[GUIElement],
    present_anchors: Sequence[ElementAnchor],
) -> Tuple[Optional[Match], Optional[Refusal]]:
    context = expected.context
    if context is None:
        return None, Refusal("an unnamed target has no named context")
    context_matches = [
        anchor
        for anchor in present_anchors
        if anchor.role == context.role
        and anchor.text == context.text
        and anchor.twins == context.twins
        and anchor.bounds.is_near(context.bounds, BOUNDS_TOLERANCE)
    ]
    if len(context_matches) != 1:
        return None, Refusal("the named context of an unnamed target is ambiguous")
    present_context = context_matches[0]
    target_matches = [
        (element, anchor)
        for element, anchor in zip(elements, present_anchors, strict=True)
        if anchor.role == expected.role
        and not anchor.text
        and _has_matching_context_offset(expected, context.bounds, anchor, present_context.bounds)
    ]
    if len(target_matches) == 1:
        element, anchor = target_matches[0]
        return Match(element=element, anchor=anchor), None
    if not target_matches:
        return None, Refusal("no unnamed target remains beside its named context")
    return None, Refusal(f"{len(target_matches)} unnamed targets remain beside the named context")


def _has_matching_context_offset(
    expected: ElementAnchor,
    expected_context: RelativeBounds,
    present: ElementAnchor,
    present_context: RelativeBounds,
) -> bool:
    return all(
        abs(expected_offset - present_offset) <= BOUNDS_TOLERANCE
        for expected_offset, present_offset in (
            (
                expected.bounds.left - expected_context.left,
                present.bounds.left - present_context.left,
            ),
            (
                expected.bounds.top - expected_context.top,
                present.bounds.top - present_context.top,
            ),
            (
                expected.bounds.right - expected_context.right,
                present.bounds.right - present_context.right,
            ),
            (
                expected.bounds.bottom - expected_context.bottom,
                present.bounds.bottom - present_context.bottom,
            ),
        )
    )


def destination_reached(
    expected: ScreenSignature,
    elements: Sequence,
    scan,
    names: ValueNames,
    key: Optional[bytes] = None,
) -> Optional[Refusal]:
    """Whether the action arrived where it used to arrive.

    The same question as the source check, asked for a different reason.
    A unique source match only proves something plausible was tapped. This
    is what proves the tap did what it did before.
    """
    # A detector can merge two overlapping boxes into one on adjacent reads.
    # That is safe on arrival because no action is selected from this screen.
    # Source and target matching keep exact twin counts.
    return screen_agrees(
        expected,
        elements,
        scan,
        names,
        key,
        allow_detector_count_drift=True,
    )
