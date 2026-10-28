"""Build a durable replay trace from one successful run."""

import logging
from typing import List, Optional, Sequence, Tuple

from yewo.agent.actuation.actions import (
    Action,
    ActionType,
    KeyEventAction,
    PlannerTapAction,
    SwipeAction,
    TapAction,
    TypeAction,
    WaitAction,
)
from yewo.agent.memory.replay import REPLAYABLE_KEYCODES
from yewo.agent.memory.replay_projection import project_actionable_elements
from yewo.agent.memory.replay_store import (
    BOUNDS_SCALE,
    AnchorContext,
    CachedAction,
    CachedTransition,
    ElementAnchor,
    RelativeBounds,
    ReplayTrace,
    ScreenSignature,
    TraceIdentity,
    ValueNames,
    anchors_of,
    relative_bounds,
)
from yewo.agent.run.postconditions import ValueVerificationPolicy, specification_for
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

logger = logging.getLogger(__name__)

TAP_ACTIONS = (TapAction, PlannerTapAction)


def _build_cached_action(
    action: Action,
    anchor: Optional[ElementAnchor],
    names: ValueNames,
    scan: Optional[ScanInfo] = None,
    verification_policies: Optional[dict[str, ValueVerificationPolicy]] = None,
) -> Optional[CachedAction]:
    """Build a secret-safe cached action, or refuse the action."""
    if isinstance(action, TypeAction):
        name = names.name_of(action.text)
        if name is None:
            return None
        postcondition = specification_for(action, names, verification_policies)
        return CachedAction(
            kind=ActionType.TYPE.value,
            target=anchor,
            described=action.target,
            target_bounds=_successful_target_bounds(action, anchor, scan),
            sensitive=False,
            value_name=name,
            postconditions=(postcondition,) if postcondition is not None else (),
        )
    if isinstance(action, TAP_ACTIONS):
        return CachedAction(
            kind=ActionType.TAP.value,
            target=anchor,
            described=action.target,
            target_bounds=_successful_target_bounds(action, anchor, scan),
            sensitive=False,
            count=action.count,
            interval_ms=action.interval_ms,
        )
    if isinstance(action, KeyEventAction) and action.key in REPLAYABLE_KEYCODES:
        return CachedAction(kind=ActionType.KEY_EVENT.value, key=action.key)
    if isinstance(action, WaitAction):
        return CachedAction(kind=ActionType.WAIT.value, duration=action.duration)
    if (
        isinstance(action, SwipeAction)
        and scan is not None
        and scan.screen_width > 0
        and scan.screen_height > 0
    ):
        return CachedAction(
            kind=ActionType.SWIPE.value,
            start_x=round(action.start_x * BOUNDS_SCALE / scan.screen_width),
            start_y=round(action.start_y * BOUNDS_SCALE / scan.screen_height),
            end_x=round(action.end_x * BOUNDS_SCALE / scan.screen_width),
            end_y=round(action.end_y * BOUNDS_SCALE / scan.screen_height),
        )
    return None


def _successful_target_bounds(
    action: Action, anchor: Optional[ElementAnchor], scan: Optional[ScanInfo]
) -> Optional[RelativeBounds]:
    """Return learned target geometry, including detector-free action points."""
    if anchor is not None:
        return anchor.bounds
    if not isinstance(action, (*TAP_ACTIONS, TypeAction)) or scan is None:
        return None
    point = action.coordinates
    point_box = BoundingBox(
        x_min=point.x,
        y_min=point.y,
        x_max=point.x,
        y_max=point.y,
    )
    return relative_bounds(point_box, scan.screen_width, scan.screen_height)


def _arrival_ms_at(durations: Optional[Sequence[int]], position: int) -> int:
    """Return the observed action-to-settlement duration at this position."""
    index = position - 1
    if durations is None or index >= len(durations):
        return 0
    return durations[index]


def _build_destination(
    source: ScreenSignature,
    elements,
    scan,
    names: ValueNames,
    key: Optional[bytes],
    description: Optional[str] = None,
    grounding: str = "",
) -> ScreenSignature:
    """Build the screen contract that a transition must reach."""
    projected_elements = project_actionable_elements(elements, scan, grounding)
    anchors = tuple(
        anchor for anchor in anchors_of(projected_elements, scan, names, key) if anchor.text
    )
    source_identities = {(anchor.role, anchor.text) for anchor in source.anchors}
    arrived_identities = {(anchor.role, anchor.text) for anchor in anchors if anchor.text}
    return ScreenSignature(
        description=description,
        anchors=anchors,
        element_count=scan.element_count,
        phash=scan.phash,
        keyboard_visible=scan.keyboard_effective_visible,
        distinguishing=tuple(sorted(arrived_identities - source_identities)),
    )


def build_trace(
    identity: TraceIdentity,
    steps: Sequence[Action],
    screens: Sequence[
        Tuple[Tuple[Sequence[GUIElement], ScanInfo], Tuple[Sequence[GUIElement], ScanInfo]]
    ],
    names: ValueNames,
    artifact: Optional[str] = None,
    key: Optional[bytes] = None,
    arrival_ms: Optional[Sequence[int]] = None,
    target_descriptions: Optional[Sequence[Optional[str]]] = None,
    screen_descriptions: Optional[Sequence[Optional[str]]] = None,
    destination_descriptions: Optional[Sequence[Optional[str]]] = None,
    segment_starts: Optional[Sequence[bool]] = None,
    verification_policies: Optional[dict[str, ValueVerificationPolicy]] = None,
) -> Optional[ReplayTrace]:
    """Build the trace that a passing scenario can save."""
    transitions: List[CachedTransition] = []
    for position, (action, (before, after)) in enumerate(zip(steps, screens, strict=True), start=1):
        if action is None:
            break
        before_elements, before_scan = before
        after_elements, after_scan = after
        projected_before_elements = project_actionable_elements(
            before_elements, before_scan, identity.grounding
        )
        detected_anchors = anchors_of(projected_before_elements, before_scan, names, key)
        source_anchors = (
            detected_anchors
            if isinstance(action, WaitAction)
            else tuple(anchor for anchor in detected_anchors if anchor.text)
        )
        source = ScreenSignature(
            description=(
                screen_descriptions[position - 1]
                if screen_descriptions and position - 1 < len(screen_descriptions)
                else None
            ),
            anchors=source_anchors,
            element_count=before_scan.element_count,
            phash=before_scan.phash,
            keyboard_visible=before_scan.keyboard_effective_visible,
        )
        if not source.anchors and not (source.description and source.phash):
            logger.info(
                "Replay path learning stopped at step %d: the screen has no named anchors.",
                position,
            )
            break
        described = (
            target_descriptions[position - 1]
            if target_descriptions and position - 1 < len(target_descriptions)
            else action.target
            if isinstance(action, (*TAP_ACTIONS, TypeAction))
            else None
        )
        if isinstance(action, (*TAP_ACTIONS, TypeAction)) and action.target != described:
            action = action.model_copy(update={"target": described})
        target = _resolve_target_anchor(action, detected_anchors, before_scan)
        if isinstance(action, (*TAP_ACTIONS, TypeAction)) and target is None and not described:
            logger.info(
                "Replay path learning stopped at step %d: no unique named target contains the action point.",
                position,
            )
            break
        cached_action = _build_cached_action(
            action,
            target,
            names,
            before_scan,
            verification_policies,
        )
        if cached_action is None:
            logger.info(
                "Replay path learning stopped at step %d: the action is not safe to persist.",
                position,
            )
            break
        transitions.append(
            CachedTransition(
                position=position,
                starts_segment=(
                    segment_starts[position - 1]
                    if segment_starts and position - 1 < len(segment_starts)
                    else False
                ),
                source=source,
                action=cached_action,
                destination=_build_destination(
                    source,
                    after_elements,
                    after_scan,
                    names,
                    key,
                    description=(
                        destination_descriptions[position - 1]
                        if destination_descriptions and position - 1 < len(destination_descriptions)
                        else screen_descriptions[position]
                        if screen_descriptions and position < len(screen_descriptions)
                        else None
                    ),
                    grounding=identity.grounding,
                ),
                arrival_ms=_arrival_ms_at(arrival_ms, position),
            )
        )

    if not transitions:
        return None
    return ReplayTrace(
        identity=identity,
        learned_from=artifact,
        transitions=tuple(transitions),
    )


def _resolve_target_anchor(
    action, anchors: Sequence[ElementAnchor], scan
) -> Optional[ElementAnchor]:
    """Resolve the unique innermost anchor that contains the action point."""
    if not isinstance(action, (*TAP_ACTIONS, TypeAction)) or not anchors:
        return None
    if scan.screen_width <= 0 or scan.screen_height <= 0:
        return None

    point = action.coordinates
    across = point.x * BOUNDS_SCALE / scan.screen_width
    down = point.y * BOUNDS_SCALE / scan.screen_height
    enclosing_anchors = [
        anchor
        for anchor in anchors
        if anchor.bounds.left <= across <= anchor.bounds.right
        and anchor.bounds.top <= down <= anchor.bounds.bottom
    ]
    if not enclosing_anchors:
        return None
    anchors_with_area = [
        (
            anchor,
            (anchor.bounds.right - anchor.bounds.left) * (anchor.bounds.bottom - anchor.bounds.top),
        )
        for anchor in enclosing_anchors
    ]
    smallest_area = min(area for _, area in anchors_with_area)
    innermost = [anchor for anchor, area in anchors_with_area if area == smallest_area]
    if len(innermost) != 1:
        return None
    target = innermost[0]
    if target.text:
        return target
    context = _resolve_anchor_context(target, anchors)
    if context is None:
        return None
    return target.model_copy(update={"context": context})


def _resolve_anchor_context(
    target: ElementAnchor, anchors: Sequence[ElementAnchor]
) -> Optional[AnchorContext]:
    candidates = [
        anchor
        for anchor in anchors
        if anchor.text and not _contains_bounds(anchor.bounds, target.bounds)
    ]
    if not candidates:
        return None
    candidates_with_distance = [
        (anchor, _bounds_distance_squared(target.bounds, anchor.bounds)) for anchor in candidates
    ]
    nearest_distance = min(distance for _, distance in candidates_with_distance)
    nearest = [
        anchor for anchor, distance in candidates_with_distance if distance == nearest_distance
    ]
    if len(nearest) != 1:
        return None
    anchor = nearest[0]
    return AnchorContext(
        role=anchor.role,
        text=anchor.text,
        bounds=anchor.bounds,
        twins=anchor.twins,
    )


def _contains_bounds(outer: RelativeBounds, inner: RelativeBounds) -> bool:
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom
    )


def _bounds_distance_squared(first: RelativeBounds, second: RelativeBounds) -> int:
    horizontal_gap = max(first.left - second.right, second.left - first.right, 0)
    vertical_gap = max(first.top - second.bottom, second.top - first.bottom, 0)
    return horizontal_gap * horizontal_gap + vertical_gap * vertical_gap
