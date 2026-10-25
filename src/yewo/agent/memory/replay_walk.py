"""Walk and learn one scenario's ordered saved path."""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import ValidationError

from yewo.agent.actuation.actions import (
    Action,
    ActionType,
    Coordinates,
    KeyEventAction,
    TapAction,
    TypeAction,
    WaitAction,
    SwipeAction,
)
from yewo.agent.actuation.orchestrator import ActionOrchestrator
from yewo.agent.grounding.description import SCREEN_MISMATCH
from yewo.agent.memory import replay_match
from yewo.agent.memory.replay import REPLAYABLE_KEYCODES
from yewo.agent.step import targeting
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.memory.replay_store import (
    BOUNDS_SCALE,
    CachedAction,
    CachedTransition,
    RelativeBounds,
    ReplayTrace,
    ScreenSignature,
    ValueNames,
)
from yewo.agent.memory.replay_learning import ScreenReading
from yewo.agent.memory.replay_builder import (
    _build_cached_action as _build_cached_action,
    _resolve_target_anchor as _resolve_target_anchor,
    build_trace as build_trace,
)
from yewo.agent.run.record import StepRecord, StepSource
from yewo.agent.run.postconditions import TextEntryReceipt

logger = logging.getLogger(__name__)

TARGETED_KINDS = frozenset({ActionType.TAP.value, ActionType.TYPE.value})
"""Actions that need a matched element before execution."""

ARRIVAL_CEILING_MS = 8000
"""Maximum time to watch a still screen for the learned destination."""

ARRIVAL_POLL_SECONDS = 0.35


@dataclass
class CachedStep:
    """A cached action and its run record."""

    action: Action
    source: ScreenReading
    record: StepRecord
    arrival_ms: int
    is_executed: bool
    entry_receipt: Optional[TextEntryReceipt] = None


def _resolve_action(
    cached: CachedAction, element: Optional[GUIElement], values: dict, scan: ScanInfo
) -> Optional[Action]:
    """Build an action at the target's current location, or refuse it."""
    if element is None:
        # Only a wait and a key press reach here without one, and neither
        # aims at a control.
        if cached.kind in TARGETED_KINDS:
            return None
        centre = Coordinates(x=0, y=0)
    else:
        box = element.bounding_box
        centre = Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)
    try:
        if cached.kind == ActionType.TAP.value:
            return TapAction(
                coordinates=centre,
                target=cached.described,
                count=cached.count,
                interval_ms=cached.interval_ms,
                reason="A saved path tapped this.",
            )
        if cached.kind == ActionType.TYPE.value:
            typed = values.get(cached.value_name or "")
            if not typed:
                return None
            return TypeAction(
                text=typed,
                coordinates=centre,
                target=cached.described,
                reason="A saved path typed here.",
            )
        if cached.kind == ActionType.KEY_EVENT.value:
            if cached.key not in REPLAYABLE_KEYCODES:
                return None
            return KeyEventAction(key=cached.key, reason="A saved path pressed this key.")
        if cached.kind == ActionType.WAIT.value:
            return WaitAction(
                duration=cached.duration or 1.0,
                reason="A saved path waited here.",
            )
        if cached.kind == ActionType.SWIPE.value:
            start_x = cached.start_x
            start_y = cached.start_y
            end_x = cached.end_x
            end_y = cached.end_y
            if start_x is None or start_y is None or end_x is None or end_y is None:
                return None
            return SwipeAction(
                start_x=round(start_x * scan.screen_width / BOUNDS_SCALE),
                start_y=round(start_y * scan.screen_height / BOUNDS_SCALE),
                end_x=round(end_x * scan.screen_width / BOUNDS_SCALE),
                end_y=round(end_y * scan.screen_height / BOUNDS_SCALE),
                reason="A saved path swiped here.",
            )
    except ValidationError as exc:
        logger.info("A saved step holds something no action takes: %s", exc.error_count())
        return None
    return None


def _element_from_relative_bounds(bounds: RelativeBounds, scan: ScanInfo) -> GUIElement:
    """Convert saved relative geometry into one current-screen target."""
    return GUIElement(
        element="Saved target",
        bounding_box=BoundingBox(
            x_min=round(bounds.left * scan.screen_width / BOUNDS_SCALE),
            y_min=round(bounds.top * scan.screen_height / BOUNDS_SCALE),
            x_max=round(bounds.right * scan.screen_width / BOUNDS_SCALE),
            y_max=round(bounds.bottom * scan.screen_height / BOUNDS_SCALE),
        ),
        confidence=1.0,
        type="saved_target",
        text="",
    )


@dataclass
class StepAttempt:
    """What asking the saved path for one step came to.

    `step` is set where a cached step ran. `refusal` is set where it did
    not, and the caller plans that one step instead.

    A refusal is never the end of the path. The next step asks again, from
    the screen the planner left.
    """

    step: Optional[CachedStep] = None
    refusal: Optional[str] = None
    source_mismatch: bool = False
    stopped: bool = False


def confirm_arrival(
    expected: ScreenSignature,
    screen: ScreenReading,
    names: ValueNames,
    key: Optional[bytes] = None,
    same_build: bool = False,
) -> Optional[str]:
    """Whether the last cached step arrived where it used to arrive.

    Asked of the next step's own reading. A cached step that landed
    somewhere else must not read as a step that worked. The path must not
    be walked on from a screen it did not expect.
    """
    if screen.scan_info is None:
        return "the screen it reached reported nothing"
    if not same_build and expected.description:
        # The next transition verifies this destination as its source in one
        # semantic request. The final destination is verified by the caller.
        return None
    if not expected.anchors:
        if same_build and expected.phash == screen.current_phash:
            return None
        return "the detector-free destination has no semantic description"
    refusal = replay_match.destination_reached(
        expected, screen.gui_elements, screen.scan_info, names, key
    )
    return refusal.reason if refusal else None


def select_transition(
    trace: ReplayTrace,
    position: int,
    screen: ScreenReading,
    names: ValueNames,
    key: Optional[bytes] = None,
) -> Optional[int]:
    """Which saved step best matches this screen, searched forward from here.

    Forward only, and only within this scenario's own chain. A path that
    matched a screen behind it would replay a circle. One that matched any
    screen in a ledger would replay another scenario's route.

    The search is what lets a planned step rejoin the path. The planner
    moves the app on one screen, and the step after it is found here
    rather than the whole remainder being abandoned.
    """
    if screen.scan_info is None:
        return None
    matches: list[int] = []
    for index in range(position, len(trace.transitions)):
        refusal = replay_match.screen_agrees(
            trace.transitions[index].source,
            screen.gui_elements,
            screen.scan_info,
            names,
            key,
        )
        if refusal is None:
            matches.append(index)
    if not matches:
        return None

    current_phash = screen.current_phash

    def phash_distance(index: int) -> int:
        expected_phash = trace.transitions[index].source.phash
        if not expected_phash or not current_phash:
            return 1 << 30
        try:
            return (int(expected_phash, 16) ^ int(current_phash, 16)).bit_count()
        except ValueError:
            return 0 if expected_phash == current_phash else 1 << 30

    return min(matches, key=lambda index: (phash_distance(index), index))


def take_cached_step(
    transition: CachedTransition,
    screen: ScreenReading,
    device_control,
    settle: Callable[[], Optional[str]],
    values: dict,
    names: ValueNames,
    step_log,
    step_number: int,
    quick_phash: Callable[[], Optional[str]],
    scrubber=None,
    key: Optional[bytes] = None,
    semantic_grounder=None,
    same_build: bool = False,
    model_image: Optional[bytes] = None,
    before_action: Optional[Callable[[Action], bool]] = None,
) -> StepAttempt:
    """Runs one saved step against the screen in front of the run.

    Refuses rather than guessing wherever the target, the count of
    lookalikes, or the state of the app has moved. Where it refuses
    nothing has been sent to the device.
    """
    scan = screen.scan_info
    if scan is None:
        return StepAttempt(refusal="this screen reported nothing to match on")

    source_matches_locally = (
        same_build
        and transition.source.phash is not None
        and transition.source.phash == screen.current_phash
    )
    if (
        transition.source.description
        and transition.action.kind not in TARGETED_KINDS
        and not source_matches_locally
        and (
            semantic_grounder is None
            or not semantic_grounder.matches_screen(
                model_image or screen.raw_img_bytes, transition.source.description
            )
        )
    ):
        return StepAttempt(refusal=SCREEN_MISMATCH, source_mismatch=True)

    # A wait and a key press touch no control, so neither needs one found.
    target_match = None
    target_element = None
    if transition.action.kind in TARGETED_KINDS:
        described = transition.action.described
        target_bounds = transition.action.target_bounds
        use_local_geometry = (
            described is not None
            and source_matches_locally
            and not transition.action.sensitive
            and target_bounds is not None
        )
        if use_local_geometry and target_bounds is not None:
            target_element = _element_from_relative_bounds(target_bounds, scan)
        elif described is not None:
            if semantic_grounder is None or not transition.source.description:
                return StepAttempt(refusal="the saved semantic target cannot be verified")
            rebound, refusal_reason = semantic_grounder.rebind(
                model_image or screen.raw_img_bytes,
                transition.source.description,
                described,
            )
            if rebound is None:
                return StepAttempt(
                    refusal=refusal_reason or "the target was not rebound",
                    source_mismatch=refusal_reason == SCREEN_MISMATCH,
                )
            target_element = rebound.placement.as_element()
        else:
            if transition.action.target is None:
                return StepAttempt(refusal="a saved step names no target")
            target_match, refusal = replay_match.find_target(
                transition.action.target, screen.gui_elements, scan, names, key
            )
            if target_match is None:
                return StepAttempt(
                    refusal=refusal.reason if refusal else "the target was not matched"
                )
            target_element = target_match.element

    action = _resolve_action(
        transition.action,
        target_element,
        values,
        scan,
    )
    if action is None:
        return StepAttempt(refusal=f"a saved step asks for {transition.action.kind}")

    # The app can move between the reading and the act. A cached step has
    # no planner to notice, so this is the only thing that would.
    stale_reason = targeting.stale_target_reason(device_control, screen, action)
    if stale_reason:
        return StepAttempt(refusal=stale_reason)
    if before_action is not None and not before_action(action):
        return StepAttempt(stopped=True)

    started_at = time.monotonic()
    failure_reason: Optional[str] = None
    entry_receipt: Optional[TextEntryReceipt] = None
    try:
        if isinstance(action, TypeAction) and scrubber is not None:
            scrubber.protect_typed_region(action.text, action.coordinates, screen.gui_elements)
        receipt = ActionOrchestrator(device_control, scrubber).execute(action)
        if isinstance(receipt, TextEntryReceipt):
            entry_receipt = receipt
    except Exception as exc:  # noqa: BLE001 - a saved path must not fail a run
        # The device can have moved before this raised, so the attempt is
        # recorded rather than dropped. A planner that was told nothing
        # happened would send the same tap again.
        failure_reason = f"a saved step would not run: {type(exc).__name__}"
    phash = settle()
    if failure_reason is None:
        deadline = started_at + min(transition.arrival_ms, ARRIVAL_CEILING_MS) / 1000
        while (
            action.type is ActionType.TAP
            and time.monotonic() < deadline
            and (
                phash == screen.current_phash
                or (same_build and transition.destination.phash != phash)
            )
        ):
            time.sleep(ARRIVAL_POLL_SECONDS)
            observed_phash = quick_phash()
            if observed_phash != phash:
                phash = settle() or observed_phash
    took_ms = (time.monotonic() - started_at) * 1000

    record = step_log.record_replayed_transition(
        n=step_number,
        source=StepSource.CACHE,
        screen_name="",
        phash=phash or screen.current_phash,
        action=action,
        description=(
            f"{transition.action.kind} {transition.action.described}".strip()
            if transition.action.described
            else (
                f"{transition.action.kind} {target_match.anchor.text}".strip()
                if target_match is not None
                else transition.action.kind
            )
        ),
        target=(
            transition.action.described
            or (target_match.anchor.text if target_match is not None else "")
        ),
        success=failure_reason is None,
        observation=failure_reason or "Replayed a saved step.",
        took_ms=took_ms,
    )
    return StepAttempt(
        step=CachedStep(
            action=action,
            source=screen,
            record=record,
            arrival_ms=round(took_ms),
            is_executed=failure_reason is None,
            entry_receipt=entry_receipt,
        ),
        refusal=failure_reason,
    )
