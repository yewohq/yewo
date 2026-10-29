"""Resolving a control the screen shows and the detector never found.

This runs after a magnified, anchor-bounded region numbered nothing. Every
cheap and reversible step has already failed. What remains is to ask the
device. Tap one candidate point. Read which element takes focus. Keep that
answer. Nothing is typed. Nothing is pressed on a grid cell alone.

A control found this way becomes an ordinary numbered target. The planner
learns only that an intervention happened. The grid is how the control was
found, not a way to name it. A planner that could see cells would choose them.
"""

import io
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from yewo.agent.actuation.target_not_listed import TargetNotListedAction
from yewo.agent.grounding import grid, marks
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.run.control_discovery import (
    ControlDiscoveryState,
    ControlDiscoveryTransaction,
)
from yewo.agent.artifacts import steps as step_artifacts

logger = logging.getLogger(__name__)

ENTER_TEXT = "enter_text"

EVIDENCE_FLOOR = 0.25
"""Confidence a detection needs before a cell may be pressed without proof.

A target to activate cannot be probed. The only guard left is that the
detector saw something there. A floor this low still refuses a cell with no
detection, which is the case the rule exists for."""

RESOLVED_NOTE = (
    "VISUAL INTERVENTION: a control in the {region} region was located by "
    "inspecting that region and confirming it on the device. It is numbered "
    "like any other target. Choose it by its number."
)

UNRESOLVED_NOTE = (
    "VISUAL INTERVENTION: the {region} region was inspected closely and "
    "confirmed on the device. No control was found there. Do not ask for that "
    "region again. Scroll, or choose a different way to reach the goal."
)


@dataclass(frozen=True)
class CellAttempt:
    """One cell, and what the device said about it."""

    index: int
    point: tuple[int, int]
    verdict: str
    bounds: Optional[tuple[int, int, int, int]] = None

    def as_record(self) -> dict:
        return {
            "cell": self.index,
            "point": list(self.point),
            "verdict": self.verdict,
            "bounds": list(self.bounds) if self.bounds else None,
        }


@dataclass(frozen=True)
class InterventionOutcome:
    """What the intervention cost and whether it found anything."""

    resolved: bool
    attempts: tuple[CellAttempt, ...]
    reason: str
    element: Optional[GUIElement] = None


class InterventionDirective(Enum):
    """What the agent loop must do after deterministic discovery."""

    PLAN = "plan"
    REOBSERVE = "reobserve"
    STOP = "stop"


def prepare(snapshot, ctx, device_control, perception, grounding):
    """Build and resolve one deterministic discovery transition."""
    from yewo.agent.step import observation

    transaction = ctx.control_discovery
    if (
        transaction is not None
        and transaction.state in (
            ControlDiscoveryState.REQUESTED,
            ControlDiscoveryState.PROBING,
        )
        and not is_same_screen(transaction.source_phash, snapshot.current_phash)
    ):
        transaction.fail(
            "The app screen changed after a focus probe. Remaining candidate "
            "coordinates were discarded instead of being replayed on stale UI."
        )
        return None, InterventionDirective.STOP

    observation.number_targets(snapshot, ctx, grounding, perception)
    if snapshot.intervention_failure_reason is not None:
        if transaction is not None and not transaction.is_terminal:
            transaction.fail(snapshot.intervention_failure_reason)
        return None, InterventionDirective.STOP
    materialize_verified(snapshot, ctx, device_control)
    if transaction is not None and transaction.state is ControlDiscoveryState.FAILED:
        return None, InterventionDirective.STOP
    record = resolve(snapshot, ctx, device_control)
    directive = record["directive"] if record is not None else InterventionDirective.PLAN
    return record, directive


def _probed_cell_indexes(ctx, plan) -> set:
    """Cells already tried for this screen, target and region."""
    transaction = ctx.control_discovery
    if transaction is None:
        raise RuntimeError("An intervention plan requires a control-discovery transaction.")
    transaction.begin_probing()
    return transaction.probed_cell_indexes


def _build_control_element(bounds, point, description: str) -> GUIElement:
    """The resolved control, as the kind of element everything else expects.

    Built from the driver's bounds where it gave them, because those are the
    control's real edges. The cell is only a fallback: it says where the tap
    went, not where the control is.
    """
    box = bounds or (point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1)
    return GUIElement(
        element="Control",
        type="focus_probe",
        confidence=1.0,
        text=description,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def _probe_for_text(
    device_control, candidates: Sequence[grid.Candidate], probed: set, description: str
) -> InterventionOutcome:
    """Try exactly one unvisited cell against the current observation."""
    for candidate in candidates:
        if candidate.index in probed:
            continue
        point = candidate.point
        try:
            probe = device_control.probe_focus(*point)
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.warning("Cell %d could not be probed: %s", candidate.index, exc)
            return InterventionOutcome(
                resolved=False,
                attempts=(CellAttempt(candidate.index, point, "unreadable"),),
                reason="probe_unreadable",
            )
        attempt = CellAttempt(candidate.index, point, probe.verdict, probe.bounds)
        if probe.is_focused:
            logger.info("Cell %d holds focus. The control is at %s.", candidate.index, probe.bounds)
            return InterventionOutcome(
                resolved=True,
                attempts=(attempt,),
                reason="focus_verified",
                element=_build_control_element(probe.bounds, point, description),
            )
        return InterventionOutcome(
            resolved=False,
            attempts=(attempt,),
            reason=probe.verdict,
        )
    return InterventionOutcome(resolved=False, attempts=(), reason="no_cells_left")


def _commit_on_evidence(
    candidates: Sequence[grid.Candidate], probed: set, description: str
) -> InterventionOutcome:
    """The one cell a target that cannot be probed may be allowed.

    A press cannot be undone, so trying several is never safe. A cell with a
    detection under it is worth one attempt. A cell with none is not, and the
    honest answer is that the control was not found.
    """
    with_detections = [
        (candidate, candidate.detection)
        for candidate in candidates
        if candidate.index not in probed
        and candidate.detection is not None
        and candidate.detection.confidence >= EVIDENCE_FLOOR
    ]
    if not with_detections:
        return InterventionOutcome(
            resolved=False, attempts=(), reason="no_detection_under_any_cell"
        )
    best, detection = max(with_detections, key=lambda pair: pair[1].confidence)
    box = detection.bounding_box
    logger.info(
        "Cell %d covers a detection numbering dropped. Offering it as the target.",
        best.index,
    )
    return InterventionOutcome(
        resolved=True,
        attempts=(CellAttempt(best.index, best.point, "accepted", None),),
        reason="detection_under_cell",
        element=_build_control_element(
            (box.x_min, box.y_min, box.x_max, box.y_max), best.point, description
        ),
    )


def _renumber_planner_screen(snapshot, ctx, found: GUIElement) -> Optional[int]:
    """Put the full screen back, with the resolved control numbered on it.

    The magnified view is finished. Leaving it up would make the planner
    choose from a crop. The target it wanted is now an ordinary number on the
    ordinary screen.
    """
    elements = list(snapshot.gui_elements) + [found]
    snapshot.marks, snapshot.planner_img_bytes = marks.number(
        elements,
        snapshot.raw_img_bytes,
        keyboard_open=ctx.keyboard_open,
        screen_height=ctx.screen_height,
        refused=ctx.refused_marks.get(snapshot.current_phash) or [],
        include_keyboard_targets=ctx.keyboard_targets_expanded,
    )
    snapshot.planner_elements = elements
    snapshot.planner_screen_height = ctx.screen_height
    box = marks.box_of(found)
    for mark in snapshot.marks:
        if marks.is_same_element(mark.element, box):
            return mark.index
    return None


def materialize_verified(snapshot, ctx, device_control) -> Optional[int]:
    """Number a verified target only on the fresh post-probe observation.

    The recovery-only element is deliberately absent from ``gui_elements``.
    Durable screen identity and replay learning therefore see only product
    evidence produced by perception.
    """
    transaction = ctx.control_discovery
    if transaction is None or transaction.state is not ControlDiscoveryState.VERIFYING:
        return None
    try:
        reading = device_control.read_focused_field()
        current_bounds = device_control.perception_box(reading.bounds)
    except Exception as exc:  # noqa: BLE001 - optional driver evidence
        logger.warning("Could not re-read the verified field after settlement: %s", exc)
        current_bounds = None
    if current_bounds is None:
        transaction.fail(
            "The focus probe succeeded, but the field did not retain verifiable "
            "focus on the settled screen."
        )
        return None
    found = _build_control_element(current_bounds, (0, 0), transaction.description)
    numbered = _renumber_planner_screen(snapshot, ctx, found)
    if numbered is None:
        transaction.fail(
            "The driver verified the control, but it could not be represented "
            "as a unique planner target."
        )
        return None
    transaction.resolve(current_bounds)
    snapshot.view_note = RESOLVED_NOTE.format(region=transaction.region)
    return numbered


def _has_unprobed_candidate(candidates: Sequence[grid.Candidate], probed: set) -> bool:
    return any(candidate.index not in probed for candidate in candidates)


def resolve(snapshot, ctx, device_control) -> Optional[dict]:
    """Run the intervention this snapshot is carrying, if it is carrying one.

    Returns what the run directory records, or None when there was nothing to
    do. Either way the snapshot ends in the state the planner should see. That
    is a numbered screen holding the control, or the same screen and a note
    that the region was searched and held nothing.
    """
    plan = snapshot.intervention
    if plan is None:
        return None
    snapshot.intervention = None

    probed = _probed_cell_indexes(ctx, plan)
    transaction = ctx.control_discovery
    if plan.intent == ENTER_TEXT:
        outcome = _probe_for_text(device_control, plan.candidates, probed, plan.description)
    else:
        outcome = _commit_on_evidence(plan.candidates, probed, plan.description)

    directive = InterventionDirective.PLAN
    numbered = None
    reason = outcome.reason
    if outcome.attempts:
        attempt = outcome.attempts[0]
        bounds = (
            marks.box_of(outcome.element)
            if outcome.resolved and outcome.element is not None
            else None
        )
        transaction.record_probe(attempt.index, bounds)
    if plan.intent == ENTER_TEXT and outcome.resolved and outcome.element is not None:
        directive = InterventionDirective.REOBSERVE
        snapshot.view_note = "VISUAL INTERVENTION: focus was verified; re-observing the app."
    elif plan.intent != ENTER_TEXT and outcome.resolved and outcome.element is not None:
        numbered = _renumber_planner_screen(snapshot, ctx, outcome.element)
        if numbered is not None:
            box = marks.box_of(outcome.element)
            transaction.resolve(box)
            snapshot.view_note = RESOLVED_NOTE.format(region=plan.region)
        else:
            reason = "numbering_dropped_the_control"
            directive = InterventionDirective.STOP
            transaction.fail(
                "The detector found the requested control, but it could not be "
                "represented as a unique planner target."
            )
    else:
        if plan.intent != ENTER_TEXT:
            directive = InterventionDirective.STOP
            snapshot.view_note = UNRESOLVED_NOTE.format(region=plan.region)
            transaction.fail(
                "Deterministic control discovery found no detection that could "
                "authorize pressing the requested control."
            )
        elif _has_unprobed_candidate(plan.candidates, probed):
            directive = InterventionDirective.REOBSERVE
            snapshot.view_note = (
                "VISUAL INTERVENTION: one candidate was rejected; re-observing "
                "before the next bounded attempt."
            )
        else:
            directive = InterventionDirective.STOP
            snapshot.view_note = UNRESOLVED_NOTE.format(region=plan.region)
            transaction.fail(
                f"Deterministic control discovery exhausted the bounded candidates "
                f"in the {plan.region} region without verifying the requested control."
            )
            logger.info("Intervention on the %s region found nothing: %s", plan.region, reason)
    if plan.dropped_cells:
        logger.info(
            "The %s region had room for %d more cells than the budget allows.",
            plan.region,
            plan.dropped_cells,
        )
    return {
        "region": plan.region,
        "intent": plan.intent,
        "description": plan.description,
        "crop": list(plan.crop),
        "scale": plan.scale,
        "derivation": plan.derived,
        "cells": [
            {"index": candidate.index, "box": list(candidate.box),
             "detected": candidate.detection is not None}
            for candidate in plan.candidates
        ],
        "cells_beyond_budget": plan.dropped_cells,
        "attempts": [attempt.as_record() for attempt in outcome.attempts],
        "resolved": outcome.resolved,
        "reason": reason,
        "numbered_as": numbered,
        "directive": directive,
        "plan": plan,
        "outcome": outcome,
    }


def split_record(record: dict) -> tuple[dict, dict]:
    """Split the record into what is written and what stays in memory."""
    stored = {
        key: value
        for key, value in record.items()
        if key not in ("directive", "plan", "outcome")
    }
    return stored, {"plan": record["plan"], "outcome": record["outcome"]}


def overlay_bytes(record: dict) -> bytes:
    """The grid as it ended, coloured by what each cell turned out to be."""
    plan = record["plan"]
    outcome = record["outcome"]
    verdicts = {attempt.index: attempt.verdict for attempt in outcome.attempts}
    close_up = Image.open(io.BytesIO(plan.close_up))
    return grid.overlay(close_up, plan.candidates, plan.crop, plan.scale, verdicts)


def save_records(record: Optional[dict], images, session_dir: Path, step: int, timestamp: str) -> dict:
    """Persist the overlay and the record, redacted like any other artifact.

    An artifact must never fail a run. A region that cannot be drawn is logged
    and skipped rather than raised.
    """
    if record is None:
        return {}
    files = {}
    record_fields, _ = split_record(record)
    stem = f"step{step + 1:03d}_{timestamp}_intervention"
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{stem}.json"
        path.write_text(
            json.dumps(record_fields, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files["intervention"] = path.name
    except Exception as exc:  # noqa: BLE001 - an artifact must never fail a run
        logger.warning("Could not save the intervention record: %s", exc)
    try:
        path = session_dir / f"{stem}_grid.png"
        images.save_redacted_bytes(overlay_bytes(record), list(record["plan"].redaction), path)
        files["intervention_grid"] = path.name
    except Exception as exc:  # noqa: BLE001 - an artifact must never fail a run
        logger.warning("Could not save the intervention grid: %s", exc)
    return files


def record_unmatched_target(images, action, decision, snapshot, ctx, session_dir, step) -> None:
    """Keep the screen a target was missing from, and arrange a closer look.

    The screen is kept even when no closer look follows. A gap nobody can
    study later is a gap that stays open.
    """
    step_artifacts.save_unmatched_screen(
        images,
        session_dir / f"step{step + 1:03d}_{snapshot.timestamp}_not_listed.png",
        snapshot,
    )
    if not isinstance(action, TargetNotListedAction):
        return
    if ctx.keyboard_open or not snapshot.current_phash:
        return
    anchors = {mark.index: marks.box_of(mark.element) for mark in snapshot.marks}
    transaction = ControlDiscoveryTransaction(
        source_phash=snapshot.current_phash,
        region=action.region,
        description=decision.target or "",
        intent=action.intent,
        anchor_above=anchors.get(action.anchor_above),
        anchor_below=anchors.get(action.anchor_below),
    )
    if ctx.control_discovery is not None and ctx.control_discovery.key == transaction.key:
        return
    ctx.control_discovery = transaction
    logger.info("Will inspect the %s region more closely on the next look.", action.region)
