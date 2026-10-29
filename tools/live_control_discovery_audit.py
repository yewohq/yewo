#!/usr/bin/env python3
"""Exercise missing-control discovery on the trainer form without a planner."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from yewo.agent.actuation.target_not_listed import TargetNotListedAction
from yewo.agent.device.platform import Platform
from yewo.agent.device.session import DeviceSession
from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.perception.blind import for_grounding
from yewo.agent.perception import settle
from yewo.agent.run.context import RunContext
from yewo.agent.step import intervention, observation


class _EvidenceImages:
    def save_redacted_bytes(self, data, _elements, path):
        path.write_bytes(data)


def _save_snapshot(snapshot, path: Path) -> None:
    path.write_bytes(snapshot.raw_img_bytes)


def _mark_containing(snapshot, fragment: str):
    fragment = fragment.casefold()
    for mark in snapshot.marks:
        if fragment in (mark.element.text or "").casefold():
            return mark
    raise RuntimeError(f"No numbered control contained {fragment!r}.")


def _capture(perception, previous, ctx):
    snapshot = observation.capture(perception, None, previous)
    ctx.current_phash = snapshot.current_phash
    ctx.screen_height = snapshot.scan_info.screen_height
    ctx.keyboard_open = perception.device_session.device_control.is_keyboard_open()
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Pixel_8")
    parser.add_argument(
        "--app",
        type=Path,
        default=Path("eval/apps/android/trainer-app-1645-buggy.apk"),
    )
    parser.add_argument("--output", type=Path, default=Path("eval/runs/phase5-control-audit"))
    args = parser.parse_args()

    session = DeviceSession(
        device_name=args.device,
        app_path=str(args.app.resolve()),
        platform=Platform.ANDROID,
        session_dir=args.output,
        keep_device=False,
        keep_app=True,
    )
    try:
        device = session.start()
        for point in ((72, 202), (154, 671), (962, 2219)):
            device.tap(*point)
            settle.settle(device, 4.0)

        perception = for_grounding(GROUNDING_MARKS, session)
        ctx = RunContext(history=AgentHistory(), loop_detector=RollingLoopDetector())
        initial = _capture(perception, None, ctx)
        observation.number_targets(initial, ctx, GROUNDING_MARKS, perception)
        above = _mark_containing(initial, "repea")
        below = _mark_containing(initial, "pause")
        _save_snapshot(initial, session.screenshots_dir / "01_form.png")

        action = TargetNotListedAction(
            description="editable repetitions field between its label and Pause",
            region="upper_left",
            intent="enter_text",
            anchor_above=above.index,
            anchor_below=below.index,
            reason="enter a repetition count",
        )
        intervention.record_unmatched_target(
            _EvidenceImages(),
            action,
            SimpleNamespace(target="repetitions field"),
            initial,
            ctx,
            session.screenshots_dir,
            0,
        )

        probing = _capture(perception, initial.current_phash, ctx)
        record, directive = intervention.prepare(
            probing, ctx, device, perception, GROUNDING_MARKS
        )
        if directive is not intervention.InterventionDirective.REOBSERVE:
            raise RuntimeError(f"Expected a verified probe, got {directive.value}.")
        intervention.save_records(
            record, _EvidenceImages(), session.screenshots_dir, 1, probing.timestamp
        )
        settle.settle(device, 4.0)

        verified = _capture(perception, probing.current_phash, ctx)
        _, directive = intervention.prepare(
            verified, ctx, device, perception, GROUNDING_MARKS
        )
        if directive is not intervention.InterventionDirective.PLAN:
            raise RuntimeError(f"Expected a materialized control, got {directive.value}.")
        focus_mark = next(
            mark for mark in verified.marks if mark.element.type == "focus_probe"
        )
        _save_snapshot(verified, session.screenshots_dir / "02_verified_focus.png")

        device.type("12", *focus_mark.center)
        settle.settle(device, 4.0)
        reading = device.read_focused_field()
        if reading is None or "12" not in reading.value:
            raise RuntimeError("The verified field did not retain the entered value.")
        final = _capture(perception, verified.current_phash, ctx)
        _save_snapshot(final, session.screenshots_dir / "03_value_retained.png")

        result = {
            "status": "passed",
            "model_calls": 0,
            "source_phash": initial.current_phash,
            "final_phash": final.current_phash,
            "anchor_above": above.index,
            "anchor_below": below.index,
            "probed_cells": sorted(ctx.control_discovery.probed_cell_indexes),
            "probe_budget": ctx.control_discovery.probe_budget,
            "terminal": ctx.control_discovery.state.value,
            "verified_bounds": list(ctx.control_discovery.verified_bounds),
            "retained_value": reading.value,
        }
        session.manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(session.layout.root)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
