"""Finding a control the detector missed, by asking the device instead."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from yewo.agent.actuation.target_not_listed import TargetNotListedAction
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.run.control_discovery import (
    ControlDiscoveryState,
    ControlDiscoveryTransaction,
)
from yewo.agent.run.context import RunContext

CROP = (36, 1156, 1044, 1286)
SCREEN = (1080, 2400)


def _el(box, confidence=0.9):
    from yewo.agent.perception.elements import BoundingBox, GUIElement

    return GUIElement(
        element="View",
        type="yolo_detection",
        confidence=confidence,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def _png(size):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeDevice:
    """A device that focuses one rectangle and nothing else."""

    def __init__(self, focusable=None, bounds=None, raises=False):
        self.focusable = focusable
        self.bounds = bounds
        self.raises = raises
        self.taps = []

    def probe_focus(self, x, y):
        from yewo.agent.device.focus import FocusProbe

        self.taps.append((x, y))
        if self.raises:
            raise RuntimeError("session gone")
        if self.focusable and self.focusable[0] <= x <= self.focusable[2] and (
            self.focusable[1] <= y <= self.focusable[3]
        ):
            return FocusProbe(is_focused=True, focus_changed=True, bounds=self.bounds or self.focusable)
        return FocusProbe(is_focused=False, focus_changed=False)

    def read_focused_field(self):
        from yewo.agent.device.focus import FocusedFieldReading

        return FocusedFieldReading(value="", bounds=self.bounds or self.focusable)

    def perception_box(self, bounds):
        return bounds


def _snapshot(elements=(), scale=3):
    from yewo.agent.grounding import grid
    from yewo.agent.step.observation import InterventionPlan, PerceptionSnapshot

    raw = _png(SCREEN)
    close_up = _png(((CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale))
    snapshot = PerceptionSnapshot(
        gui_elements=list(elements),
        marks=[],
        raw_img_bytes=raw,
        planner_img_bytes=raw,
        annotated_img=Image.new("RGB", SCREEN, "white"),
        current_phash="screen",
        timestamp="20260101_000000",
    )
    snapshot.intervention = InterventionPlan(
        region="middle_left",
        crop=CROP,
        scale=scale,
        close_up=close_up,
        candidates=tuple(grid.build(CROP, [])),
        intent="enter_text",
        description="repetitions field",
    )
    return snapshot


def _context(armed=True):
    ctx = RunContext(
        history=AgentHistory(),
        loop_detector=RollingLoopDetector(),
        current_phash="screen",
        screen_height=SCREEN[1],
    )
    if armed:
        ctx.control_discovery = ControlDiscoveryTransaction(
            source_phash="screen",
            region="middle_left",
            description="repetitions field",
            intent="enter_text",
        )
    return ctx


class TextTargetTests(unittest.TestCase):
    """A field can be probed, because a tap that only moves the cursor is undoable."""

    def test_a_cell_that_holds_focus_becomes_a_numbered_target(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=CROP, bounds=(42, 1170, 1038, 1270))
        ctx = _context()

        record = intervention.resolve(snapshot, ctx, device)

        self.assertTrue(record["resolved"])
        self.assertEqual(record["reason"], "focus_verified")
        self.assertIsNone(record["numbered_as"])
        self.assertEqual(record["directive"], intervention.InterventionDirective.REOBSERVE)
        self.assertEqual(ctx.control_discovery.state, ControlDiscoveryState.VERIFYING)

    def test_the_control_takes_the_bounds_the_driver_reported(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=CROP, bounds=(42, 1170, 1038, 1270))

        ctx = _context()
        intervention.resolve(snapshot, ctx, device)

        self.assertEqual(ctx.control_discovery.verified_bounds, (42, 1170, 1038, 1270))
        self.assertFalse(any(e.type == "focus_probe" for e in snapshot.gui_elements))

    def test_probing_stops_at_the_first_cell_that_answers(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=CROP, bounds=CROP)

        intervention.resolve(snapshot, _context(), device)

        self.assertEqual(len(device.taps), 1)

    def test_one_observation_probes_exactly_one_candidate(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=None)

        record = intervention.resolve(snapshot, _context(), device)

        self.assertEqual(len(device.taps), 1)
        self.assertFalse(record["resolved"])
        self.assertEqual(record["reason"], "no_focus")
        self.assertEqual(record["directive"], intervention.InterventionDirective.REOBSERVE)

    def test_reobservations_probe_unique_cells_for_the_same_target(self):
        from yewo.agent.step import intervention

        ctx = _context()
        device = FakeDevice(focusable=None)

        intervention.resolve(_snapshot(), ctx, device)
        first_point = device.taps[-1]
        intervention.resolve(_snapshot(), ctx, device)

        self.assertEqual(len(device.taps), 2)
        self.assertNotEqual(device.taps[-1], first_point)

    def test_exhausting_unique_candidates_returns_a_typed_stop(self):
        from yewo.agent.step import intervention

        ctx = _context()
        device = FakeDevice(focusable=None)
        record = None
        candidate_count = len(_snapshot().intervention.candidates)
        for _ in range(candidate_count):
            record = intervention.resolve(_snapshot(), ctx, device)
        self.assertEqual(len(device.taps), candidate_count)
        self.assertEqual(len(set(device.taps)), candidate_count)
        self.assertEqual(record["directive"], intervention.InterventionDirective.STOP)
        self.assertIn("exhausted", ctx.control_discovery.failure_reason)

    def test_a_driver_that_cannot_answer_does_not_end_the_search(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(raises=True)

        record = intervention.resolve(snapshot, _context(), device)

        self.assertFalse(record["resolved"])
        self.assertTrue(all(a["verdict"] == "unreadable" for a in record["attempts"]))

    def test_the_planner_is_told_an_intervention_happened_and_not_how(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()

        intervention.resolve(snapshot, _context(), FakeDevice(focusable=CROP))

        self.assertIn("VISUAL INTERVENTION", snapshot.view_note)
        for leak in ("grid", "cell", "coordinate"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, snapshot.view_note.lower())

    def test_a_rejected_candidate_requires_a_fresh_observation(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()

        intervention.resolve(snapshot, _context(), FakeDevice(focusable=None))

        self.assertIn("re-observing", snapshot.view_note)


class ActivateTargetTests(unittest.TestCase):
    def _plan(self, elements):
        from dataclasses import replace

        from yewo.agent.grounding import grid

        snapshot = _snapshot()
        snapshot.intervention = replace(
            snapshot.intervention,
            intent="activate",
            candidates=tuple(grid.build(CROP, elements)),
        )
        return snapshot

    def test_a_cell_with_nothing_under_it_is_never_pressed(self):
        from yewo.agent.step import intervention

        snapshot = self._plan([])
        device = FakeDevice(focusable=CROP)
        ctx = _context()

        record = intervention.resolve(snapshot, ctx, device)

        self.assertEqual(device.taps, [])
        self.assertFalse(record["resolved"])
        self.assertEqual(record["reason"], "no_detection_under_any_cell")
        self.assertEqual(record["directive"], intervention.InterventionDirective.STOP)
        self.assertEqual(ctx.control_discovery.state, ControlDiscoveryState.FAILED)

    def test_a_cell_over_a_dropped_detection_becomes_the_target(self):
        from yewo.agent.step import intervention

        snapshot = self._plan([_el((300, 1180, 800, 1260))])

        record = intervention.resolve(snapshot, _context(), FakeDevice())

        self.assertTrue(record["resolved"])
        self.assertEqual(record["reason"], "detection_under_cell")

    def test_a_detection_below_the_confidence_floor_is_not_evidence(self):
        from yewo.agent.step import intervention
        from yewo.agent.step.intervention import EVIDENCE_FLOOR

        snapshot = self._plan([_el((300, 1180, 800, 1260), confidence=EVIDENCE_FLOOR / 2)])

        record = intervention.resolve(snapshot, _context(), FakeDevice())

        self.assertFalse(record["resolved"])

    def test_only_one_cell_is_ever_offered(self):
        from yewo.agent.step import intervention

        snapshot = self._plan([_el(CROP)])

        record = intervention.resolve(snapshot, _context(), FakeDevice())

        self.assertEqual(len(record["attempts"]), 1)

class NumberingTests(unittest.TestCase):
    """A driver-found control must survive numbering without becoming evidence."""
    def test_verified_control_is_planner_only_on_the_fresh_screen(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=CROP, bounds=(42, 1170, 1038, 1270))
        ctx = _context()
        intervention.resolve(snapshot, ctx, device)
        numbered = intervention.materialize_verified(snapshot, ctx, device)
        self.assertIsNotNone(numbered)
        self.assertTrue(any(e.type == "focus_probe" for e in snapshot.planner_elements))
        self.assertFalse(any(e.type == "focus_probe" for e in snapshot.gui_elements))

    def test_lost_focus_after_settlement_is_not_materialized(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        device = FakeDevice(focusable=CROP, bounds=CROP)
        ctx = _context()
        intervention.resolve(snapshot, ctx, device)
        device.bounds = None
        device.focusable = None
        self.assertIsNone(intervention.materialize_verified(snapshot, ctx, device))
        self.assertIn("did not retain", ctx.control_discovery.failure_reason)

    def test_a_control_swallowed_by_its_container_counts_as_not_found(self):
        from yewo.agent.step import intervention

        container = _el((100, 1100, 900, 1400))
        snapshot = _snapshot([container])
        device = FakeDevice(focusable=CROP, bounds=(300, 1200, 700, 1300))

        ctx = _context()
        record = intervention.resolve(snapshot, ctx, device)
        self.assertTrue(record["resolved"])
        numbered = intervention.materialize_verified(snapshot, ctx, device)

        self.assertIsNone(numbered)
        self.assertIn("could not be represented", ctx.control_discovery.failure_reason)


class UnmatchedTargetTests(unittest.TestCase):
    def _saved(self, action):
        from yewo.agent.step import intervention

        class Images:
            def save_redacted_bytes(self, data, elements, path):
                path.write_bytes(data)

        ctx = _context(armed=False)
        snapshot = _snapshot()
        decision = SimpleNamespace(target="repetitions field")
        with tempfile.TemporaryDirectory() as directory:
            intervention.record_unmatched_target(
                Images(), action, decision, snapshot, ctx, Path(directory), 0
            )
            kept = [path.name for path in Path(directory).iterdir()]
        return ctx, kept

    def test_the_screen_is_kept_and_the_region_is_armed(self):
        action = TargetNotListedAction(
            description="repetitions field",
            region="middle_left",
            intent="enter_text",
            reason="enter the repetition count",
        )

        ctx, kept = self._saved(action)

        self.assertEqual(kept, ["step001_20260101_000000_not_listed.png"])
        self.assertEqual(ctx.control_discovery.region, "middle_left")
        self.assertEqual(ctx.control_discovery.intent, "enter_text")

    def test_a_target_named_in_words_keeps_the_screen_and_arms_nothing(self):
        """Describe mode can also leave a target unmatched, with no region to inspect."""
        ctx, kept = self._saved(SimpleNamespace(type="tap"))

        self.assertEqual(kept, ["step001_20260101_000000_not_listed.png"])
        self.assertIsNone(ctx.control_discovery)

    def test_one_screen_and_target_are_only_armed_once(self):
        action = TargetNotListedAction(
            description="repetitions field",
            region="middle_left",
            intent="enter_text",
            reason="enter the repetition count",
        )

        ctx, _ = self._saved(action)
        original = ctx.control_discovery
        from yewo.agent.step import intervention

        class Images:
            def save_redacted_bytes(self, data, elements, path):
                path.write_bytes(data)

        with tempfile.TemporaryDirectory() as directory:
            intervention.record_unmatched_target(
                Images(),
                action,
                SimpleNamespace(target="repetitions field"),
                _snapshot(),
                ctx,
                Path(directory),
                1,
            )

        self.assertIs(ctx.control_discovery, original)


class RecordTests(unittest.TestCase):
    def test_nothing_to_resolve_records_nothing(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        snapshot.intervention = None

        self.assertIsNone(intervention.resolve(snapshot, _context(), FakeDevice()))

    def test_the_record_lists_every_cell_the_budget_left_out(self):
        from dataclasses import replace

        from yewo.agent.step import intervention

        snapshot = _snapshot()
        snapshot.intervention = replace(snapshot.intervention, dropped_cells=7)

        record = intervention.resolve(snapshot, _context(), FakeDevice())

        self.assertEqual(record["cells_beyond_budget"], 7)

    def test_the_overlay_is_drawn_over_the_magnified_crop(self):
        from yewo.agent.step import intervention

        snapshot = _snapshot()
        expected = Image.open(io.BytesIO(snapshot.intervention.close_up)).size

        record = intervention.resolve(snapshot, _context(), FakeDevice(focusable=CROP))
        drawn = Image.open(io.BytesIO(intervention.overlay_bytes(record)))

        self.assertEqual(drawn.size, expected)

    def test_the_written_record_carries_no_objects_a_reader_cannot_read(self):
        import json

        from yewo.agent.step import intervention

        record = intervention.resolve(_snapshot(), _context(), FakeDevice())
        record_fields, _ = intervention.split_record(record)

        json.dumps(record_fields)


class ArtifactTests(unittest.TestCase):
    """What the run directory keeps, and what it never keeps unredacted."""

    def _images(self):
        class Images:
            def __init__(self):
                self.redacted = []

            def save_redacted_bytes(self, data, elements, path):
                self.redacted.append((len(elements), path))
                path.write_bytes(data)

        return Images()

    def test_both_files_are_written_and_named_after_the_step(self):
        import tempfile
        from pathlib import Path

        from yewo.agent.step import intervention

        record = intervention.resolve(_snapshot(), _context(), FakeDevice())
        with tempfile.TemporaryDirectory() as directory:
            files = intervention.save_records(
                record, self._images(), Path(directory), 4, "20260101_000000"
            )
            written = sorted(path.name for path in Path(directory).iterdir())

        self.assertEqual(
            written,
            [
                "step005_20260101_000000_intervention.json",
                "step005_20260101_000000_intervention_grid.png",
            ],
        )
        self.assertEqual(set(files), {"intervention", "intervention_grid"})

    def test_the_grid_goes_through_redaction_with_the_crop_its_evidence_sits_in(self):
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        from yewo.agent.step import intervention

        snapshot = _snapshot()
        snapshot.intervention = replace(
            snapshot.intervention, redaction=(_el((10, 10, 40, 40)),)
        )
        record = intervention.resolve(snapshot, _context(), FakeDevice())
        images = self._images()
        with tempfile.TemporaryDirectory() as directory:
            intervention.save_records(record, images, Path(directory), 0, "t")

        self.assertEqual([count for count, _ in images.redacted], [1])

    def test_nothing_is_written_when_there_was_no_intervention(self):
        import tempfile
        from pathlib import Path

        from yewo.agent.step import intervention

        with tempfile.TemporaryDirectory() as directory:
            files = intervention.save_records(
                None, self._images(), Path(directory), 0, "t"
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

        self.assertEqual(files, {})

    def test_a_file_that_cannot_be_written_does_not_fail_the_run(self):
        from pathlib import Path

        from yewo.agent.step import intervention

        record = intervention.resolve(_snapshot(), _context(), FakeDevice())

        files = intervention.save_records(
            record, self._images(), Path("/nonexistent/\x00"), 0, "t"
        )

        self.assertEqual(files, {})

    def test_the_overlay_marks_a_cell_that_answered_apart_from_one_that_did_not(self):
        from yewo.agent.grounding.grid import CELL_COLOR, HIT_COLOR, MISS_COLOR, build, overlay

        candidates = build(CROP)
        scale = 3
        size = ((CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale)
        verdicts = {candidates[0].index: "accepted", candidates[1].index: "no_focus"}

        drawn = Image.open(
            io.BytesIO(overlay(Image.new("RGB", size, "white"), candidates, CROP, scale, verdicts))
        )
        colours = {colour for _, colour in drawn.convert("RGB").getcolors(1 << 20)}

        self.assertIn(HIT_COLOR, colours)
        self.assertIn(MISS_COLOR, colours)
        self.assertIn(CELL_COLOR, colours)

    def test_the_overlay_marks_a_cell_a_detection_already_sits_under(self):
        from yewo.agent.grounding.grid import EVIDENCE_COLOR, build, overlay

        scale = 3
        size = ((CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale)
        candidates = build(CROP, [_el((300, 1180, 800, 1260))])

        drawn = Image.open(
            io.BytesIO(overlay(Image.new("RGB", size, "white"), candidates, CROP, scale))
        )
        colours = {colour for _, colour in drawn.convert("RGB").getcolors(1 << 20)}

        self.assertIn(EVIDENCE_COLOR, colours)
