"""Semantic rebinding rules for changed builds and sensitive actions."""

import unittest
from types import SimpleNamespace

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.grounding.description import ReboundTarget
from yewo.agent.grounding.placement import Placement
from yewo.agent.memory import replay_learning, replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, ScanInfo
from yewo.agent.run.record import StepLog

WIDTH, HEIGHT = 1000, 2000


def screen_read(phash):
    return SimpleNamespace(
        gui_elements=[],
        scan_info=ScanInfo(
            mode="coordinates",
            reason="step",
            phash=phash,
            screen_width=WIDTH,
            screen_height=HEIGHT,
            element_count=0,
        ),
        current_phash=phash,
        raw_img_bytes=b"screen",
    )


class Device:
    def __init__(self):
        self.sent = []

    def get_screenshot(self):
        return b"not a readable image"


class Orchestrator:
    def __init__(self, device_control, scrubber=None):
        self.device = device_control

    def execute(self, action):
        self.device.sent.append(action)


def transition(*, source_phash, sensitive=False):
    return replay_store.CachedTransition(
        position=1,
        source=replay_store.ScreenSignature(description="Checkout screen", phash=source_phash),
        action=replay_store.CachedAction(
            kind="tap",
            described="Pay button" if sensitive else "Continue button",
            target_bounds=replay_store.RelativeBounds(left=100, top=100, right=300, bottom=200),
            sensitive=sensitive,
        ),
        destination=replay_store.ScreenSignature(description="Receipt screen", phash="after"),
        arrival_ms=0,
    )


class SemanticReplayTests(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        self.real_orchestrator = replay_walk.ActionOrchestrator
        replay_walk.ActionOrchestrator = Orchestrator
        self.addCleanup(setattr, replay_walk, "ActionOrchestrator", self.real_orchestrator)

    def _rebound(self):
        return ReboundTarget(
            placement=Placement(
                box=BoundingBox(x_min=400, y_min=600, x_max=600, y_max=700),
                present=1.0,
                box_confidence=1.0,
            ),
            screen_score=1.0,
            uniqueness_score=1.0,
        )

    def _take(self, cached_transition, grounder, *, same_build):
        return replay_walk.take_cached_step(
            transition=cached_transition,
            screen=screen_read(cached_transition.source.phash or "current"),
            device_control=self.device,
            settle=lambda: "after",
            values={},
            names=replay_store.ValueNames(),
            step_log=StepLog(),
            step_number=1,
            quick_phash=lambda: "after",
            semantic_grounder=grounder,
            same_build=same_build,
            model_image=b"redacted screen",
        )

    def test_a_changed_build_rebinds_a_detector_free_target(self):
        grounder = SimpleNamespace(rebind=lambda *_args: (self._rebound(), None))

        attempt = self._take(transition(source_phash="old"), grounder, same_build=False)

        self.assertIsNotNone(attempt.step)
        self.assertEqual(self.device.sent[0].coordinates, Coordinates(x=500, y=650))

    def test_a_sensitive_same_build_step_uses_semantic_rebinding(self):
        calls = []

        def rebind(*args):
            calls.append(args)
            return self._rebound(), None

        attempt = self._take(
            transition(source_phash="same", sensitive=True),
            SimpleNamespace(rebind=rebind),
            same_build=True,
        )

        self.assertIsNotNone(attempt.step)
        self.assertEqual(len(calls), 1)

    def test_a_changed_build_verifies_a_targetless_marks_step_semantically(self):
        bounds = replay_store.RelativeBounds(
            left=100, top=100, right=300, bottom=200
        )
        cached_transition = replay_store.CachedTransition(
            position=1,
            source=replay_store.ScreenSignature(
                description="Checkout screen",
                phash="old",
                anchors=(
                    replay_store.ElementAnchor(
                        role="button", text="digest", bounds=bounds, twins=1
                    ),
                ),
                element_count=1,
            ),
            action=replay_store.CachedAction(kind="wait", duration=0.1),
            destination=replay_store.ScreenSignature(
                description="Checkout screen", phash="after"
            ),
            arrival_ms=0,
        )
        calls = []

        def matches_screen(*args):
            calls.append(args)
            return True

        attempt = self._take(
            cached_transition,
            SimpleNamespace(matches_screen=matches_screen),
            same_build=False,
        )

        self.assertIsNotNone(attempt.step)
        self.assertEqual(len(calls), 1)

    def test_a_terminal_screen_keeps_its_semantic_destination(self):
        replay = replay_learning.DurableReplay(
            trace=None,
            values={},
            names=replay_store.ValueNames(),
        )
        replay.record_step(
            TapAction(
                coordinates=Coordinates(x=500, y=650),
                target="Continue button",
                reason="continue",
            ),
            screen_read("before"),
            target_description="Continue button",
            screen_description="Welcome screen",
        )
        replay.describe_arrival("Account screen")
        replay.record_arrival(screen_read("after"))
        steps, screens = replay.select_learned_steps()

        trace = replay_walk.build_trace(
            replay_store.TraceIdentity(
                scenario_uuid="checkout",
                scenario_digest="digest",
                platform="android",
                target_profile="Pixel",
                detector="",
                grounding="coordinates",
            ),
            steps,
            screens,
            replay.names,
            target_descriptions=replay.target_descriptions,
            screen_descriptions=replay.screen_descriptions,
        )

        self.assertIsNotNone(trace)
        self.assertEqual(
            trace.transitions[0].destination.description, "Account screen"
        )
