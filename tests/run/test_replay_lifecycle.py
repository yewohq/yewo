"""A later run walks the path an earlier run saved."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory import replay_scenarios
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.plan.schema import DeviceStepResponse
from yewo.agent.run.agent import DeviceAgent, SetupMode
from yewo.agent.run.record import StepSource

WIDTH, HEIGHT = 1000, 2000
AN_IMAGE = Image.new("RGB", (4, 4))


def element(text, role="button", box=(100, 100, 300, 200)):
    left, top, right, bottom = box
    return GUIElement(
        element=text,
        bounding_box=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
        confidence=0.9,
        type=role,
        text=text,
    )


def centre_of(target):
    box = target.bounding_box
    return Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)


SIGN_IN = element("Sign in")
ORDERS = element("Orders")
CONFIRMED = element("Order confirmed", "text")


class Screens:
    """A fake app that moves through a list of screens."""

    def __init__(self, screens):
        self.screens = list(screens)
        self.at = 0
        self.taps = []

    @property
    def elements(self):
        return self.screens[min(self.at, len(self.screens) - 1)]

    def tap(self, x, y, with_feedback=True):
        self.taps.append((x, y))
        self.at += 1
        return True

    def get_screenshot(self):
        return b"a screenshot, which target_moved reads as unchanged"

    driver = None
    platform = "android"

    def showing_system_prompt(self):
        return False

    def is_keyboard_open(self):
        return False

    def is_still_in_sut(self):
        return True

    @property
    def screen_height(self):
        return HEIGHT

    @property
    def current_hash(self):
        return f"phash-{min(self.at, len(self.screens) - 1)}"

    def perceive(self):
        return list(self.elements), AN_IMAGE, AN_IMAGE

    @property
    def last_scan_info(self):
        return ScanInfo(
            mode="detect",
            reason="step",
            phash=self.current_hash,
            screen_width=WIDTH,
            screen_height=HEIGHT,
            element_count=len(self.elements),
        )


class CountingPlanner:
    """A planner that taps whatever it is told to, and counts being asked."""

    grounding = "coordinates"

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0

    def plan_step(self, **kwargs):
        self.calls += 1
        target = self.plan[min(self.calls - 1, len(self.plan) - 1)]
        return DeviceStepResponse(
            current_screen="a screen",
            observation="what is on it",
            reasoning="why this tap",
            action=TapCoordinateAction(
                coordinates=centre_of(target),
                target=target.text,
                reason="planned",
            ),
        )


def scenario():
    return SimpleNamespace(
        scenario_uuid="checkout", setup="CLEAR_DATA", steps=[], expected_outcomes=[]
    )


class ReplayRunCase(unittest.TestCase):
    """A run of the real loop against an app of a few screens."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.session = self.root / "session"

        for target in (
            "yewo.agent.run.task.restart_app",
            "yewo.agent.run.task.clear_app_data",
        ):
            patched = patch(target)
            patched.start()
            self.addCleanup(patched.stop)
        settled = patch("yewo.agent.perception.settle.settle", side_effect=lambda *a, **k: None)
        settled.start()
        self.addCleanup(settled.stop)
        slept = patch("yewo.agent.memory.replay_walk.time.sleep")
        slept.start()
        self.addCleanup(slept.stop)

    def _store(self, approve_update=False, allow_creation=True):
        return replay_scenarios.TraceStore(
            root=self.root,
            values={},
            platform="android",
            target_profile="Pixel_8",
            detector="d@abc",
            grounding="coordinates",
            key=b"test replay digest key",
            artifact="test-build",
            allow_path_creation=allow_creation,
            approve_path_update=lambda _path: approve_update,
        )

    def _agent(self, planner, max_steps=2):
        return DeviceAgent(planner=planner, max_steps=max_steps, settle_time=0.0, grounder=None)

    def _run(self, planner, app, replay, max_steps=2, observation_hook=None):
        agent = self._agent(planner, max_steps)
        agent.durable_replay = replay
        return agent.run(
            goal="buy something",
            device_control=app,
            perception=app,
            session_dir=self.session,
            observation_hook=observation_hook,
            setup=SetupMode.PRESERVE,
        )

    def _learn(self, app, plan):
        """One whole run, then the trace it leaves on disk."""
        store = self._store()
        identity = store.resolve_identity(scenario())
        replay = store.build_replay(identity)
        planner = CountingPlanner(plan)

        self._run(planner, app, replay)
        store.save_trace(scenario(), replay, "PASSED")
        return planner


class LifecycleTests(ReplayRunCase):
    """A learned path, read back by an agent that never saw the first run."""

    def test_a_run_learns_a_path_and_a_later_run_walks_it(self):
        learning = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        first = self._learn(learning, [SIGN_IN, ORDERS])

        self.assertEqual(first.calls, 2, "the planner drove the first run")
        self.assertTrue(list((self.root / ".yewo").rglob("*.json")), "nothing was written")

        walking = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        store = self._store()
        replay = store.build_replay(store.resolve_identity(scenario()))
        second = CountingPlanner([SIGN_IN])

        self.assertTrue(replay.has_path, "the second run found no path")
        result = self._run(second, walking, replay)

        cached = [step for step in result.steps if step.source is StepSource.CACHE]
        self.assertEqual(len(cached), 2, "the saved steps were not replayed")
        self.assertEqual(second.calls, 0, "a cached step still asked a planner")
        self.assertEqual(walking.taps[:2], learning.taps[:2], "a different route")

    def test_the_replayed_steps_cost_no_planner_call(self):
        """Which is the whole point. A cached step that still plans has
        saved a tap and paid for the part that costs money."""
        self._learn(Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), [SIGN_IN, ORDERS])

        store = self._store()
        replay = store.build_replay(store.resolve_identity(scenario()))
        second = CountingPlanner([SIGN_IN])
        self._run(second, Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), replay)

        self.assertEqual(second.calls, 0)

    def test_a_scenario_of_one_action_is_learned(self):
        """The destination of the last step is read by nothing else, so a
        run that does not take one more reading learns nothing at all."""
        app = Screens([[SIGN_IN], [CONFIRMED]])
        store = self._store()
        replay = store.build_replay(store.resolve_identity(scenario()))
        agent = self._agent(CountingPlanner([SIGN_IN]))
        agent.durable_replay = replay

        agent.run(
            goal="sign in",
            device_control=app,
            perception=app,
            session_dir=self.session,
            max_steps=1,
            setup=SetupMode.PRESERVE,
        )
        store.save_trace(scenario(), replay, "PASSED")

        written = list((self.root / ".yewo").rglob("*.json"))
        self.assertEqual(len(written), 1, "a one-action scenario saved nothing")

    def test_a_read_only_run_never_creates_a_path(self):
        store = self._store(allow_creation=False)
        replay = store.build_replay(store.resolve_identity(scenario()))
        self._run(
            CountingPlanner([SIGN_IN, ORDERS]),
            Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]),
            replay,
        )

        store.save_trace(scenario(), replay, "PASSED")

        self.assertEqual(list((self.root / ".yewo").rglob("*.json")), [])

    def test_a_changed_path_is_not_replaced_without_approval(self):
        self._learn(Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), [SIGN_IN, ORDERS])
        store = self._store()
        first = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, store.resolve_identity(scenario())),
            store.resolve_identity(scenario()),
        )

        # The app changed: the second screen no longer holds what the
        # saved step expects, so the walk refuses there and the planner
        # takes over on the screen the walk stopped on.
        moved = element("Your orders", box=(100, 600, 300, 700))
        app = Screens([[SIGN_IN], [moved], [CONFIRMED]])
        replay = store.build_replay(store.resolve_identity(scenario()))
        self._run(CountingPlanner([moved]), app, replay)
        store.save_trace(scenario(), replay, "PASSED")

        second = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, store.resolve_identity(scenario())),
            store.resolve_identity(scenario()),
        )
        self.assertEqual(
            second.transitions,
            first.transitions,
            "an unapproved run replaced the reviewed path",
        )

    def test_an_approved_update_keeps_the_cached_head_and_planned_tail(self):
        self._learn(Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), [SIGN_IN, ORDERS])
        store = self._store(approve_update=True)
        identity = store.resolve_identity(scenario())
        first = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, identity), identity
        )

        moved = element("Your orders", box=(100, 600, 300, 700))
        app = Screens([[SIGN_IN], [moved], [CONFIRMED]])
        replay = store.build_replay(identity)
        self._run(CountingPlanner([moved]), app, replay)
        store.save_trace(scenario(), replay, "PASSED")

        updated = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, identity), identity
        )
        self.assertEqual(
            updated.transitions[0].source.anchors,
            first.transitions[0].source.anchors,
            "the approved path no longer starts where the scenario does",
        )
        self.assertNotEqual(
            updated.transitions[1].source.anchors,
            first.transitions[1].source.anchors,
            "the approved planned tail was not written",
        )


class AuditEvidenceTests(ReplayRunCase):
    """Learning and replay expose the same live observations to QA."""

    @staticmethod
    def _texts(screen):
        return [found.text for found in screen.gui_elements]

    def test_a_replayed_run_shows_the_auditor_the_same_screens(self):
        store = self._store()
        learned = []

        def watch(screen, number):
            learned.append((self._texts(screen), number))

        learning = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        replay = store.build_replay(store.resolve_identity(scenario()))
        self._run(
            CountingPlanner([SIGN_IN, ORDERS]),
            learning,
            replay,
            observation_hook=watch,
        )
        store.save_trace(scenario(), replay, "PASSED")

        replayed = []
        walking = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        second = store.build_replay(store.resolve_identity(scenario()))
        self._run(
            CountingPlanner([SIGN_IN]),
            walking,
            second,
            observation_hook=lambda screen, number: replayed.append((self._texts(screen), number)),
        )

        self.assertEqual(replayed, learned)

    def test_a_replayed_run_also_reads_the_screen_it_ended_on(self):
        store = self._store()
        learning = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        replay = store.build_replay(store.resolve_identity(scenario()))
        self._run(CountingPlanner([SIGN_IN, ORDERS]), learning, replay)
        store.save_trace(scenario(), replay, "PASSED")

        replayed = []
        walking = Screens([[SIGN_IN], [ORDERS], [CONFIRMED]])
        second = store.build_replay(store.resolve_identity(scenario()))
        self._run(
            CountingPlanner([SIGN_IN]),
            walking,
            second,
            observation_hook=lambda screen, number: replayed.append((self._texts(screen), number)),
        )

        self.assertEqual(
            replayed,
            [(["Sign in"], 1), (["Orders"], 2), (["Order confirmed"], 3)],
        )


class FailedActionTests(ReplayRunCase):
    def test_a_scenario_that_recovered_does_not_learn_what_failed(self):
        """The next run would send it again and wait for it to fail
        again. What ran before it is still a path."""

        class Refusing(Screens):
            def tap(self, x, y, with_feedback=True):
                self.taps.append((x, y))
                if len(self.taps) == 2:
                    raise RuntimeError("the driver said no")
                self.at += 1
                return True

        app = Refusing([[SIGN_IN], [ORDERS], [CONFIRMED]])
        store = self._store()
        replay = store.build_replay(store.resolve_identity(scenario()))
        agent = DeviceAgent(
            planner=CountingPlanner([SIGN_IN, ORDERS, ORDERS]),
            max_steps=3,
            settle_time=0.0,
            grounder=None,
        )
        agent.durable_replay = replay
        agent.run(
            goal="buy something",
            device_control=app,
            perception=app,
            session_dir=self.session,
            setup=SetupMode.PRESERVE,
        )
        store.save_trace(scenario(), replay, "PASSED")

        trace = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, store.resolve_identity(scenario())),
            store.resolve_identity(scenario()),
        )
        self.assertEqual(len(trace.transitions), 2)
        self.assertFalse(trace.transitions[0].starts_segment)
        self.assertTrue(trace.transitions[1].starts_segment)

    def test_a_failed_cached_action_starts_a_new_learning_segment(self):
        """A cached dispatch can mutate before it raises, so the planned
        recovery is retained behind a segment boundary."""
        self._learn(Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), [SIGN_IN, ORDERS])

        class RefusingCachedStep(Screens):
            def tap(self, x, y, with_feedback=True):
                self.taps.append((x, y))
                if len(self.taps) == 2:
                    raise RuntimeError("the cached dispatch failed")
                self.at += 1
                return True

        store = self._store()
        replay = store.build_replay(store.resolve_identity(scenario()))
        app = RefusingCachedStep([[SIGN_IN], [ORDERS], [CONFIRMED]])
        self._run(CountingPlanner([ORDERS]), app, replay, max_steps=3)

        # Read off the run rather than the file. What reaches disk is also
        # decided by the no-shrink rule, which would hide this.
        steps, _ = replay.select_learned_steps()

        self.assertEqual(len(steps), 2)
        self.assertEqual(replay.selected_segment_starts, [False, True])

    def test_a_run_that_learned_less_does_not_shorten_the_saved_path(self):
        """A run whose learning stopped early saw a prefix, not a path.
        Writing it would trade a route that worked for part of one."""
        self._learn(Screens([[SIGN_IN], [ORDERS], [CONFIRMED]]), [SIGN_IN, ORDERS])
        store = self._store()
        before = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, store.resolve_identity(scenario())),
            store.resolve_identity(scenario()),
        )

        short = store.build_replay(store.resolve_identity(scenario()))
        short.record_step(
            TapAction(coordinates=centre_of(SIGN_IN), reason="r"),
            SimpleNamespace(
                gui_elements=[SIGN_IN],
                scan_info=Screens([[SIGN_IN]]).last_scan_info,
                current_phash="p",
                raw_img_bytes=b"",
            ),
        )
        short.record_arrival(
            SimpleNamespace(
                gui_elements=[ORDERS],
                scan_info=Screens([[ORDERS]]).last_scan_info,
                current_phash="p",
                raw_img_bytes=b"",
            )
        )
        store.save_trace(scenario(), short, "PASSED")

        after = replay_scenarios.replay_files.read_trace(
            replay_scenarios.replay_files.trace_path(self.root, store.resolve_identity(scenario())),
            store.resolve_identity(scenario()),
        )
        self.assertEqual(len(after.transitions), len(before.transitions), "the path was shortened")
