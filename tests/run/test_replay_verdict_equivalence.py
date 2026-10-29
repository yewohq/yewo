"""A persisted path must not replace evidence used to decide a verdict."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fakes import walk
from yewo.agent.actuation.actions import Coordinates
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.device.launch import SetupMode
from yewo.agent.device.platform import Platform
from yewo.agent.memory.replay_scenarios import TraceStore
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.plan.schema import DoneAction
from yewo.agent.run.agent import DeviceAgent
from yewo.agent.run.record import StepSource
from yewo.testing.evidence import RuleVerification
from yewo.testing.judge.qa_auditor import ScreenAssessment
from yewo.testing.models import GeneratedTestCase
from yewo.testing.qa_agent import QAAgent
from yewo.testing.qa_planner import QAStepResponse


def _element(text, role="button"):
    return GUIElement(
        element=text,
        bounding_box=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=200),
        confidence=0.9,
        type=role,
        text=text,
    )


SIGN_IN = _element("Sign in")
ORDERS = _element("Orders")
CONFIRMED = _element("Order confirmed", "text")


class Screens:
    platform = Platform.ANDROID
    serial = ""
    os_version = None
    package_id = None
    driver = None
    screen_height = 2000

    def __init__(self):
        self.screens = [[SIGN_IN], [ORDERS], [CONFIRMED]]
        self.position = 0

    def reset(self):
        pass

    def tap(self, _x, _y, with_feedback=True):
        self.position += 1
        return True

    def get_screenshot(self):
        return b"unchanged"

    def showing_system_prompt(self):
        return False

    def is_keyboard_open(self):
        return False

    def is_still_in_sut(self):
        return True

    @property
    def current_hash(self):
        return f"screen-{self.position}"

    def perceive(self):
        image = Image.new("RGB", (4, 4))
        return list(self.screens[self.position]), image, image

    @property
    def last_scan_info(self):
        return ScanInfo(
            mode="detect",
            reason="step",
            phash=self.current_hash,
            screen_width=1000,
            screen_height=2000,
            element_count=1,
        )


class Planner:
    grounding = "coordinates"
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def plan_step(self, gui_elements, **kwargs):
        type(self).calls += 1
        target = gui_elements[0]
        if target.text == "Order confirmed":
            action = DoneAction(reason="The instructions ran.")
            complete = True
        else:
            box = target.bounding_box
            action = TapCoordinateAction(
                coordinates=Coordinates(
                    x=(box.x_min + box.x_max) // 2,
                    y=(box.y_min + box.y_max) // 2,
                ),
                target=target.text,
                reason="Follow the scenario.",
            )
            complete = False
        return QAStepResponse(
            current_screen=target.text,
            observation=target.text,
            reasoning="Follow the scenario.",
            action=action,
            goal_satisfied=complete,
            goal_evidence=target.text if complete else "",
        )


class Auditor:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def audit(self, _image, screen_id=None, gui_elements=()):
        type(self).calls += 1
        return ScreenAssessment(
            screen_name=screen_id or "screen",
            rule_verifications=[
                RuleVerification(
                    rule="Order confirmed",
                    rule_id="expected:0",
                    status="passed",
                    reasoning="The outcome is visible.",
                )
            ],
        )


class VerdictEquivalenceTests(unittest.TestCase):
    def setUp(self):
        Planner.calls = 0
        Auditor.calls = 0
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = TraceStore(
            root=self.root,
            values={},
            platform="android",
            target_profile="Pixel_8",
            detector="d@abc",
            grounding="coordinates",
            key=b"test replay digest key",
            artifact="test-build",
        )
        self.case = GeneratedTestCase(
            scenario_uuid="checkout",
            test_id="TS_1",
            scenario_name="Checkout",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Tap Sign in.", "Tap Orders."),
            expected_outcomes=["Order confirmed"],
            setup=SetupMode.PRESERVE,
        )

    def _agent(self, max_steps=8):
        qa = QAAgent.__new__(QAAgent)
        qa._fast_llm = object()
        qa._grounding = "coordinates"
        qa._trace_store = self.store
        qa._device_agent = DeviceAgent(
            planner=Planner(), max_steps=max_steps, settle_time=0.0, grounder=None
        )
        return qa

    def _run(self, directory, max_steps=8):
        app = Screens()
        return self._agent(max_steps).execute_scenario(
            self.case, [], app, app, self.root / directory
        )

    def _add_legacy_proof(self):
        path = next((self.root / ".yewo" / "paths").rglob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["proof"] = {
            "verifications": [{"rule_id": "expected:0", "status": "passed", "applicable": True}]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @patch("yewo.testing.obligation_auditing.QAAuditor", Auditor)
    @patch("yewo.testing.qa_agent.QAPlanner", Planner)
    @patch("yewo.agent.perception.settle.settle", return_value=None)
    @patch("yewo.agent.memory.replay_walk.time.sleep")
    def test_a_persisted_run_keeps_the_verdict_inputs(self, _sleep, _settle):
        planned = self._run("planned")
        planned_audits = Auditor.calls
        self._add_legacy_proof()
        cached = self._run("cached")

        self.assertEqual(cached.status, planned.status)
        self.assertEqual(cached.cause, planned.cause)
        self.assertEqual(
            [(item.rule_id, item.status, item.applicable) for item in cached.rule_verifications],
            [(item.rule_id, item.status, item.applicable) for item in planned.rule_verifications],
        )
        self.assertEqual(cached.verdict_inputs, planned.verdict_inputs)
        self.assertTrue(any(step.source is StepSource.CACHE for step in cached.steps))
        self.assertEqual(Planner.calls, 3, "the cached run still called the planner")
        self.assertEqual(planned_audits, 4)
        self.assertEqual(
            Auditor.calls,
            8,
            "each run needs universal screen audits and one final obligation audit",
        )

    @patch("yewo.testing.obligation_auditing.QAAuditor", Auditor)
    @patch("yewo.testing.qa_agent.QAPlanner", Planner)
    @patch("yewo.agent.perception.settle.settle", return_value=None)
    @patch("yewo.agent.memory.replay_walk.time.sleep")
    def test_a_complete_path_can_end_on_the_step_limit(self, _sleep, _settle):
        self._run("planned")
        planned_audits = Auditor.calls

        cached = self._run("cached", max_steps=2)

        self.assertEqual(cached.status, "PASSED")
        self.assertEqual([step.source for step in cached.steps], [StepSource.CACHE] * 2)
        self.assertEqual(Planner.calls, 3)
        self.assertEqual(planned_audits, 4)
        self.assertEqual(Auditor.calls, 8)


if __name__ == "__main__":
    unittest.main()
