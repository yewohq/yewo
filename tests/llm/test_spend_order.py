"""The order a run commits its money in."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generation_fakes import scenario_generation_evidence
from yewo.testing.generation_startup import ScenarioStartupKnowledge


class SpendOrderTests(unittest.TestCase):
    """Scenario generation is the most expensive call a run makes.

    It is also useless if the device never comes up, so the device has to be
    proven first. A boot failure after generation costs the suite to discover.
    """

    def _run(self):
        from yewo.testing.models import UserJourney
        from yewo.testing.runner import YewoTestRunner

        order = []

        class Session:
            def __init__(self, *_, **__):
                self.telemetry_dir = Path(tempfile.mkdtemp())
                self.screenshots_dir = self.telemetry_dir
                self.session_id = "s1"
                self.installed_by_run = False
                self.root_log_file = self.telemetry_dir / "run.log"
                self.appium_log_path = self.telemetry_dir / "appium.log"
                self.manifest_path = self.telemetry_dir / "run.json"
                self.prompts_dir = self.telemetry_dir / "prompts"

            def start(self, **_):
                order.append("device")
                return SimpleNamespace()

            def checkpoint_recording(self):
                pass

            def close(self, **_):
                pass

        class Generator:
            def __init__(self, *_, **__):
                pass

            def generate_suite(self, _journey, _cells=None):
                order.append("generator")
                raise RuntimeError("stop here")

        class DiscoveryProvider:
            def discover(self, _journey):
                order.append("discovery")
                return ScenarioStartupKnowledge(scenario_generation_evidence())

        with (
            patch("yewo.testing.runner.DeviceSession", Session),
            patch("yewo.testing.generation_startup.ScenarioGenerator", Generator),
            patch("yewo.testing.runner.QAAgent"),
            patch("yewo.testing.runner.ensure_device_ready") as ready,
            patch("yewo.testing.runner.for_grounding"),
            patch("yewo.testing.runner.evaluate_package_policy", return_value=[]),
        ):
            ready.side_effect = lambda *_: order.append("asked adb")
            YewoTestRunner(
                device_name="Pixel_8",
                app_path="app.apk",
                discovery_provider=DiscoveryProvider(),
            ).run_journey(
                UserJourney(id="j", name="J", description="A journey")
            )
        return order

    def test_the_device_is_open_before_the_generator_is_asked(self):
        self.assertEqual(
            self._run(),
            ["asked adb", "device", "discovery", "generator"],
        )
