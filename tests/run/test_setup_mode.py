"""The state a scenario needs, established rather than assumed."""

import contextlib
import unittest

from fakes import walk
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SetupModeTests(unittest.TestCase):
    """Required state is part of what a test is.

    Stated in a prompt instead, it is an assumption, and a prompt enforces
    nothing.
    """

    def test_the_default_is_the_one_that_makes_scenarios_independent(self):
        from yewo.agent.device.launch import SetupMode
        from yewo.testing.models import GeneratedTestCase

        case = GeneratedTestCase(
            test_id="t",
            scenario_name="s",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Open the app."),
            expected_outcomes=["the app opens"],
        )

        self.assertIs(case.setup, SetupMode.CLEAR_DATA)

    def test_what_the_planner_is_told_never_names_a_screen(self):
        """A named screen is a claim about one app, and a wrong one is obeyed."""
        from yewo.agent.device.launch import SetupMode

        for mode in SetupMode:
            described = mode.describe().lower()
            self.assertNotIn("login screen", described, mode)
            self.assertNotIn("home screen", described, mode)

    def test_each_mode_says_something_different(self):
        from yewo.agent.device.launch import SetupMode

        said = {mode.describe() for mode in SetupMode}

        self.assertEqual(len(said), len(list(SetupMode)))


class SetupExecutionTests(unittest.TestCase):
    """Each mode has to do what it says, or the contract is another assumption."""

    def _run(self, mode):
        from yewo.agent.run.agent import DeviceAgent

        from yewo.utils.secrets import SecretScrubber

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.replay_ledger = None
        agent._default_fast_planner = Mock()
        agent.scrubber = SecretScrubber()
        agent.step_opened = None
        calls = []

        # The run is cut short once setup is done. Only the setup matters.
        with (
            patch(
                "yewo.agent.run.agent.clear_app_data",
                lambda _d: calls.append("clear") or True,
            ),
            patch("yewo.agent.run.agent.restart_app", lambda _d: calls.append("restart")),
            patch("yewo.agent.run.agent.replay_known_path", side_effect=RuntimeError("stop")),
            contextlib.suppress(Exception),
        ):
                DeviceAgent.run(
                    agent,
                    goal="g",
                    device_control=SimpleNamespace(screen_height=2400),
                    perception=Mock(),
                    session_dir=Mock(),
                    setup=mode,
                )
        return calls

    def test_clear_data_wipes_then_restarts(self):
        from yewo.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.CLEAR_DATA), ["clear", "restart"])

    def test_relaunch_restarts_without_wiping(self):
        from yewo.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.RELAUNCH), ["restart"])

    def test_preserve_touches_nothing(self):
        from yewo.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.PRESERVE), [])
