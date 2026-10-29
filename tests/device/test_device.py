"""Driving a device, and reporting when it cannot be driven."""

import io
import unittest

from fakes import FakeClock, walk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch


from yewo.agent.device.control import DeviceControl
from yewo.agent.device.platform import Platform
from yewo.errors import ActionRejected, ObservationUnavailable


class ActionSafetyTests(unittest.TestCase):
    def _control(self):
        return SimpleNamespace(
            screen_width=100,
            screen_height=200,
            driver=object(),
            platform=__import__(
                "yewo.agent.device.platform", fromlist=["Platform"]
            ).Platform.ANDROID,
            tap=Mock(),
        )

    def _tap(self, x, y):
        from yewo.agent.actuation.actions import Coordinates, TapAction

        return TapAction(coordinates=Coordinates(x=x, y=y), reason="tap")

    def test_out_of_bounds_action_is_rejected_before_dispatch(self):
        from yewo.agent.actuation.orchestrator import ActionOrchestrator

        control = self._control()
        with self.assertRaisesRegex(ValueError, "outside screen width"):
            ActionOrchestrator(control).execute(self._tap(100, 50))

        control.tap.assert_not_called()

    def test_device_failure_propagates(self):
        from yewo.agent.actuation.orchestrator import ActionOrchestrator

        control = self._control()
        control.tap.side_effect = RuntimeError("gesture failed")

        with self.assertRaisesRegex(RuntimeError, "gesture failed"):
            ActionOrchestrator(control).execute(self._tap(50, 50))

class DeviceFailureTests(unittest.TestCase):
    """A failed device instruction has to say what it was, and keep its cause."""

    def _failing_adb(self):
        from yewo.utils.subprocess_runner import SubprocessRunError

        return SubprocessRunError("adb failed", 1, "", "device offline")

    def test_a_failed_command_names_the_intent_and_keeps_the_cause(self):
        from yewo.agent.device.adb_control import AdbControl
        from yewo.errors import DeviceUnavailable
        from yewo.utils.subprocess_runner import SubprocessRunError

        cause = self._failing_adb()
        with (
            patch("yewo.agent.device.adb_control.run_command", side_effect=cause),
            self.assertRaises(DeviceUnavailable) as raised,
        ):
            AdbControl.set_airplane_mode("emulator-5554", True)

        self.assertIn("set airplane mode", str(raised.exception))
        self.assertIn("airplane-mode enable", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, SubprocessRunError)
        self.assertEqual(raised.exception.__cause__.stderr, "device offline")

    def test_a_nested_call_reports_the_instruction_that_failed(self):
        """Not the outermost wrapper: the step that could not be carried out."""
        from yewo.agent.device.adb_control import AdbControl
        from yewo.errors import DeviceUnavailable

        with (
            patch(
                "yewo.agent.device.adb_control.run_command",
                side_effect=self._failing_adb(),
            ),
            self.assertRaises(DeviceUnavailable) as raised,
        ):
            AdbControl.enable_feedback("emulator-5554", True, True)

        self.assertIn("toggle pointer location", str(raised.exception))

    def test_visual_feedback_reads_both_android_settings(self):
        from yewo.agent.device.adb_control import AdbControl, VisualFeedbackSettings

        with patch.object(AdbControl, "ask", side_effect=["1", "0"]) as ask:
            settings = AdbControl.read_visual_feedback("emulator-5554")

        self.assertEqual(settings, VisualFeedbackSettings(True, False))
        self.assertEqual(
            ask.call_args_list,
            [
                call(
                    "emulator-5554", "settings", "get", "system", "pointer_location"
                ),
                call("emulator-5554", "settings", "get", "system", "show_touches"),
            ],
        )

    def test_an_unreadable_screen_is_reported_as_such(self):
        from yewo.agent.device.control import DeviceControl
        from yewo.errors import ObservationUnavailable

        control = DeviceControl.__new__(DeviceControl)
        control.device_name = "emulator-5554"
        control.driver = SimpleNamespace(
            get_screenshot_as_png=Mock(side_effect=OSError("connection reset"))
        )

        with self.assertRaises(ObservationUnavailable) as raised:
            control.get_screenshot()

        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_a_report_that_cannot_be_written_says_so(self):
        from yewo.errors import ArtifactWriteFailed
        from yewo.testing.models import JourneyReport
        from yewo.testing.report.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j", journey_name="Journey", description="d", status="PASSED"
        )

        with self.assertRaises(ArtifactWriteFailed):
            save_markdown_report(report, Path("/nonexistent-directory-for-yewo"))


class LifecycleOwnershipTests(unittest.TestCase):
    @patch("yewo.agent.device.session.urllib.request.urlopen")
    def test_existing_appium_service_is_not_claimed(self, urlopen):
        from yewo.agent.device.session import DeviceSession

        response = MagicMock()
        response.__enter__.return_value.status = 200
        urlopen.return_value = response
        session = DeviceSession.__new__(DeviceSession)
        session._appium_url = "http://127.0.0.1:4723"
        session._owns_appium = False
        session._appium_service = MagicMock()

        session._start_appium()

        session._appium_service.start.assert_not_called()
        self.assertFalse(session._owns_appium)

class LaunchPaintTests(unittest.TestCase):
    """A debug build can draw nothing for a second or two after am start."""

    def _device(self, png):
        return SimpleNamespace(get_screenshot=Mock(return_value=png))

    def _await_first_paint(self, device):
        from yewo.agent.device.launch import await_first_paint

        clock = FakeClock()
        with (
            patch("yewo.agent.device.launch.time.monotonic", side_effect=clock.monotonic),
            patch("yewo.agent.device.launch.time.sleep", side_effect=clock.sleep),
        ):
            await_first_paint(device)
        return clock.elapsed

    def _png(self, pixels):

        from PIL import Image

        image = Image.new("L", (10, 100))
        image.putdata(pixels)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_an_undrawn_screen_is_blank(self):
        from yewo.agent.device.launch import screen_is_blank

        self.assertTrue(screen_is_blank(self._device(self._png([255] * 1000))))

    def test_a_drawn_screen_is_not_blank(self):
        from yewo.agent.device.launch import screen_is_blank

        pixels = [0 if index % 2 else 255 for index in range(1000)]
        self.assertFalse(screen_is_blank(self._device(self._png(pixels))))

    def test_an_unreadable_screen_does_not_block_the_launch(self):
        from yewo.agent.device.launch import screen_is_blank

        device = SimpleNamespace(
            get_screenshot=Mock(side_effect=ObservationUnavailable("no screen"))
        )

        self.assertIsNone(screen_is_blank(device))
        self._await_first_paint(device)

    def _drawn(self, seed=0):
        return self._png(
            [255 if (index + seed) % 3 else 0 for index in range(1000)]
        )

    def test_waiting_stops_once_a_drawn_screen_repeats(self):
        blank = self._png([255] * 1000)
        device = self._device(blank)
        device.get_screenshot = Mock(
            side_effect=[blank, blank, self._drawn(), self._drawn()]
        )

        self._await_first_paint(device)

        self.assertEqual(device.get_screenshot.call_count, 4)

    def test_a_screen_that_never_settles_is_not_waited_out(self):
        device = self._device(self._drawn())
        device.get_screenshot = Mock(
            side_effect=[self._drawn(seed=index) for index in range(60)]
        )

        elapsed = self._await_first_paint(device)

        from yewo.defaults import LAUNCH_CONTENT_GRACE, LAUNCH_PAINT_TIMEOUT

        self.assertLess(elapsed, LAUNCH_PAINT_TIMEOUT)
        self.assertGreaterEqual(elapsed, LAUNCH_CONTENT_GRACE)

    def test_a_splash_is_not_mistaken_for_the_first_screen(self):
        splash, real = self._drawn(seed=0), self._drawn(seed=1)
        device = self._device(splash)
        device.get_screenshot = Mock(side_effect=[splash, real, real])

        self._await_first_paint(device)

        self.assertEqual(device.get_screenshot.call_count, 3)


class CapabilityTests(unittest.TestCase):
    """An outcome Yewo has no way to evidence, refused before the device opens.

    The check reads outcomes rather than steps. A step is typed now, so one
    cannot ask for network capture: no action names it and no capability
    applies it. An outcome is still prose, and one asking for evidence Yewo
    never collects is the case left to catch.
    """

    def _case(self, outcome):
        from yewo.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Security",
            cell="security.negative.0",
            category="security",
            scenario_type="negative",
            priority="P0",
            steps=walk("Launch the app."),
            expected_outcomes=[outcome],
        )

    def test_an_outcome_needing_network_capture_is_rejected_before_execution(self):
        from yewo.testing.judge.capabilities import unsupported_reason

        self.assertIn(
            "network traffic capture",
            unsupported_reason(self._case("Network inspection shows no plaintext.")),
        )

    def test_a_visible_outcome_remains_supported(self):
        from yewo.testing.judge.capabilities import unsupported_reason

        self.assertIsNone(unsupported_reason(self._case("The Login button is visible.")))


class CapabilityMatchingTests(unittest.TestCase):
    """Seeing a contrasting element is observable. Measuring a ratio is not."""

    def _reason(self, *outcomes):
        from yewo.testing.judge.capabilities import unsupported_reason

        return unsupported_reason(self._case(*outcomes))

    def _case(self, *outcomes):
        from yewo.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Contrast",
            cell="accessibility.happy_path.0",
            category="accessibility",
            scenario_type="happy_path",
            priority="P2",
            steps=walk("Launch the app."),
            expected_outcomes=list(outcomes),
        )

    def test_a_visible_high_contrast_element_is_supported(self):
        self.assertIsNone(self._reason("The high-contrast error banner is visible"))

    def test_a_measured_contrast_ratio_is_unsupported(self):
        self.assertIn("contrast", self._reason("The contrast ratio is at least 4.5:1"))

    def test_a_wcag_assertion_is_unsupported(self):
        self.assertIn("contrast", self._reason("Text meets WCAG AA"))


class ScenarioIsolationTests(unittest.TestCase):
    def test_supported_scenario_resets_perception_before_running(self):
        from yewo.agent.run.agent import DeviceRunResult
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.plan.planner import GROUNDING_COORDINATES
        from yewo.testing.models import GeneratedTestCase
        from yewo.testing.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent._fast_llm = object()
        agent._grounding = GROUNDING_COORDINATES
        agent._device_agent = SimpleNamespace(
            max_steps=40,
            replay_ledger=None,
            scrubber=__import__(
                "yewo.utils.secrets", fromlist=["SecretScrubber"]
            ).SecretScrubber(),
            run=Mock(
                return_value=DeviceRunResult(
                    status="step_limit",
                    steps_executed=1,
                    failure_reason="limit",
                    history=AgentHistory(),
                )
            ),
        )
        perception = SimpleNamespace(reset=Mock())
        case = GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Visible button",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Tap the Login button."),
            expected_outcomes=["the Login button is visible"],
            setup="preserve",
        )

        # A serial-less control is enough: the capability preflight reads the
        # platform, and reads nothing off a device it cannot address.
        device_control = SimpleNamespace(
            platform=Platform.ANDROID,
            serial="",
            os_version=None,
            package_id=None,
            driver=None,
        )

        report = agent.execute_scenario(
            case, [], device_control, perception, Path("unused"), session_trail=object()
        )

        perception.reset.assert_called_once_with()
        self.assertEqual(report.status, "UNVERIFIED")


class DeviceTypingTests(unittest.TestCase):
    def _control(self, rect):
        element = SimpleNamespace(id="field", rect=rect, send_keys=Mock())
        driver = SimpleNamespace(
            switch_to=SimpleNamespace(active_element=element),
        )
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.driver = driver
        control.tap = Mock()
        control.screen_width = 1080
        control.screen_height = 2400
        return control, element

    @patch("yewo.agent.device.control.time.sleep")
    def test_type_reuses_already_focused_field(self, _sleep):
        control, element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        control.type("test@email.com", 540, 893)

        control.tap.assert_not_called()
        element.send_keys.assert_called_once_with("test@email.com")

    @patch("yewo.agent.device.control.time.sleep")
    def test_type_is_refused_when_nothing_reports_focus(self, _sleep):
        """Text sent with no focus goes wherever focus happens to be."""
        control, element = self._control({"x": 63, "y": 820, "width": 954, "height": 147})
        control.driver.switch_to.active_element = None

        with self.assertRaisesRegex(ActionRejected, "nothing reported focus"):
            control.type("test@email.com", 537, 1260)

        element.send_keys.assert_not_called()

    @patch("yewo.agent.device.control.time.sleep")
    def test_type_is_refused_when_the_focused_element_has_no_bounds(self, _sleep):
        """An element that cannot report a rect cannot be checked against a point."""
        control, element = self._control({"x": 63, "y": 820, "width": 954, "height": 147})
        unreadable = Mock(spec=["id", "rect", "send_keys"])
        unreadable.id = "field"
        type(unreadable).rect = PropertyMock(side_effect=RuntimeError("stale element"))
        control.driver.switch_to.active_element = unreadable

        with self.assertRaisesRegex(ActionRejected, "nothing reported focus"):
            control.type("test@email.com", 537, 1260)

        unreadable.send_keys.assert_not_called()

    @patch("yewo.agent.device.control.time.sleep")
    def test_a_full_screen_surface_still_accepts_typing(self, _sleep):
        """Some apps focus one element for the whole screen.

        Refusing that would mean never typing in them, so the tap on the point
        stands in as the weaker evidence.
        """
        control, element = self._control({"x": 0, "y": 0, "width": 1080, "height": 2400})

        control.type("test@email.com", 537, 1260)

        element.send_keys.assert_called_once_with("test@email.com")

    @patch("yewo.agent.device.control.time.sleep")
    def test_clear_is_refused_when_nothing_reports_focus(self, _sleep):
        control, _element = self._control({"x": 63, "y": 820, "width": 954, "height": 147})
        control.driver.switch_to.active_element = None

        with self.assertRaisesRegex(ActionRejected, "nothing reported focus"):
            control.clear(537, 1260)

    @patch("yewo.agent.device.control.time.sleep")
    def test_type_rejects_a_tap_that_left_focus_where_it_was(self, _sleep):
        """Focus that never moved means the tap hit something that takes no text.

        A picture of a field reads like the field. The planner has to be told
        to choose a different target, not to re-read the coordinates.
        """
        control, element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        with self.assertRaisesRegex(ActionRejected, "holds no text"):
            control.type("test@email.com", 537, 1260)

        control.tap.assert_called_once_with(537, 1260)
        element.send_keys.assert_not_called()

    @patch("yewo.agent.device.control.time.sleep")
    def test_type_rejects_a_tap_that_focused_a_different_field(self, _sleep):
        """Any text field opens the keyboard, so a wrong field looks like a reflow."""
        before = SimpleNamespace(id="nothing", rect={"x": 0, "y": 0, "width": 0, "height": 0})
        email = SimpleNamespace(
            id="email",
            rect={"x": 63, "y": 820, "width": 954, "height": 147},
            send_keys=Mock(),
        )
        switch_to = SimpleNamespace(active_element=before)
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.driver = SimpleNamespace(switch_to=switch_to)
        control.screen_width, control.screen_height = 1080, 2400

        def tap(_x, _y):
            switch_to.active_element = email

        control.tap = Mock(side_effect=tap)

        with self.assertRaisesRegex(ActionRejected, "screen moved"):
            control.type("password123", 537, 1444)

        email.send_keys.assert_not_called()

    @patch("yewo.agent.device.control.time.sleep")
    def test_clear_propagates_the_focus_failure(self, _sleep):
        control, _element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        with self.assertRaisesRegex(ActionRejected, "holds no text"):
            control.clear(537, 1260)

        control.tap.assert_called_once_with(537, 1260)


class WarmSessionTests(unittest.TestCase):
    def _control(self):
        control = DeviceControl.__new__(DeviceControl)
        control.device_name = "Pixel_8"
        return control

    def _options(self):
        captured = {}
        return SimpleNamespace(set_capability=lambda k, v: captured.__setitem__(k, v)), captured

    @patch("yewo.agent.device.control.AdbControl")
    def test_skips_setup_when_the_helper_apps_are_present(self, adb):
        adb.resolve_emulator_serial.return_value = "emulator-5554"
        adb.installed_packages.return_value = {
            "io.appium.uiautomator2.server",
            "io.appium.uiautomator2.server.test",
            "io.appium.settings",
        }
        options, captured = self._options()

        self._control()._skip_redundant_setup(options)

        self.assertTrue(captured["appium:skipServerInstallation"])
        self.assertTrue(captured["appium:skipDeviceInitialization"])

    @patch("yewo.agent.device.control.AdbControl")
    def test_does_a_full_setup_when_the_server_is_missing(self, adb):
        adb.resolve_emulator_serial.return_value = "emulator-5554"
        adb.installed_packages.return_value = {"io.appium.settings"}
        options, captured = self._options()

        self._control()._skip_redundant_setup(options)

        self.assertEqual(captured, {})


class IdleSessionTests(unittest.TestCase):
    """Appium ends a session that sends no command while a step is thinking."""

    def test_both_platforms_ask_for_longer_than_appium_allows_by_default(self):
        from yewo.defaults import IDLE_SESSION_TIMEOUT_SECONDS

        self.assertGreater(IDLE_SESSION_TIMEOUT_SECONDS, 60)

    def test_the_capability_reaches_the_driver(self):
        from appium.options.android import UiAutomator2Options

        from yewo.defaults import IDLE_SESSION_TIMEOUT_SECONDS

        options = UiAutomator2Options()
        options.new_command_timeout = IDLE_SESSION_TIMEOUT_SECONDS

        self.assertEqual(
            options.to_capabilities()["appium:newCommandTimeout"],
            IDLE_SESSION_TIMEOUT_SECONDS,
        )
