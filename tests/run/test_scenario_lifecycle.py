"""Deterministic state before and after a visible user journey."""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fakes import walk

from yewo.agent.device.launch import SetupMode, clear_app_data
from yewo.agent.device.platform import Platform
from yewo.agent.device.receipts import Receipt
from yewo.agent.device.restoration import RestorationEvent
from yewo.errors import DeviceUnavailable
from yewo.testing.lifecycle import (
    LifecyclePhase,
    LifecycleStatus,
    ScenarioPreconditions,
    SetupOrigin,
    apply_capability_precondition,
    resolve_setup,
    restore_app_precondition,
    restoration_operations,
)
from yewo.testing.models import GeneratedTestCase, ScenarioDraft
from yewo.testing.suite import load_workflows


class InitialStateContractTests(unittest.TestCase):
    def test_the_generator_cannot_choose_initial_app_state(self):
        self.assertNotIn("setup", ScenarioDraft.model_json_schema()["properties"])

    def test_an_independent_scenario_defaults_to_clean_state(self):
        mode, origin = resolve_setup(ScenarioPreconditions())

        self.assertIs(mode, SetupMode.CLEAR_DATA)
        self.assertIs(origin, SetupOrigin.DEFAULT)

    def test_an_explicit_state_is_kept_as_declared(self):
        mode, origin = resolve_setup(ScenarioPreconditions(app_state=SetupMode.RELAUNCH))

        self.assertIs(mode, SetupMode.RELAUNCH)
        self.assertIs(origin, SetupOrigin.DECLARED)

    def test_a_journey_can_declare_only_typed_preconditions(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        path = directory / "yewo.yaml"
        path.write_text(
            """
journey: Let the user check out.
preconditions:
  app_state: relaunch
  capabilities:
    - kind: permission
      permission: camera
      grant: false
""",
            encoding="utf-8",
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertIs(journey.preconditions.app_state, SetupMode.RELAUNCH)
        self.assertEqual(journey.preconditions.capabilities[0].kind, "permission")

    def test_workflow_preconditions_are_inherited_and_can_be_overridden(self):
        raw = {
            "workflows": {
                "shop": {
                    "preconditions": {"app_state": "relaunch"},
                    "journeys": [
                        {"journey": "Browse the shop."},
                        {
                            "journey": "Start fresh.",
                            "preconditions": {"app_state": "clear_data"},
                        },
                    ],
                }
            }
        }

        journeys = load_workflows(Path("yewo.yaml"), raw=raw)[0].journeys

        self.assertIs(journeys[0].preconditions.app_state, SetupMode.RELAUNCH)
        self.assertIs(journeys[1].preconditions.app_state, SetupMode.CLEAR_DATA)


class AppStateApplicationTests(unittest.TestCase):
    def test_ios_clear_state_reinstalls_and_confirms_the_app(self):
        driver = SimpleNamespace(
            remove_app=Mock(),
            install_app=Mock(),
            is_app_installed=Mock(return_value=True),
        )
        device = SimpleNamespace(
            platform=Platform.IOS,
            app_id="com.example.App",
            package_id=None,
            app_path="/tmp/Example.app",
            driver=driver,
        )

        self.assertTrue(clear_app_data(device))
        driver.remove_app.assert_called_once_with("com.example.App")
        driver.install_app.assert_called_once_with("/tmp/Example.app")
        driver.is_app_installed.assert_called_once_with("com.example.App")

    def test_ios_clear_state_refuses_an_unconfirmed_reinstall(self):
        driver = SimpleNamespace(
            remove_app=Mock(),
            install_app=Mock(),
            is_app_installed=Mock(return_value=False),
        )
        device = SimpleNamespace(
            platform=Platform.IOS,
            app_id="com.example.App",
            package_id=None,
            app_path="/tmp/Example.app",
            driver=driver,
        )

        self.assertFalse(clear_app_data(device))

    def test_a_failed_clear_does_not_continue_as_a_clean_scenario(self):
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.memory.replay_store import ValueNames
        from yewo.agent.run.agent import DeviceAgent
        from yewo.utils.secrets import SecretScrubber

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.replay_ledger = None
        agent._default_fast_planner = Mock()
        agent.scrubber = SecretScrubber()
        agent.step_opened = None
        agent.max_steps = 1
        agent.declared_values = {}
        agent.value_verifications = {}
        agent.value_names = ValueNames()
        with (
            patch("yewo.agent.run.task.clear_app_data", return_value=False),
            patch("yewo.agent.run.task.restart_app") as restart,
        ):
            result = DeviceAgent.run(
                agent,
                goal="Open the app.",
                device_control=SimpleNamespace(screen_height=2400),
                perception=Mock(),
                session_dir=Path("unused"),
                setup=SetupMode.CLEAR_DATA,
            )

        self.assertEqual(result.status, "exception")
        self.assertIsInstance(result.history, AgentHistory)
        self.assertIn("clean app state", result.failure_reason)
        restart.assert_not_called()

    def test_a_clean_scenario_restores_the_clean_boundary(self):
        device = SimpleNamespace(platform=Platform.ANDROID)
        with patch("yewo.testing.lifecycle.clear_app_data", return_value=True) as clear:
            operation = restore_app_precondition(device, SetupMode.CLEAR_DATA)

        clear.assert_called_once_with(device)
        self.assertEqual(operation.status, LifecycleStatus.RESTORED)
        self.assertEqual(operation.phase, LifecyclePhase.RESTORATION)

    def test_a_failed_clean_boundary_is_failed_restoration(self):
        device = SimpleNamespace(platform=Platform.IOS)
        with patch("yewo.testing.lifecycle.clear_app_data", return_value=False):
            operation = restore_app_precondition(device, SetupMode.CLEAR_DATA)

        self.assertEqual(operation.status, LifecycleStatus.FAILED)


class CapabilityPreconditionTests(unittest.TestCase):
    def test_a_precondition_requires_a_confirmed_readback(self):
        from yewo.agent.actuation.actions import PermissionCapability

        controller = Mock()
        controller.execute.return_value = None

        with self.assertRaises(DeviceUnavailable):
            apply_capability_precondition(
                controller,
                PermissionCapability(permission="camera", grant=False),
            )

    def test_a_confirmed_precondition_records_no_private_value(self):
        from yewo.agent.actuation.actions import ClipboardCapability

        secret = "private-token"
        controller = Mock()
        controller.execute.return_value = Receipt(
            capability="clipboard",
            requested=secret,
            observed=secret,
            target="device",
            source="mobile: getClipboard",
            private=True,
        )

        operation, _ = apply_capability_precondition(controller, ClipboardCapability(text=secret))

        self.assertEqual(operation.status, LifecycleStatus.CONFIRMED)
        self.assertNotIn(secret, operation.model_dump_json())


class RestorationEvidenceTests(unittest.TestCase):
    def test_a_failed_restore_is_distinct_lifecycle_evidence(self):
        operations = restoration_operations(
            [RestorationEvent("airplane mode", False, "device offline")]
        )

        self.assertEqual(operations[0].phase, LifecyclePhase.RESTORATION)
        self.assertEqual(operations[0].status, LifecycleStatus.FAILED)
        self.assertIn("device offline", operations[0].failure_reason)

    def test_an_unchanged_device_still_records_restoration_completion(self):
        operation = restoration_operations([])[0]

        self.assertEqual(operation.status, LifecycleStatus.RESTORED)
        self.assertEqual(operation.observed, "no changed state")


class ScenarioLifecycleIntegrationTests(unittest.TestCase):
    def _agent(self):
        from yewo.agent.memory.history import AgentHistory
        from yewo.agent.run.agent import DeviceRunResult
        from yewo.testing.models import ScenarioReport
        from yewo.testing.qa_agent import QAAgent
        from yewo.utils.secrets import SecretScrubber

        agent = QAAgent.__new__(QAAgent)
        agent._fast_llm = object()
        agent._grounding = "marks"
        agent._trace_store = None
        agent._build_replay = lambda _case: None
        agent._save_trace = Mock()
        agent._build_report = Mock(
            return_value=ScenarioReport(
                scenario_uuid="scenario",
                test_id="TS_001",
                scenario_name="Checkout",
                cell="functional.happy_path.0",
                status="PASSED",
            )
        )
        agent._device_agent = SimpleNamespace(
            max_steps=40,
            replay_ledger=None,
            durable_replay=None,
            scrubber=SecretScrubber(),
            run=Mock(
                return_value=DeviceRunResult(
                    status="goal_reached",
                    steps_executed=1,
                    failure_reason=None,
                    history=AgentHistory(),
                )
            ),
        )
        return agent

    def _case(self):
        return GeneratedTestCase(
            scenario_uuid="scenario",
            test_id="TS_001",
            scenario_name="Checkout",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Open the app."),
            expected_outcomes=["Checkout is visible"],
            setup="preserve",
        )

    def _device(self):
        return SimpleNamespace(
            platform=Platform.ANDROID,
            serial="",
            os_version=None,
            package_id=None,
            app_id=None,
            driver=None,
        )

    def test_setup_failure_stops_before_the_ui_agent(self):
        agent = self._agent()
        with patch(
            "yewo.testing.qa_agent.apply_app_precondition",
            side_effect=DeviceUnavailable("clear failed"),
        ):
            report = agent.execute_scenario(
                self._case(),
                [],
                self._device(),
                SimpleNamespace(reset=Mock()),
                Path("unused"),
            )

        self.assertEqual(report.status, "ERROR")
        self.assertIn("clear failed", report.failure_reason)
        agent._device_agent.run.assert_not_called()

    def test_failed_restoration_overrides_a_passing_ui_result(self):
        agent = self._agent()

        @contextmanager
        def failed_restoration(*_args, events, **_kwargs):
            yield None
            events.append(RestorationEvent("airplane mode", False, "adb offline"))

        with patch("yewo.testing.qa_agent.restoration.preserved", failed_restoration):
            report = agent.execute_scenario(
                self._case(),
                [],
                self._device(),
                SimpleNamespace(reset=Mock()),
                Path("unused"),
            )

        self.assertEqual(report.status, "ERROR")
        self.assertEqual(report.cause, "device")
        self.assertIn("restoration failed", report.failure_reason.lower())


if __name__ == "__main__":
    unittest.main()
