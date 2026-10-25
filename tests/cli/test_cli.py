"""What the command line answers, and what it refuses to start."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _workflow(name: str = "", **settings):
    """One unnamed workflow holding one journey, as the loader would return it."""
    from yewo.testing.suite import Workflow

    return Workflow(name, settings, [SimpleNamespace(id="j", name="J")])


class ExitCodeTests(unittest.TestCase):
    """Only the entry point ends the process. Everything below answers with a code."""

    def test_a_usage_problem_is_reported_once_and_exits_two(self):
        from yewo.cli.main import main
        from yewo.errors import UsageError

        with (
            patch("yewo.cli.main.load_config", side_effect=UsageError("no app binary")),
            patch("sys.stderr", new=io.StringIO()) as captured,
        ):
            code = main(["doctor"])

        self.assertEqual(code, 2)
        self.assertIn("no app binary", captured.getvalue())

    def test_an_unloadable_suite_answers_two_without_exiting(self):
        from yewo.testing.cli import YewoCLI

        cli = YewoCLI()

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            code = cli.run([str(missing)])

        self.assertEqual(code, 2)

    def test_no_journey_file_is_a_usage_error_not_an_exit(self):
        from yewo.app_discovery import resolve_journey_file
        from yewo.errors import UsageError

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.app_discovery.Path.cwd", return_value=Path(directory)),
            self.assertRaises(UsageError),
        ):
            resolve_journey_file(None)

    def test_a_setting_yewo_config_does_not_know_is_not_looked_up_there(self):
        """grounding comes from a flag or the journey, or falls back to the default."""
        from yewo.agent.plan.planner import GROUNDING_MARKS
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        cli = YewoCLI()
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )

        with (
            patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            cli.run(["j.yaml"])

        self.assertEqual(runner_class.call_args.kwargs["grounding"], GROUNDING_MARKS)

    def test_the_suite_result_decides_the_code(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        def report(status):
            return JourneyReport(
                journey_id="j", journey_name="J", description="d", status=status
            )

        for status, expected in (("PASSED", 0), ("FAILED", 1)):
            cli = YewoCLI()
            with (
                patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
                patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
                patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
                patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
            ):
                runner_class.return_value.run_journey.return_value = report(status)
                self.assertEqual(cli.run(["j.yaml"]), expected)

    def test_an_interruption_exits_130_without_escaping_the_cli(self):
        from yewo.testing.cli import YewoCLI

        with (
            patch("yewo.testing.cli.load_workflows", return_value=[_workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.side_effect = KeyboardInterrupt
            code = YewoCLI().run(["j.yaml"])

        self.assertEqual(code, 130)

    def test_update_paths_is_explicit(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch("yewo.testing.cli.load_workflows", return_value=[_workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml", "--update-paths"])

        self.assertTrue(runner_class.call_args.kwargs["update_paths"])

    def test_secret_exposure_is_off_unless_explicitly_allowed(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch("yewo.testing.cli.load_workflows", return_value=[_workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml", "--allow-secret-exposure"])

        self.assertTrue(runner_class.call_args.kwargs["allow_secret_exposure"])

    def test_secret_exposure_can_be_allowed_by_workflow_yaml(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow(allow_secret_exposure=True)],
            ),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml"])

        self.assertTrue(runner_class.call_args.kwargs["allow_secret_exposure"])

    def test_command_line_secret_exposure_wins_over_yaml(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow(allow_secret_exposure=False)],
            ),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml", "--allow-secret-exposure"])

        self.assertTrue(runner_class.call_args.kwargs["allow_secret_exposure"])

    def test_command_line_can_restore_safe_behavior_over_yaml(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow(allow_secret_exposure=True)],
            ),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml", "--no-allow-secret-exposure"])

        self.assertFalse(runner_class.call_args.kwargs["allow_secret_exposure"])


class ConfigFileTests(unittest.TestCase):
    """A setting that does nothing must not look like one that was applied."""

    def _write(self, text):
        path = Path(tempfile.mkdtemp()) / "yewo.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_misspelled_setting_is_named_and_the_rest_still_load(self):
        from yewo.config import read_file

        path = self._write("platfrom: ios\nmodel: a/model\n")

        with self.assertLogs("yewo.config", level="WARNING") as captured:
            settings = read_file(path)

        self.assertEqual(settings, {"model": "a/model"})
        self.assertIn("platfrom", captured.output[0])

    def test_a_file_of_known_settings_warns_about_nothing(self):
        from yewo.config import read_file

        path = self._write("platform: ios\nmax_steps: 7\n")

        with self.assertNoLogs("yewo.config", level="WARNING"):
            settings = read_file(path)

        self.assertEqual(settings, {"platform": "ios", "max_steps": 7})

    def test_project_yaml_can_allow_secret_exposure(self):
        from yewo.config import load_config

        path = self._write("allow_secret_exposure: true\n")

        config = load_config(path.parent)

        self.assertTrue(config.get("allow_secret_exposure"))


class ReuseAppFlagTests(unittest.TestCase):
    """Whether a run reinstalls the app, or tests the build already on the device."""

    def _runner_kwargs(self, argv):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport

        cli = YewoCLI()
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch(
                "yewo.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            cli.run(argv)
        return runner_class.call_args.kwargs

    def test_the_app_is_reinstalled_by_default(self):
        """A stale build would be tested silently, so this is opt-in."""
        self.assertFalse(self._runner_kwargs(["j.yaml"])["reuse_app"])

    def test_the_flag_reaches_the_runner(self):
        self.assertTrue(self._runner_kwargs(["j.yaml", "--reuse-app"])["reuse_app"])


class PlatformNameTests(unittest.TestCase):
    """A platform Yewo does not know stops the run."""

    def test_a_known_name_resolves(self):
        from yewo.agent.device.platform import Platform

        self.assertIs(Platform.named("android"), Platform.ANDROID)
        self.assertIs(Platform.named("iOS"), Platform.IOS)

    def test_a_misspelled_platform_is_refused(self):
        """A typo fails at the name, not deep in a run on the wrong platform."""
        from yewo.agent.device.platform import Platform
        from yewo.errors import UsageError

        with self.assertRaises(UsageError) as caught:
            Platform.named("androdi")

        self.assertIn("androdi", str(caught.exception))


class TargetMatrixCommandTests(unittest.TestCase):
    """One command coordinates isolated target runs without suppressing later ones."""

    def _workflow(self):
        from yewo.testing.suite import Target, Workflow

        journey = SimpleNamespace(id="j", name="Journey")
        return Workflow(
            "checkout",
            journeys=[journey],
            targets=(
                Target("pixel", {"platform": "android", "device": "Pixel_8"}),
                Target("iphone", {"platform": "ios", "device": "iPhone 16"}),
            ),
        )

    @staticmethod
    def _report(status="PASSED"):
        from yewo.testing.models import JourneyReport

        return JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status=status,
        )

    def test_a_matrix_refuses_to_start_without_a_lock(self):
        from yewo.testing.cli import YewoCLI

        with (
            patch("yewo.testing.cli.load_workflows", return_value=[self._workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.cli.runner_for") as runner_for,
        ):
            code = YewoCLI().run(["j.yaml"])

        self.assertEqual(code, 2)
        runner_for.assert_not_called()

    def test_a_failed_target_does_not_suppress_the_next_target(self):
        from yewo.testing.cli import YewoCLI

        broken = SimpleNamespace(
            run_journey=Mock(side_effect=RuntimeError("driver failed")),
            last_report_paths={},
        )
        passed = SimpleNamespace(
            run_journey=Mock(return_value=self._report()),
            last_report_paths={},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.testing.cli.load_workflows", return_value=[self._workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch.object(YewoCLI, "_require_matrix_locks"),
            patch(
                "yewo.testing.cli.output_dir_for", return_value=Path(directory)
            ),
            patch("yewo.testing.cli.runner_for", side_effect=[broken, passed]) as runner_for,
            patch("yewo.testing.cli.write_execution_summary"),
            patch("yewo.testing.cli.new_uuid", side_effect=["execution", "run-1", "run-2"]),
        ):
            code = YewoCLI().run(["j.yaml", "--locked"])

        self.assertEqual(code, 1)
        self.assertEqual(runner_for.call_count, 2)
        passed.run_journey.assert_called_once()
        run_dirs = [call.kwargs["assets_dir"] for call in runner_for.call_args_list]
        self.assertIn("pixel", run_dirs[0].parts)
        self.assertIn("iphone", run_dirs[1].parts)

    def test_target_selection_runs_only_the_named_target(self):
        from yewo.testing.cli import YewoCLI

        passed = SimpleNamespace(
            run_journey=Mock(return_value=self._report()),
            last_report_paths={},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.testing.cli.load_workflows", return_value=[self._workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch.object(YewoCLI, "_require_matrix_locks"),
            patch(
                "yewo.testing.cli.output_dir_for", return_value=Path(directory)
            ),
            patch("yewo.testing.cli.runner_for", return_value=passed) as runner_for,
            patch("yewo.testing.cli.write_execution_summary"),
            patch("yewo.testing.cli.new_uuid", side_effect=["execution", "run-1"]),
        ):
            code = YewoCLI().run(
                ["j.yaml", "--locked", "--target", "iphone"]
            )

        self.assertEqual(code, 0)
        self.assertEqual(runner_for.call_count, 1)
        self.assertEqual(runner_for.call_args.kwargs["target"].id, "iphone")

    def test_a_target_with_failed_restoration_is_not_reused(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.lifecycle import LifecycleOperation
        from yewo.testing.models import ScenarioReport
        from yewo.testing.suite import Target, Workflow

        journeys = [
            SimpleNamespace(id="first", name="First"),
            SimpleNamespace(id="second", name="Second"),
        ]
        workflow = Workflow(
            "checkout",
            journeys=journeys,
            targets=(Target("pixel", {"platform": "android", "device": "Pixel_8"}),),
        )
        failed_report = self._report("FAILED").model_copy(
            update={
                "journey_id": "first",
                "journey_name": "First",
                "scenarios": [
                    ScenarioReport(
                        test_id="TS_1",
                        scenario_name="First",
                        cell="functional.happy_path.0",
                        status="FAILED",
                        lifecycle_operations=[
                            LifecycleOperation(
                                operation_id="restore-1",
                                phase="restoration",
                                kind="device_state",
                                requested="portrait",
                                status="failed",
                                source="orientation",
                            )
                        ],
                    )
                ],
            }
        )
        failed = SimpleNamespace(
            run_journey=Mock(return_value=failed_report),
            last_report_paths={},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.testing.cli.load_workflows", return_value=[workflow]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch.object(YewoCLI, "_require_matrix_locks"),
            patch("yewo.testing.cli.output_dir_for", return_value=Path(directory)),
            patch("yewo.testing.cli.runner_for", return_value=failed) as runner_for,
            patch("yewo.testing.cli.write_execution_summary") as write_summary,
            patch(
                "yewo.testing.cli.new_uuid",
                side_effect=["execution", "run-1", "run-2"],
            ),
        ):
            code = YewoCLI().run(["j.yaml", "--locked"])

        self.assertEqual(code, 1)
        self.assertEqual(runner_for.call_count, 1)
        self.assertEqual(write_summary.call_count, 2)
        second_results = write_summary.call_args.args[3]
        self.assertEqual(second_results[-1].status, "ERROR")
        self.assertIn("failed restoration", second_results[-1].error)

    def test_an_interruption_checkpoints_the_active_run_and_stops(self):
        from yewo.testing.cli import YewoCLI

        interrupted = SimpleNamespace(
            run_journey=Mock(side_effect=KeyboardInterrupt),
            last_report_paths={},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.testing.cli.load_workflows", return_value=[self._workflow()]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch.object(YewoCLI, "_require_matrix_locks"),
            patch(
                "yewo.testing.cli.output_dir_for", return_value=Path(directory)
            ),
            patch("yewo.testing.cli.runner_for", return_value=interrupted) as runner_for,
            patch("yewo.testing.cli.write_execution_summary") as write_summary,
            patch("yewo.testing.cli.new_uuid", side_effect=["execution", "run-1"]),
        ):
            code = YewoCLI().run(["j.yaml", "--locked"])

        self.assertEqual(code, 130)
        self.assertEqual(runner_for.call_count, 1)
        write_summary.assert_called_once()

    def test_target_settings_and_run_identity_reach_the_runner(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.suite import Target, Workflow
        from yewo.testing.wiring import runner_for

        cli = YewoCLI()
        args = cli.parser.parse_args(["j.yaml", "--locked"])
        workflow = Workflow(
            "checkout",
            settings={
                "platform": "android",
                "device": "fallback",
                "app_path": "fallback.apk",
            },
        )
        declared = Target(
            "iphone",
            {
                "platform": "ios",
                "device": "iPhone 16",
                "app_path": "Runner.app",
            },
        )
        with (
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("Runner.app")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_for(
                workflow,
                args,
                Path("j.yaml"),
                cli.config,
                target=declared,
                assets_dir=Path("isolated"),
                run_id="run-1",
                execution_id="execution-1",
            )

        kwargs = runner_class.call_args.kwargs
        self.assertEqual(kwargs["platform"].value, "ios")
        self.assertEqual(kwargs["device_name"], "iPhone 16")
        self.assertEqual(kwargs["app_path"], "Runner.app")
        self.assertEqual(kwargs["assets_dir"], Path("isolated"))
        self.assertEqual(kwargs["target_id"], "iphone")
        self.assertEqual(kwargs["run_id"], "run-1")
        self.assertEqual(kwargs["execution_id"], "execution-1")
