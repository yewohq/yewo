"""What the command line answers, and what it refuses to start."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
