import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class PerceptionModelCliTests(unittest.TestCase):
    def test_a_pinned_perception_model_reaches_the_runner(self):
        from yewo.testing.cli import YewoCLI
        from yewo.testing.models import JourneyReport
        from yewo.testing.suite import Workflow

        reference = f"hf://example/yewo-ui@{'a' * 40}"
        workflow = Workflow(
            "",
            {},
            [SimpleNamespace(id="j", name="J")],
        )
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch("yewo.testing.cli.load_workflows", return_value=[workflow]),
            patch("yewo.testing.cli.resolve_journey_file", return_value=Path("j.yaml")),
            patch("yewo.testing.wiring.resolve_app_path", return_value=Path("app.apk")),
            patch("yewo.testing.wiring.YewoTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            YewoCLI().run(["j.yaml", "--perception-model", reference])

        self.assertEqual(
            runner_class.call_args.kwargs["perception_model"], reference
        )


if __name__ == "__main__":
    unittest.main()
