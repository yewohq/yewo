"""What a run is refused for, before it opens anything."""

import unittest
from unittest import mock


class GroundingRequirementTests(unittest.TestCase):
    def test_a_mode_that_reads_detections_needs_the_detector(self):
        from yewo.preflight import packages_for

        with mock.patch("yewo.preflight.missing_vision_packages", return_value=["ultralytics"]):
            self.assertIn("vision", packages_for("marks")[0])

    def test_a_mode_that_names_its_target_needs_no_detector(self):
        from yewo.preflight import packages_for

        with (
            mock.patch("yewo.preflight.missing_vision_packages", return_value=["ultralytics"]),
            mock.patch("yewo.preflight.missing_jev_packages", return_value=[]),
        ):
            self.assertEqual(packages_for("describe"), [])

    def test_a_mode_that_names_its_target_needs_the_decision_sdk(self):
        from yewo.preflight import packages_for

        with mock.patch("yewo.preflight.missing_jev_packages", return_value=["typesafe_sdk"]):
            self.assertIn("typesafe_sdk", packages_for("describe")[0])

    def test_describe_mode_also_wants_the_decision_model_key(self):
        from yewo.preflight import models_for

        self.assertEqual(models_for("describe"), ("typesafe/jev-1.14.0",))
        self.assertEqual(models_for("marks"), ())

    def test_an_installed_detector_is_not_reported(self):
        from yewo.preflight import packages_for

        with mock.patch("yewo.preflight.missing_vision_packages", return_value=[]):
            self.assertEqual(packages_for("marks"), [])


class RunRefusalTests(unittest.TestCase):
    """Two missing things cost one message, not two device boots."""

    def test_both_a_key_and_an_extra_are_named_at_once(self):
        from yewo.errors import UsageError
        from yewo.preflight import require_run

        with (
            mock.patch("yewo.preflight.missing_credentials", side_effect=lambda m: ["A_KEY"] if m else []),
            mock.patch("yewo.preflight.missing_vision_packages", return_value=["ultralytics"]),
            self.assertRaises(UsageError) as refused,
        ):
            require_run("some/model", None, "marks")

        message = str(refused.exception)
        self.assertIn("A_KEY", message)
        self.assertIn("ultralytics", message)

    def test_a_run_with_everything_it_needs_starts(self):
        from yewo.preflight import require_run

        with (
            mock.patch("yewo.preflight.missing_credentials", return_value=[]),
            mock.patch("yewo.preflight.missing_vision_packages", return_value=[]),
        ):
            self.assertIsNone(require_run("some/model", "some/fast", "marks"))

    def test_each_key_is_named_once_however_many_models_want_it(self):
        from yewo.errors import UsageError
        from yewo.preflight import require_run

        with (
            mock.patch("yewo.preflight.missing_credentials", return_value=["A_KEY"]),
            mock.patch("yewo.preflight.missing_vision_packages", return_value=[]),
            self.assertRaises(UsageError) as refused,
        ):
            require_run("one/model", "other/model", "describe")

        self.assertEqual(str(refused.exception).count("A_KEY"), 1)


if __name__ == "__main__":
    unittest.main()
