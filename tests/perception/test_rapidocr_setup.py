"""Setup reports detector and RapidOCR readiness independently."""

import unittest
from unittest import mock

from yewo.agent.perception.rapidocr_diagnostics import RapidOCRDiagnostic
from yewo.cli.console import FAIL, INFO, OK
from yewo.setup.steps import SetupContext, prefetch_detector, verify_rapidocr


class RapidOCRSetupTests(unittest.TestCase):
    @mock.patch("yewo.setup.steps.vision_available", return_value=True)
    @mock.patch("yewo.setup.steps.diagnose_rapidocr")
    def test_a_successful_smoke_test_reports_its_identity(self, diagnose, _vision):
        diagnose.return_value = RapidOCRDiagnostic(
            "ok",
            "smoke passed",
            identity="rapidocr@test",
        )

        result = verify_rapidocr(SetupContext(assume_yes=True))

        self.assertEqual(result.level, OK)
        self.assertIn("rapidocr@test", result.detail)
        diagnose.assert_called_once_with(smoke_test=True)

    @mock.patch("yewo.setup.steps.vision_available", return_value=True)
    @mock.patch("yewo.setup.steps.diagnose_rapidocr")
    def test_a_broken_runtime_fails_setup_with_its_hint(self, diagnose, _vision):
        diagnose.return_value = RapidOCRDiagnostic(
            "provider_missing",
            "CPU provider is unavailable",
            "install a CPU build",
        )

        result = verify_rapidocr(SetupContext(assume_yes=True))

        self.assertEqual(result.level, FAIL)
        self.assertEqual(result.hint, "install a CPU build")

    @mock.patch("yewo.setup.steps.vision_available", return_value=True)
    @mock.patch("yewo.setup.steps.diagnose_rapidocr")
    def test_a_dry_run_does_not_initialize_ocr(self, diagnose, _vision):
        result = verify_rapidocr(SetupContext(dry_run=True))

        self.assertEqual(result.level, INFO)
        diagnose.assert_not_called()


class DetectorSetupTests(unittest.TestCase):
    @mock.patch("yewo.setup.steps.vision_available", return_value=True)
    @mock.patch("yewo.setup.steps._run_visible", return_value=1)
    def test_detector_failure_is_not_reported_as_an_ocr_failure(self, _run, _vision):
        result = prefetch_detector(SetupContext(assume_yes=True), "hf://owner/model@" + "a" * 40)

        self.assertEqual(result.level, FAIL)
        self.assertEqual(result.label, "detector model")


if __name__ == "__main__":
    unittest.main()
