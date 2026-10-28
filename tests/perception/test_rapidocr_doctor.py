"""Doctor reports RapidOCR readiness without running model inference."""

import unittest
from unittest import mock

from yewo.agent.perception.rapidocr_diagnostics import RapidOCRDiagnostic
from yewo.doctor.checks import Status, check_rapidocr_runtime


class RapidOCRDoctorTests(unittest.TestCase):
    @mock.patch("yewo.doctor.checks.missing_vision_packages", return_value=[])
    @mock.patch("yewo.doctor.checks.diagnose_rapidocr")
    def test_a_healthy_runtime_reports_versions_and_provider(self, diagnose, _missing):
        diagnose.return_value = RapidOCRDiagnostic(
            "ok",
            "ready",
            versions=(("onnxruntime", "1.31.1"), ("rapidocr", "3.10.0")),
            providers=("CPUExecutionProvider",),
        )

        result = check_rapidocr_runtime()

        self.assertEqual(result.status, Status.OK)
        self.assertIn("rapidocr 3.10.0", result.detail)
        diagnose.assert_called_once_with()

    @mock.patch("yewo.doctor.checks.missing_vision_packages", return_value=[])
    @mock.patch("yewo.doctor.checks.diagnose_rapidocr")
    def test_a_broken_runtime_is_blocking_and_actionable(self, diagnose, _missing):
        diagnose.return_value = RapidOCRDiagnostic(
            "models_missing",
            "model files are missing",
            "reinstall vision",
        )

        result = check_rapidocr_runtime()

        self.assertEqual(result.status, Status.FAIL)
        self.assertEqual(result.hint, "reinstall vision")

    @mock.patch(
        "yewo.doctor.checks.missing_vision_packages",
        return_value=["rapidocr", "onnxruntime"],
    )
    @mock.patch("yewo.doctor.checks.diagnose_rapidocr")
    def test_missing_packages_skip_the_runtime_process(self, diagnose, _missing):
        result = check_rapidocr_runtime()

        self.assertEqual(result.status, Status.SKIP)
        diagnose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
