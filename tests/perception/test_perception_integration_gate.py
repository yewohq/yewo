import unittest
from pathlib import Path

from tools.perception_eval.corpus import PerceptionCase, PerceptionCorpus, TargetPoint
from tools.perception_eval.integration_gate import (
    build_baseline,
    evaluate_run,
    incremental_failures,
)
from tools.perception_eval.metrics import (
    Box,
    CaseObservation,
    EvaluationRun,
    ObservedElement,
)


class PerceptionIntegrationGateTests(unittest.TestCase):
    def setUp(self):
        self.corpus = PerceptionCorpus(
            schema_version=1,
            cases=[
                PerceptionCase(
                    id="form",
                    screenshot=Path("form.png"),
                    platform="android",
                    traits={"form"},
                    keyboard_visible=False,
                    expected_text=["Save", "Name"],
                    target_points=[TargetPoint(name="save", x=5, y=5)],
                    digest="fixture",
                )
            ],
            missing_platforms=set(),
            missing_traits=set(),
        )

    def test_reports_the_exact_evidence_that_regressed(self):
        baseline_run = self._run([self._element("Save"), self._element("Name")])
        baseline = build_baseline(self.corpus, baseline_run)
        candidate = self._run([self._element("Name", x=20)])

        report = evaluate_run(self.corpus, baseline, candidate, validate_replay=False)

        self.assertFalse(report.passed)
        self.assertEqual(report.regressions[0].case_id, "form")
        self.assertIn("lost expected text: 'Save'", report.regressions[0].failures)
        self.assertIn("lost target coverage: save", report.regressions[0].failures)

    def test_enforces_projection_and_bounds_contracts(self):
        run = self._run([self._element("Save")])
        baseline = build_baseline(self.corpus, run)
        run.observations[0].out_of_bounds_count = 1
        run.observations[0].normal_keyboard_target_count = 1

        report = evaluate_run(self.corpus, baseline, run, validate_replay=False)

        failures = report.regressions[0].failures
        self.assertIn("one or more detections are outside the screenshot", failures)
        self.assertIn("keyboard-owned detections leaked into normal app targets", failures)

    def test_incremental_contract_is_relative_not_a_fixed_latency(self):
        result = {
            "detector_reused": True,
            "surface_roles": ["keyboard_region"],
            "unique_app_text_equal": True,
            "incremental_app_class_counts": {"Button": 1},
            "full_app_class_counts": {"Button": 1},
            "speedup": 1.2,
        }

        self.assertEqual(incremental_failures(result), [])
        result["speedup"] = 0.9
        self.assertIn("not faster", incremental_failures(result)[0])

    def _run(self, elements):
        return EvaluationRun(
            detector="detector@commit",
            ocr="rapidocr@models",
            observations=[
                CaseObservation(
                    case_id="form",
                    fixture_digest="fixture",
                    visual_keyboard_visible=False,
                    mark_count=1,
                    latency_ms=1,
                    peak_rss_mib=1,
                    elements=elements,
                )
            ],
        )

    @staticmethod
    def _element(text, x=0):
        return ObservedElement(
            element="Text",
            text=text,
            confidence=0.9,
            bounding_box=Box(x_min=x, y_min=0, x_max=x + 10, y_max=10),
        )


if __name__ == "__main__":
    unittest.main()
