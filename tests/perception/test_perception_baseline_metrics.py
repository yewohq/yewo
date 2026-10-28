import unittest
from pathlib import Path

from tools.perception_eval.corpus import PerceptionCase, PerceptionCorpus, TargetPoint
from tools.perception_eval.metrics import (
    Box,
    CaseObservation,
    EvaluationRun,
    ObservedElement,
    score,
)


class PerceptionMetricsTests(unittest.TestCase):
    def test_scores_keyboard_text_targets_and_resources(self):
        corpus = self._corpus()
        run = EvaluationRun(
            detector="detector@revision",
            ocr="ocr@revision",
            observations=[
                self._observation("open", "open-digest", keyboard=True, latency=12, rss=80),
                self._observation("closed", "closed-digest", keyboard=True, latency=8, rss=90),
            ],
        )

        metrics = score(corpus, run)

        self.assertEqual(metrics.keyboard_true_positives, 1)
        self.assertEqual(metrics.keyboard_false_positives, 1)
        self.assertEqual(metrics.keyboard_false_negatives, 0)
        self.assertEqual(metrics.keyboard_precision, 0.5)
        self.assertEqual(metrics.keyboard_recall, 1.0)
        self.assertEqual(metrics.normalized_text_recall, 1.0)
        self.assertEqual(metrics.target_coverage, 1.0)
        self.assertEqual(metrics.mean_mark_count, 2.0)
        self.assertEqual(metrics.mean_latency_ms, 10.0)
        self.assertEqual(metrics.peak_rss_mib, 90.0)

    def test_refuses_an_observation_from_an_old_fixture_version(self):
        corpus = self._corpus()
        observations = [
            self._observation("open", "wrong", keyboard=False, latency=1, rss=1),
            self._observation("closed", "closed-digest", keyboard=False, latency=1, rss=1),
        ]

        with self.assertRaisesRegex(ValueError, "different fixture version"):
            score(corpus, EvaluationRun(detector="d", ocr="o", observations=observations))

    def test_requires_every_case_once(self):
        corpus = self._corpus()
        run = EvaluationRun(
            detector="d",
            ocr="o",
            observations=[self._observation("open", "open-digest", False, 1, 1)],
        )

        with self.assertRaisesRegex(ValueError, "every corpus case"):
            score(corpus, run)

    def test_structural_regions_do_not_count_as_interactive_target_coverage(self):
        corpus = self._corpus()
        observations = [
            self._observation("open", "open-digest", True, 1, 1),
            self._observation("closed", "closed-digest", False, 1, 1),
        ]
        observations[0].elements = [
            ObservedElement(
                element="Keyboard",
                text="Save",
                confidence=0.9,
                bounding_box=Box(x_min=0, y_min=0, x_max=10, y_max=10),
            )
        ]

        metrics = score(corpus, EvaluationRun(detector="d", ocr="o", observations=observations))

        self.assertEqual(metrics.target_coverage, 0.5)

    @staticmethod
    def _corpus() -> PerceptionCorpus:
        cases = [
            PerceptionCase(
                id="open",
                screenshot=Path("open.png"),
                platform="android",
                traits={"keyboard"},
                keyboard_visible=True,
                expected_text=["Save"],
                target_points=[TargetPoint(name="save", x=5, y=5)],
                digest="open-digest",
            ),
            PerceptionCase(
                id="closed",
                screenshot=Path("closed.png"),
                platform="ios",
                traits={"form"},
                keyboard_visible=False,
                expected_text=["Save"],
                target_points=[TargetPoint(name="save", x=5, y=5)],
                digest="closed-digest",
            ),
        ]
        return PerceptionCorpus(
            schema_version=1,
            cases=cases,
            missing_platforms=set(),
            missing_traits=set(),
        )

    @staticmethod
    def _observation(
        case_id: str, digest: str, keyboard: bool, latency: float, rss: float
    ) -> CaseObservation:
        return CaseObservation(
            case_id=case_id,
            fixture_digest=digest,
            visual_keyboard_visible=keyboard,
            mark_count=2,
            latency_ms=latency,
            peak_rss_mib=rss,
            elements=[
                ObservedElement(
                    element="Text",
                    text="  SAVE ",
                    confidence=0.9,
                    bounding_box=Box(x_min=0, y_min=0, x_max=10, y_max=10),
                )
            ],
        )
