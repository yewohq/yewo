import unittest
from pathlib import Path

from tools.perception_eval.corpus import PerceptionCase, PerceptionCorpus, TargetPoint
from tools.perception_eval.ocr_evaluation import (
    OCRCaseObservation,
    OCREvaluationRun,
    score_ocr,
)
from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming


class OCREvaluationTests(unittest.TestCase):
    def test_scores_exact_contained_quality_and_cold_warm_resources(self):
        corpus = self._corpus()
        run = OCREvaluationRun(
            ocr="rapidocr@test",
            observations=[
                self._observation(
                    "first",
                    "first-digest",
                    ["Don't have an account? Sign Up", "保存"],
                    wall_ms=120,
                    engine_ms=100,
                    rss=80,
                ),
                self._observation(
                    "second",
                    "second-digest",
                    ["Address"],
                    wall_ms=20,
                    engine_ms=15,
                    rss=90,
                ),
            ],
        )

        metrics = score_ocr(corpus, run)

        self.assertEqual(metrics.cases, 2)
        self.assertEqual(metrics.expected_text_count, 3)
        self.assertEqual(metrics.exact_line_recall, 0.3333)
        self.assertEqual(metrics.contained_text_recall, 0.6667)
        self.assertEqual(metrics.mean_line_count, 1.5)
        self.assertEqual(metrics.cold_wall_ms, 120)
        self.assertEqual(metrics.mean_warm_wall_ms, 20)
        self.assertEqual(metrics.mean_engine_ms, 57.5)
        self.assertEqual(metrics.peak_rss_mib, 90)

    def test_refuses_an_old_fixture_digest(self):
        corpus = self._corpus()
        run = OCREvaluationRun(
            ocr="rapidocr@test",
            observations=[
                self._observation("first", "wrong", [], 1, 1, 1),
                self._observation("second", "second-digest", [], 1, 1, 1),
            ],
        )

        with self.assertRaisesRegex(ValueError, "different fixture version"):
            score_ocr(corpus, run)

    def test_refuses_duplicate_or_reordered_cases(self):
        corpus = self._corpus()
        run = OCREvaluationRun(
            ocr="rapidocr@test",
            observations=[
                self._observation("second", "second-digest", [], 1, 1, 1),
                self._observation("first", "first-digest", [], 1, 1, 1),
            ],
        )

        with self.assertRaisesRegex(ValueError, "cases and order exactly"):
            score_ocr(corpus, run)

    @staticmethod
    def _corpus() -> PerceptionCorpus:
        cases = [
            PerceptionCase(
                id="first",
                screenshot=Path("first.png"),
                platform="ios",
                traits={"form"},
                keyboard_visible=False,
                expected_text=["Sign Up", "保存"],
                target_points=[TargetPoint(name="save", x=1, y=1)],
                digest="first-digest",
            ),
            PerceptionCase(
                id="second",
                screenshot=Path("second.png"),
                platform="android",
                traits={"form"},
                keyboard_visible=False,
                expected_text=["Add"],
                target_points=[TargetPoint(name="add", x=1, y=1)],
                digest="second-digest",
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
        case_id: str,
        digest: str,
        texts: list[str],
        wall_ms: float,
        engine_ms: float,
        rss: float,
    ) -> OCRCaseObservation:
        box = (
            OCRPoint(x=0, y=0),
            OCRPoint(x=2, y=0),
            OCRPoint(x=2, y=2),
            OCRPoint(x=0, y=2),
        )
        return OCRCaseObservation(
            case_id=case_id,
            fixture_digest=digest,
            peak_rss_mib=rss,
            result=OCRResult(
                engine_identity="rapidocr@test",
                lines=tuple(OCRLine(box=box, text=text, confidence=0.9) for text in texts),
                timing=OCRTiming(
                    detection_ms=1,
                    classification_ms=1,
                    recognition_ms=1,
                    engine_total_ms=engine_ms,
                    wall_ms=wall_ms,
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
