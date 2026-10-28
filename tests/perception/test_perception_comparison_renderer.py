import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.perception_eval.corpus import read_corpus
from tools.perception_eval.metrics import Box, CaseObservation, EvaluationRun, ObservedElement
from tools.perception_eval.render_comparison import render_comparison


class PerceptionComparisonRendererTests(unittest.TestCase):
    def test_renders_side_by_side_case_overview_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self._write_corpus(root)
            case = read_corpus(manifest_path).cases[0]
            left_run = self._run("left/model@revision", case.digest, "Text")
            right_run = self._run("right/model@revision", case.digest, "View")
            output_directory = root / "runs" / "timestamp"

            comparison_paths = render_comparison(
                left_run,
                right_run,
                output_directory,
                manifest_path,
            )

            self.assertEqual([path.name for path in comparison_paths], ["sample.png"])
            self.assertTrue((output_directory / "overview.png").is_file())
            summary = json.loads((output_directory / "summary.json").read_text())
            self.assertEqual(summary["left"]["detector"], "left/model@revision")
            self.assertEqual(summary["right"]["detector"], "right/model@revision")
            self.assertEqual(summary["left"]["metrics"]["target_coverage"], 1.0)
            with Image.open(comparison_paths[0]) as comparison:
                self.assertGreater(comparison.width, comparison.height)

    @staticmethod
    def _write_corpus(root: Path) -> Path:
        screens = root / "screens"
        screens.mkdir()
        Image.new("RGB", (100, 60), "white").save(screens / "sample.png")
        manifest = {
            "schema_version": 1,
            "required_platforms": ["ios"],
            "required_traits": ["form"],
            "cases": [
                {
                    "id": "sample",
                    "screenshot": "sample.png",
                    "platform": "ios",
                    "traits": ["form"],
                    "keyboard_visible": False,
                    "expected_text": ["Save"],
                    "target_points": [{"name": "save", "x": 50, "y": 30}],
                    "provenance": {
                        "source": "test fixture",
                        "license": "project-owned",
                        "training_overlap": "excluded",
                    },
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _run(detector: str, digest: str, element: str) -> EvaluationRun:
        return EvaluationRun(
            detector=detector,
            inference_image_size=100,
            ocr="test-ocr",
            observations=[
                CaseObservation(
                    case_id="sample",
                    fixture_digest=digest,
                    visual_keyboard_visible=False,
                    mark_count=1,
                    latency_ms=1,
                    peak_rss_mib=1,
                    elements=[
                        ObservedElement(
                            element=element,
                            text="Save",
                            confidence=0.9,
                            bounding_box=Box(x_min=25, y_min=20, x_max=75, y_max=40),
                        )
                    ],
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
