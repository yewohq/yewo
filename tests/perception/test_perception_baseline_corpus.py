import json
import tempfile
import unittest
from pathlib import Path

from fakes import solid_png

from tools.perception_eval.corpus import CorpusError, read_corpus


class PerceptionCorpusTests(unittest.TestCase):
    def test_repository_corpus_covers_declared_platforms_and_traits(self):
        corpus = read_corpus()

        self.assertEqual(corpus.missing_platforms, set())
        self.assertEqual(corpus.missing_traits, set())
        self.assertEqual(len(corpus.cases), 14)
        self.assertTrue(all(len(case.digest) == 16 for case in corpus.cases))
        self.assertTrue(
            all(case.provenance.training_overlap == "excluded" for case in corpus.cases)
        )

    def test_screenshot_cannot_escape_its_fixture_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "screens").mkdir()
            manifest = self._manifest("../secret.png")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CorpusError, "plain .png"):
                read_corpus(path)

    def test_fixture_requires_real_png_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screens = root / "screens"
            screens.mkdir()
            (screens / "screen.png").write_text("not an image", encoding="utf-8")
            path = root / "manifest.json"
            path.write_text(json.dumps(self._manifest("screen.png")), encoding="utf-8")

            with self.assertRaisesRegex(CorpusError, "is not a PNG"):
                read_corpus(path)

    def test_trait_order_does_not_change_the_fixture_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screens = root / "screens"
            screens.mkdir()
            (screens / "screen.png").write_bytes(solid_png())
            path = root / "manifest.json"
            manifest = self._manifest("screen.png")
            manifest["cases"][0]["traits"] = ["keyboard", "form"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            first = read_corpus(path).cases[0].digest
            manifest["cases"][0]["traits"] = ["form", "keyboard"]
            path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(read_corpus(path).cases[0].digest, first)

    @staticmethod
    def _manifest(screenshot: str) -> dict:
        return {
            "schema_version": 1,
            "required_platforms": ["android"],
            "required_traits": ["form"],
            "cases": [
                {
                    "id": "case",
                    "screenshot": screenshot,
                    "platform": "android",
                    "traits": ["form"],
                    "keyboard_visible": False,
                    "expected_text": ["Save"],
                    "target_points": [{"name": "save", "x": 5, "y": 5}],
                    "provenance": {
                        "source": "test fixture",
                        "license": "project-owned",
                        "training_overlap": "excluded",
                    },
                }
            ],
        }
