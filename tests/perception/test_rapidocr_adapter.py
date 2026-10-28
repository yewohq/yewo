import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from yewo.agent.perception.rapidocr_adapter import (
    OCR_MODEL_SIZE,
    OCR_MODEL_VERSION,
    RAPIDOCR_LANGUAGE_ROUTE,
    RapidOCRBoundaryError,
    RapidOCRReader,
    build_rapidocr_identity,
)


class FakeRapidOCREngine:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, image, *, use_det, use_cls, use_rec):
        self.calls.append((image, use_det, use_cls, use_rec))
        return self.result


class RapidOCRReaderTests(unittest.TestCase):
    @patch(
        "yewo.agent.perception.rapidocr_adapter._package_version",
        side_effect=("3.10.0", "1.31.1"),
    )
    def test_identity_comes_from_the_same_explicit_model_configuration(self, _version):
        self.assertEqual(OCR_MODEL_VERSION, "PP-OCRv6")
        self.assertEqual(OCR_MODEL_SIZE, "small")
        self.assertEqual(RAPIDOCR_LANGUAGE_ROUTE, "ch")
        self.assertEqual(
            build_rapidocr_identity(960),
            "rapidocr@3.10.0+onnxruntime@1.31.1/PP-OCRv6-small:ch/max-side-960",
        )

    def test_normalizes_one_whole_screen_call_and_clamps_coordinates(self):
        raw_result = self._result(
            boxes=np.array([[[-1, -2], [12, 0], [12, 9], [0, 9]]]),
            texts=("Save",),
            scores=(0.91,),
        )
        engine = FakeRapidOCREngine(raw_result)
        reader = RapidOCRReader(lambda: engine, identity="rapidocr@test")
        image = Image.new("RGB", (10, 8), "white")

        result = reader.recognize(image)

        self.assertEqual(engine.calls, [(image, True, True, True)])
        self.assertEqual(result.engine_identity, "rapidocr@test")
        self.assertEqual(result.lines[0].text, "Save")
        self.assertEqual(result.lines[0].confidence, 0.91)
        self.assertEqual(
            [(point.x, point.y) for point in result.lines[0].box],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        )
        self.assertEqual(result.timing.detection_ms, 10)
        self.assertEqual(result.timing.classification_ms, 20)
        self.assertEqual(result.timing.recognition_ms, 30)
        self.assertEqual(result.timing.engine_total_ms, 60)
        self.assertGreaterEqual(result.timing.wall_ms, 0)

    def test_accepts_an_empty_reading(self):
        reader = RapidOCRReader(
            lambda: FakeRapidOCREngine(
                self._result(
                    boxes=np.empty((0, 4, 2)),
                    texts=(),
                    scores=(),
                )
            ),
            identity="rapidocr@test",
        )

        result = reader.recognize(Image.new("RGB", (10, 8), "white"))

        self.assertEqual(result.lines, ())

    def test_accepts_an_empty_reading_without_upstream_stage_timings(self):
        raw_result = self._result(
            boxes=None,
            texts=(),
            scores=(),
        )
        raw_result.elapse_list = ()
        reader = RapidOCRReader(
            lambda: FakeRapidOCREngine(raw_result),
            identity="rapidocr@test",
        )

        result = reader.recognize(Image.new("RGB", (10, 8), "white"))

        self.assertEqual(result.lines, ())
        self.assertIsNone(result.timing.detection_ms)
        self.assertEqual(result.timing.engine_total_ms, result.timing.wall_ms)

    def test_rejects_mismatched_output_lengths(self):
        reader = RapidOCRReader(
            lambda: FakeRapidOCREngine(
                self._result(
                    boxes=np.zeros((1, 4, 2)),
                    texts=("one",),
                    scores=(),
                )
            ),
            identity="rapidocr@test",
        )

        with self.assertRaisesRegex(RapidOCRBoundaryError, "lengths differ"):
            reader.recognize(Image.new("RGB", (10, 8), "white"))

    def test_rejects_a_malformed_box_shape(self):
        reader = RapidOCRReader(
            lambda: FakeRapidOCREngine(
                self._result(
                    boxes=np.zeros((1, 2, 2)),
                    texts=("one",),
                    scores=(0.5,),
                )
            ),
            identity="rapidocr@test",
        )

        with self.assertRaisesRegex(RapidOCRBoundaryError, "shape"):
            reader.recognize(Image.new("RGB", (10, 8), "white"))

    def test_rejects_non_finite_output(self):
        reader = RapidOCRReader(
            lambda: FakeRapidOCREngine(
                self._result(
                    boxes=np.array([[[0, 0], [1, 0], [1, np.nan], [0, 1]]]),
                    texts=("one",),
                    scores=(0.5,),
                )
            ),
            identity="rapidocr@test",
        )

        with self.assertRaisesRegex(RapidOCRBoundaryError, "invalid RapidOCR line"):
            reader.recognize(Image.new("RGB", (10, 8), "white"))

    def test_rejects_an_invalid_maximum_side_length(self):
        with self.assertRaisesRegex(ValueError, "max_side_len must be positive"):
            RapidOCRReader(
                lambda: FakeRapidOCREngine(self._result(boxes=None, texts=(), scores=())),
                identity="rapidocr@test",
                max_side_len=0,
            )

    @staticmethod
    def _result(*, boxes, texts, scores):
        return SimpleNamespace(
            boxes=boxes,
            txts=texts,
            scores=scores,
            elapse_list=(0.01, 0.02, 0.03),
            elapse=0.06,
        )


if __name__ == "__main__":
    unittest.main()
