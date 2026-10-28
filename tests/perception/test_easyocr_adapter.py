import unittest

import numpy as np
from PIL import Image

from yewo.agent.perception.easyocr_adapter import (
    EasyOCRBoundaryError,
    EasyOCRReader,
)


class FakeEasyOCREngine:
    def __init__(self, result):
        self.result = result
        self.images = []

    def readtext(self, image):
        self.images.append(image)
        return self.result


class EasyOCRReaderTests(unittest.TestCase):
    def test_normalizes_one_whole_screen_call_and_clamps_coordinates(self):
        engine = FakeEasyOCREngine(
            [
                (
                    [[-1, -2], [12, 0], [12, 9], [0, 9]],
                    "Save",
                    np.float64(0.91),
                )
            ]
        )
        reader = EasyOCRReader(lambda: engine, identity="easyocr@test")

        result = reader.recognize(Image.new("RGB", (10, 8), "white"))

        self.assertEqual(len(engine.images), 1)
        self.assertEqual(engine.images[0].shape, (8, 10, 3))
        self.assertEqual(result.engine_identity, "easyocr@test")
        self.assertEqual(result.lines[0].text, "Save")
        self.assertEqual(
            [(point.x, point.y) for point in result.lines[0].box],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        )
        self.assertIsNone(result.timing.detection_ms)
        self.assertGreaterEqual(result.timing.wall_ms, 0)

    def test_rejects_a_malformed_box(self):
        reader = EasyOCRReader(
            lambda: FakeEasyOCREngine([([[0, 0], [1, 1]], "Save", 0.9)]),
            identity="easyocr@test",
        )

        with self.assertRaisesRegex(EasyOCRBoundaryError, "shape"):
            reader.recognize(Image.new("RGB", (10, 8), "white"))


if __name__ == "__main__":
    unittest.main()
