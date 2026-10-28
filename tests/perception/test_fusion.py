import unittest

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.fusion import fuse_perception
from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming


class PerceptionFusionTests(unittest.TestCase):
    def test_associates_by_geometry_without_consulting_the_element_class(self):
        elements = [self._element("UnexpectedFutureClass", 0, 0, 100, 50)]

        frame = fuse_perception(
            elements, self._ocr([("Save", 10, 10, 60, 30)]), "a" * 64, "ui@test"
        )

        self.assertEqual(frame.elements[0].text, "Save")
        self.assertEqual(frame.associations[0].element_index, 0)
        self.assertEqual(frame.associations[0].line_index, 0)

    def test_prefers_the_smallest_container_and_keeps_unmatched_text_independent(self):
        elements = [
            self._element("View", 0, 0, 200, 200),
            self._element("Control", 20, 20, 120, 80),
        ]
        result = self._ocr(
            [
                ("Inside", 30, 30, 100, 60),
                ("Outside", 220, 220, 280, 250),
            ]
        )

        frame = fuse_perception(elements, result, "b" * 64, "ui@test")

        self.assertEqual(frame.elements[0].text, "")
        self.assertEqual(frame.elements[1].text, "Inside")
        self.assertEqual(len(frame.text_lines), 2)
        self.assertEqual(len(frame.associations), 1)
        self.assertEqual(frame.unmatched_text(), ("Outside",))
        self.assertEqual(len(frame.elements), 2)
        self.assertEqual(len(frame.ocr_redaction_evidence()), 2)
        self.assertTrue(all(item.type == "ocr_evidence" for item in frame.ocr_redaction_evidence()))

    def test_orders_multiple_lines_and_does_not_mutate_detector_output(self):
        element = self._element("Anything", 0, 0, 200, 200)
        result = self._ocr(
            [
                ("Second", 10, 50, 80, 70),
                ("First", 10, 10, 70, 30),
            ]
        )

        frame = fuse_perception([element], result, "c" * 64, "ui@test")

        self.assertEqual(frame.elements[0].text, "First Second")
        self.assertEqual(element.text, "")

    @staticmethod
    def _element(label: str, x1: int, y1: int, x2: int, y2: int) -> GUIElement:
        return GUIElement(
            element=label,
            bounding_box=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
            confidence=0.9,
            type="yolo_detection",
        )

    @staticmethod
    def _ocr(lines: list[tuple[str, int, int, int, int]]) -> OCRResult:
        return OCRResult(
            engine_identity="rapidocr@test",
            lines=tuple(
                OCRLine(
                    box=(
                        OCRPoint(x=x1, y=y1),
                        OCRPoint(x=x2, y=y1),
                        OCRPoint(x=x2, y=y2),
                        OCRPoint(x=x1, y=y2),
                    ),
                    text=text,
                    confidence=0.9,
                )
                for text, x1, y1, x2, y2 in lines
            ),
            timing=OCRTiming(engine_total_ms=1, wall_ms=1),
        )


if __name__ == "__main__":
    unittest.main()
