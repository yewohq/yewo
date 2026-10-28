import io
import threading
import time
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from yewo.agent.perception.detector import ImageDescriptor
from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming
from yewo.agent.perception.rapidocr_adapter import RapidOCRBoundaryError
from yewo.agent.perception.rapidocr_diagnostics import RapidOCRDiagnostic
from yewo.defaults import DEFAULT_PERCEPTION_MODEL


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[0, 0, 20, 20]])
    cls = FakeTensor([0])
    conf = FakeTensor([0.9])


class FakeDetectorResult:
    boxes = FakeBoxes()
    names = {0: "FutureClass"}


class FakeYolo:
    def __init__(self, delay: float = 0):
        self.calls = 0
        self.delay = delay
        self.lock = threading.Lock()

    def __call__(self, _image, **_options):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return [FakeDetectorResult()]


class FakeOCRReader:
    identity = "rapidocr@test/max-side-960"

    def __init__(self, delay: float = 0):
        self.calls = 0
        self.delay = delay
        self.lock = threading.Lock()

    def recognize(self, _image):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return OCRResult(
            engine_identity=self.identity,
            lines=(
                OCRLine(
                    box=(
                        OCRPoint(x=2, y=2),
                        OCRPoint(x=18, y=2),
                        OCRPoint(x=18, y=12),
                        OCRPoint(x=2, y=12),
                    ),
                    text="Save",
                    confidence=0.9,
                ),
            ),
            timing=OCRTiming(engine_total_ms=1, wall_ms=1),
        )


class FusedDescriptorCacheTests(unittest.TestCase):
    def test_ocr_initialization_failure_becomes_an_actionable_boundary_error(self):
        fake_modules = {
            "ultralytics": SimpleNamespace(
                YOLO=lambda _path: SimpleNamespace(names={})
            ),
        }
        manifest = SimpleNamespace(
            inference_image_size=768,
            validate_model_classes=lambda _names: None,
            semantic_roles_by_class_id=lambda: {},
        )
        package = SimpleNamespace(
            reference=SimpleNamespace(value="hf://owner/model@" + "a" * 40),
            weights_path="/tmp/model.pt",
            manifest=manifest,
        )
        with (
            mock.patch.dict("sys.modules", fake_modules),
            mock.patch("yewo.agent.perception.detector.require_vision"),
            mock.patch(
                "yewo.agent.perception.detector.resolve_model_package",
                return_value=package,
            ) as resolve_package,
            mock.patch(
                "yewo.agent.perception.detector.RapidOCRReader",
                side_effect=RuntimeError("native failure"),
            ),
            mock.patch(
                "yewo.agent.perception.detector.diagnose_rapidocr",
                return_value=RapidOCRDiagnostic(
                    "provider_missing",
                    "CPU provider unavailable",
                    "reinstall vision",
                ),
            ) as diagnose,
            self.assertRaisesRegex(RapidOCRBoundaryError, "CPU provider unavailable"),
        ):
            ImageDescriptor()

        resolve_package.assert_called_once_with(DEFAULT_PERCEPTION_MODEL)
        diagnose.assert_called_once_with(smoke_test=True)

    def test_detector_class_roles_reach_fused_elements(self):
        descriptor, _, _ = self._descriptor()
        descriptor.semantic_roles_by_class_id = {0: frozenset({"keyboard_region"})}

        frame, _ = descriptor.observe(self._png("white"))

        self.assertEqual(frame.elements[0].semantic_roles, frozenset({"keyboard_region"}))

    def test_incremental_refresh_can_reuse_detector_and_refresh_only_ocr_crop(self):
        descriptor, yolo, ocr = self._descriptor()
        previous, _ = descriptor.observe(self._png("white"))
        before_yolo = yolo.calls

        elements, _ = descriptor.describe_incremental(
            self._png("black"),
            previous,
            ((4, 4, 24, 24),),
            reuse_detector=True,
        )
        descriptor.observe(self._png("black"))

        self.assertEqual(yolo.calls, before_yolo)
        self.assertEqual(ocr.calls, 2)
        self.assertEqual(elements[0].element, "FutureClass")
        self.assertEqual(len(descriptor.last_frame.image_digest), 64)

    def test_incremental_refresh_runs_detector_when_reuse_is_not_proven(self):
        descriptor, yolo, _ = self._descriptor()
        previous, _ = descriptor.observe(self._png("white"))
        before_yolo = yolo.calls

        descriptor.describe_incremental(
            self._png("black"),
            previous,
            ((4, 4, 24, 24),),
            reuse_detector=False,
        )

        self.assertEqual(yolo.calls, before_yolo + 1)

    def test_exact_frame_cache_reuses_models_and_returns_defensive_copies(self):
        descriptor, yolo, ocr = self._descriptor()
        screenshot = self._png("white")

        first, _ = descriptor.observe(screenshot)
        first.elements[0].text = "changed by consumer"
        second, _ = descriptor.observe(screenshot)

        self.assertEqual(yolo.calls, 1)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(second.elements[0].text, "Save")

    def test_pixel_change_misses_the_cache(self):
        descriptor, yolo, ocr = self._descriptor()

        descriptor.observe(self._png("white"))
        descriptor.observe(self._png("black"))

        self.assertEqual(yolo.calls, 2)
        self.assertEqual(ocr.calls, 2)

    def test_lru_is_bounded_and_clear_removes_sensitive_text(self):
        descriptor, yolo, ocr = self._descriptor(cache_entries=1)
        first = self._png("white")

        descriptor.observe(first)
        descriptor.observe(self._png("black"))
        descriptor.observe(first)
        descriptor.clear_cache()
        descriptor.observe(first)

        self.assertEqual(yolo.calls, 4)
        self.assertEqual(ocr.calls, 4)

    def test_concurrent_duplicate_requests_share_one_inflight_computation(self):
        descriptor, yolo, ocr = self._descriptor(delay=0.05)
        screenshot = self._png("white")

        with ThreadPoolExecutor(max_workers=2) as executor:
            frames = list(executor.map(descriptor.observe, (screenshot, screenshot)))

        self.assertEqual(yolo.calls, 1)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(frames[0][0], frames[1][0])

    @staticmethod
    def _descriptor(
        cache_entries: int = 8,
        delay: float = 0,
    ) -> tuple[ImageDescriptor, FakeYolo, FakeOCRReader]:
        descriptor = ImageDescriptor.__new__(ImageDescriptor)
        yolo = FakeYolo(delay)
        ocr = FakeOCRReader(delay)
        descriptor.yolo = yolo
        descriptor.names = yolo(None)[0].names
        yolo.calls = 0
        descriptor.detection_image_size = 896
        descriptor.model_identity = "ui@test"
        descriptor.ocr_reader = ocr
        descriptor._cache_entries = cache_entries
        descriptor._frame_cache = OrderedDict()
        descriptor._inflight = {}
        descriptor._cache_lock = threading.Lock()
        descriptor._cache_generation = 0
        descriptor._detector_lock = threading.Lock()
        descriptor._ocr_lock = threading.Lock()
        descriptor.last_frame = None
        return descriptor, yolo, ocr

    @staticmethod
    def _png(color: str) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (24, 24), color).save(buffer, format="PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
