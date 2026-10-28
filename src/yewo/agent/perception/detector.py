"""Concurrent UI detection and whole-screen OCR with geometry-only fusion."""

import hashlib
import io
import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from PIL import Image, ImageDraw, ImageFont

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.fusion import FUSION_VERSION, PerceptionFrame, fuse_perception
from yewo.agent.perception.incremental import merge_ocr_evidence, rescale_ocr_result
from yewo.agent.perception.model_package import resolve_model_package
from yewo.agent.perception.ocr import OCRReader
from yewo.agent.perception.rapidocr_adapter import RapidOCRBoundaryError, RapidOCRReader
from yewo.agent.perception.rapidocr_diagnostics import diagnose_rapidocr
from yewo.defaults import DEFAULT_PERCEPTION_MODEL
from yewo.utils.vision_deps import require_vision

logger = logging.getLogger(__name__)


OCR_MAX_SIDE_LENGTH = 960
FRAME_CACHE_ENTRIES = 8


class ImageDescriptor:
    def __init__(
        self,
        model_reference: str | None = None,
        ocr_reader: OCRReader | None = None,
        cache_entries: int = FRAME_CACHE_ENTRIES,
    ):
        if cache_entries <= 0:
            raise ValueError("cache_entries must be positive")
        require_vision()
        # Deferred: these pull torch, which lives behind the 'vision' extra.
        from ultralytics import YOLO

        package = resolve_model_package(model_reference or DEFAULT_PERCEPTION_MODEL)
        logger.info("Loading %s...", package.reference.value)
        self.yolo = YOLO(package.weights_path)
        self.detection_image_size = package.manifest.inference_image_size
        self.model_identity = package.reference.value
        self.names = self.yolo.names
        package.manifest.validate_model_classes(self.names)
        self.semantic_roles_by_class_id = package.manifest.semantic_roles_by_class_id()
        logger.info("✓ Model loaded")
        logger.info("Loading RapidOCR...")
        if ocr_reader is not None:
            self.ocr_reader = ocr_reader
        else:
            try:
                self.ocr_reader = RapidOCRReader(max_side_len=OCR_MAX_SIDE_LENGTH)
            except Exception as exc:  # noqa: BLE001 - normalize the external native boundary
                diagnostic = diagnose_rapidocr(smoke_test=True)
                raise RapidOCRBoundaryError(
                    f"RapidOCR could not initialize: {diagnostic.detail}. {diagnostic.hint}"
                ) from exc
        logger.info("✓ OCR loaded")
        self._cache_entries = cache_entries
        self._frame_cache: OrderedDict[str, PerceptionFrame] = OrderedDict()
        self._inflight: dict[str, Future[PerceptionFrame]] = {}
        self._cache_lock = Lock()
        self._cache_generation = 0
        self._detector_lock = Lock()
        self._ocr_lock = Lock()
        self.last_frame: PerceptionFrame | None = None

    def describe(
        self,
        png_bytes: bytes,
        box_color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> tuple[list[GUIElement], Image.Image]:
        frame, image = self.observe(png_bytes)
        elements = [element.model_copy(deep=True) for element in frame.elements]
        return elements, self._draw_gui_elements(image, elements, box_color, thickness)

    def observe(self, png_bytes: bytes) -> tuple[PerceptionFrame, Image.Image]:
        """Return independent evidence and fused elements for one screenshot."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image_digest = _pixel_digest(image)
        cache_key = self._cache_key(image_digest)
        frame = self._cached_or_compute(cache_key, image_digest, image)
        returned_frame = frame.model_copy(deep=True)
        self.last_frame = returned_frame
        return returned_frame, image

    def describe_incremental(
        self,
        png_bytes: bytes,
        previous_frame: PerceptionFrame,
        regions: tuple[tuple[int, int, int, int], ...],
        reuse_detector: bool,
        box_color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> tuple[list[GUIElement], Image.Image]:
        """Refresh OCR crops and optionally reuse detector geometry."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image_digest = _pixel_digest(image)
        with self._cache_lock:
            cache_generation = self._cache_generation

        def refresh_ocr():
            reference_max_side = getattr(self.ocr_reader, "reference_max_side", None)
            scale = min(1.0, reference_max_side / max(image.size)) if reference_max_side else 1.0
            refreshed = []
            for region in regions:
                crop = image.crop(region)
                if scale < 1.0:
                    crop = crop.resize(
                        (
                            max(1, round(crop.width * scale)),
                            max(1, round(crop.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                result = self._run_ocr(crop)
                if scale < 1.0:
                    result = rescale_ocr_result(result, 1.0 / scale)
                refreshed.append((region, result))
            return merge_ocr_evidence(
                previous_frame.text_lines,
                refreshed,
                self.ocr_reader.identity,
            )

        if reuse_detector:
            detector_elements = _detector_evidence(previous_frame.elements)
            ocr_result = refresh_ocr()
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                detector_future = executor.submit(self._run_detector, image)
                ocr_future = executor.submit(refresh_ocr)
                result = detector_future.result()
                ocr_result = ocr_future.result()
            names = result.names if hasattr(result, "names") else self.names
            detector_elements = self._get_yolo_gui_elements(result, names)

        frame = fuse_perception(
            detector_elements,
            ocr_result,
            image_digest,
            self.model_identity,
        )
        cache_key = self._cache_key(image_digest)
        with self._cache_lock:
            if cache_generation == self._cache_generation:
                self._frame_cache[cache_key] = frame
                while len(self._frame_cache) > self._cache_entries:
                    self._frame_cache.popitem(last=False)
        self.last_frame = frame.model_copy(deep=True)
        elements = [element.model_copy(deep=True) for element in frame.elements]
        return elements, self._draw_gui_elements(image, elements, box_color, thickness)

    def clear_cache(self) -> None:
        """Forget session-scoped OCR text and detector observations."""
        with self._cache_lock:
            self._frame_cache.clear()
            self._cache_generation += 1
        self.last_frame = None

    def _cache_key(self, image_digest: str) -> str:
        return ":".join(
            (
                image_digest,
                self.model_identity,
                str(self.detection_image_size),
                self.ocr_reader.identity,
                f"fusion-{FUSION_VERSION}",
            )
        )

    def _cached_or_compute(
        self,
        cache_key: str,
        image_digest: str,
        image: Image.Image,
    ) -> PerceptionFrame:
        with self._cache_lock:
            cached = self._frame_cache.pop(cache_key, None)
            if cached is not None:
                self._frame_cache[cache_key] = cached
                return cached
            future = self._inflight.get(cache_key)
            owns_computation = future is None
            generation = self._cache_generation
            if future is None:
                future = Future()
                self._inflight[cache_key] = future

        if not owns_computation:
            return future.result()

        try:
            frame = self._compute_frame(image, image_digest)
        except Exception as exc:
            with self._cache_lock:
                self._inflight.pop(cache_key, None)
                future.set_exception(exc)
            raise

        with self._cache_lock:
            if generation == self._cache_generation:
                self._frame_cache[cache_key] = frame
                while len(self._frame_cache) > self._cache_entries:
                    self._frame_cache.popitem(last=False)
            self._inflight.pop(cache_key, None)
            future.set_result(frame)
        return frame

    def _compute_frame(self, image: Image.Image, image_digest: str) -> PerceptionFrame:
        with ThreadPoolExecutor(max_workers=2) as executor:
            detector_future = executor.submit(self._run_detector, image)
            ocr_future = executor.submit(self._run_ocr, image)
            result = detector_future.result()
            ocr_result = ocr_future.result()
        names = result.names if hasattr(result, "names") else self.names
        gui_elements = self._get_yolo_gui_elements(result, names)
        return fuse_perception(
            gui_elements,
            ocr_result,
            image_digest,
            self.model_identity,
        )

    def _run_detector(self, image: Image.Image):
        with self._detector_lock:
            return self.yolo(image, imgsz=self.detection_image_size, verbose=False)[0]

    def _run_ocr(self, image: Image.Image):
        with self._ocr_lock:
            return self.ocr_reader.recognize(image)

    def _get_yolo_gui_elements(self, result, names) -> list[GUIElement]:
        if result.boxes is None:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        elements = []
        for box, cls_id, conf in zip(xyxy, clss, confs):
            x_min, y_min, x_max, y_max = box
            class_name = (
                names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            )
            elements.append(
                GUIElement(
                    element=class_name,
                    bounding_box=BoundingBox(
                        x_min=int(x_min),
                        y_min=int(y_min),
                        x_max=int(x_max),
                        y_max=int(y_max),
                    ),
                    confidence=float(conf),
                    type="yolo_detection",
                    semantic_roles=getattr(self, "semantic_roles_by_class_id", {}).get(
                        int(cls_id), frozenset()
                    ),
                )
            )
        return elements

    def _draw_gui_elements(self, image, elements, box_color, thickness) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        w, h = image.size
        scale_ref = max(w, h)
        t = max(1, int(thickness * (scale_ref / 1000.0)))
        font_size = max(10, int(14 * (scale_ref / 1200.0)))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for item in elements:
            x1, y1, x2, y2 = item.bounding_box
            label_text = f"{item.element} {item.confidence:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=t)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x, text_y = max(0, x2 - tw - 3), max(3, y1 - th - 5)
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + tw + 2, text_y + th + 2],
                fill=(0, 0, 0),
            )
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
        return image


def _pixel_digest(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:{image.mode}\0".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _detector_evidence(elements: tuple[GUIElement, ...]) -> list[GUIElement]:
    """Remove fused OCR text while retaining detector geometry and semantics."""
    evidence = [element.model_copy(deep=True) for element in elements]
    for element in evidence:
        element.text = ""
    return evidence
