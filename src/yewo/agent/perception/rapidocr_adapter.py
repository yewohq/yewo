"""Normalize RapidOCR's dynamic output into Yewo's OCR contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, SupportsFloat, cast

import numpy as np
from PIL import Image

from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming

# RapidOCR names this multilingual PP-OCRv6 routing alias ``ch``. For the
# small v6 model, supported language aliases resolve to the same multilingual
# detector and recognizer weights; this does not restrict Yewo to Chinese UI.
RAPIDOCR_LANGUAGE_ROUTE = "ch"
OCR_MODEL_VERSION = "PP-OCRv6"
OCR_MODEL_SIZE = "small"
RAPIDOCR_MODEL_FILES = (
    "PP-OCRv6_det_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "PP-OCRv6_rec_small.onnx",
)


class RapidOCRBoundaryError(RuntimeError):
    """RapidOCR returned a result that cannot satisfy Yewo's OCR contract."""


class RapidOCREngine(Protocol):
    """Call shape used from the external RapidOCR package."""

    def __call__(
        self,
        image: Image.Image,
        *,
        use_det: bool,
        use_cls: bool,
        use_rec: bool,
    ) -> object: ...


EngineFactory = Callable[[], RapidOCREngine]


class RapidOCRReader:
    """Run one whole-screen RapidOCR pass and validate its observations."""

    def __init__(
        self,
        engine_factory: EngineFactory | None = None,
        identity: str | None = None,
        max_side_len: int | None = None,
    ):
        if max_side_len is not None and max_side_len <= 0:
            raise ValueError("max_side_len must be positive")
        factory = engine_factory or (lambda: _build_engine(max_side_len))
        self._engine = factory()
        self._identity = identity or build_rapidocr_identity(max_side_len)
        self._max_side_len = max_side_len

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def reference_max_side(self) -> int | None:
        """The scale ceiling partial crops must share with a full-screen pass."""
        return self._max_side_len

    def recognize(self, image: Image.Image) -> OCRResult:
        """Read the complete screenshot once, without consulting detector classes."""
        started = time.perf_counter()
        raw_result = self._engine(image, use_det=True, use_cls=True, use_rec=True)
        wall_ms = (time.perf_counter() - started) * 1000
        return _normalize_result(raw_result, image.size, self.identity, wall_ms)


def _build_engine(max_side_len: int | None = None) -> RapidOCREngine:
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

    params = {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": RAPIDOCR_LANGUAGE_ROUTE,
        "Det.model_type": ModelType(OCR_MODEL_SIZE),
        "Det.ocr_version": OCRVersion(OCR_MODEL_VERSION),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": RAPIDOCR_LANGUAGE_ROUTE,
        "Rec.model_type": ModelType(OCR_MODEL_SIZE),
        "Rec.ocr_version": OCRVersion(OCR_MODEL_VERSION),
    }
    if max_side_len is not None:
        params["Global.max_side_len"] = max_side_len
    return cast(
        RapidOCREngine,
        RapidOCR(params=params),
    )


def build_rapidocr_identity(max_side_len: int | None = None) -> str:
    """Return the exact installed engine and model configuration identity."""
    identity = (
        f"rapidocr@{_package_version('rapidocr')}+"
        f"onnxruntime@{_package_version('onnxruntime')}/"
        f"{OCR_MODEL_VERSION}-{OCR_MODEL_SIZE}:{RAPIDOCR_LANGUAGE_ROUTE}"
    )
    return f"{identity}/max-side-{max_side_len}" if max_side_len is not None else identity


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as exc:
        raise RapidOCRBoundaryError(f"required OCR package is not installed: {package}") from exc


def _normalize_result(
    raw_result: object,
    image_size: tuple[int, int],
    identity: str,
    wall_ms: float,
) -> OCRResult:
    boxes = _read_field(raw_result, "boxes")
    texts = _read_sequence(raw_result, "txts")
    scores = _read_sequence(raw_result, "scores")
    if boxes is None:
        if texts or scores:
            raise RapidOCRBoundaryError("RapidOCR returned text or scores without boxes")
        lines: tuple[OCRLine, ...] = ()
    else:
        box_array = np.asarray(boxes, dtype=float)
        if box_array.shape != (len(texts), 4, 2):
            raise RapidOCRBoundaryError(
                "RapidOCR boxes must have shape (line_count, 4, 2): "
                f"boxes={box_array.shape}, text_count={len(texts)}"
            )
        if len(scores) != len(texts):
            raise RapidOCRBoundaryError(
                f"RapidOCR result lengths differ: texts={len(texts)}, scores={len(scores)}"
            )
        lines = tuple(
            _normalize_line(box, text, score, image_size)
            for box, text, score in zip(box_array, texts, scores, strict=True)
        )
    return OCRResult(
        engine_identity=identity,
        lines=lines,
        timing=_normalize_timing(raw_result, wall_ms),
    )


def _normalize_line(
    box: np.ndarray,
    text: object,
    score: object,
    image_size: tuple[int, int],
) -> OCRLine:
    if not isinstance(text, str):
        raise RapidOCRBoundaryError(f"RapidOCR text must be a string, got {type(text).__name__}")
    width, height = image_size
    try:
        points = (
            _normalize_point(box[0], width, height),
            _normalize_point(box[1], width, height),
            _normalize_point(box[2], width, height),
            _normalize_point(box[3], width, height),
        )
        confidence = _to_float(score, "confidence")
        return OCRLine(box=points, text=text, confidence=confidence)
    except RapidOCRBoundaryError:
        raise
    except (TypeError, ValueError) as exc:
        raise RapidOCRBoundaryError(f"invalid RapidOCR line: {exc}") from exc


def _normalize_point(point: np.ndarray, width: int, height: int) -> OCRPoint:
    return OCRPoint(
        x=min(max(_to_float(point[0], "x coordinate"), 0.0), float(width)),
        y=min(max(_to_float(point[1], "y coordinate"), 0.0), float(height)),
    )


def _normalize_timing(raw_result: object, wall_ms: float) -> OCRTiming:
    stages = _read_sequence(raw_result, "elapse_list")
    if not stages:
        return OCRTiming(engine_total_ms=wall_ms, wall_ms=wall_ms)
    if len(stages) != 3:
        raise RapidOCRBoundaryError(
            f"RapidOCR elapse_list must contain three stages, got {len(stages)}"
        )
    engine_total = _read_field(raw_result, "elapse")
    if engine_total is None:
        raise RapidOCRBoundaryError("RapidOCR result has no total elapsed time")
    try:
        durations_ms = [_to_float(duration, "stage elapsed time") * 1000 for duration in stages]
        engine_total_ms = _to_float(engine_total, "total elapsed time") * 1000
        return OCRTiming(
            detection_ms=durations_ms[0],
            classification_ms=durations_ms[1],
            recognition_ms=durations_ms[2],
            engine_total_ms=engine_total_ms,
            wall_ms=wall_ms,
        )
    except RapidOCRBoundaryError:
        raise
    except (TypeError, ValueError) as exc:
        raise RapidOCRBoundaryError("RapidOCR timing values must be numeric") from exc


def _to_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float, np.number)):
        raise RapidOCRBoundaryError(f"RapidOCR {label} must be numeric")
    return float(cast(SupportsFloat, value))


def _read_field(result: object, name: str) -> object | None:
    try:
        return getattr(result, name)
    except AttributeError as exc:
        raise RapidOCRBoundaryError(f"RapidOCR result is missing {name!r}") from exc


def _read_sequence(result: object, name: str) -> Sequence[object]:
    value = _read_field(result, name)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise RapidOCRBoundaryError(f"RapidOCR {name} must be a sequence")
    return value
