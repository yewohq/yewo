"""Normalize independent whole-screen EasyOCR output into Yewo's OCR contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, SupportsFloat, cast

import numpy as np
from PIL import Image

from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming


class EasyOCRBoundaryError(RuntimeError):
    """EasyOCR returned a result that cannot satisfy Yewo's OCR contract."""


class EasyOCREngine(Protocol):
    """Call shape used from an initialized EasyOCR reader."""

    def readtext(self, image: np.ndarray) -> object: ...


EngineFactory = Callable[[], EasyOCREngine]


class EasyOCRReader:
    """Run one independent whole-screen EasyOCR pass for comparison."""

    def __init__(
        self,
        engine_factory: EngineFactory | None = None,
        identity: str | None = None,
    ):
        self._engine = (engine_factory or _build_engine)()
        self._identity = identity or _build_identity()

    @property
    def identity(self) -> str:
        return self._identity

    def recognize(self, image: Image.Image) -> OCRResult:
        """Read the complete screenshot once, without consulting detector classes."""
        started = time.perf_counter()
        raw_result = self._engine.readtext(np.asarray(image.convert("RGB")))
        wall_ms = (time.perf_counter() - started) * 1000
        return OCRResult(
            engine_identity=self.identity,
            lines=_normalize_lines(raw_result, image.size),
            timing=OCRTiming(engine_total_ms=wall_ms, wall_ms=wall_ms),
        )


def _build_engine() -> EasyOCREngine:
    import easyocr

    return cast(EasyOCREngine, easyocr.Reader(["en"], gpu=False))


def _build_identity() -> str:
    try:
        package_version = version("easyocr")
    except PackageNotFoundError as exc:
        raise EasyOCRBoundaryError("required OCR package is not installed: easyocr") from exc
    return f"easyocr@{package_version}/en"


def _normalize_lines(raw_result: object, image_size: tuple[int, int]) -> tuple[OCRLine, ...]:
    if isinstance(raw_result, (str, bytes)) or not isinstance(raw_result, Sequence):
        raise EasyOCRBoundaryError("EasyOCR result must be a sequence")
    return tuple(_normalize_line(line, image_size) for line in raw_result)


def _normalize_line(raw_line: object, image_size: tuple[int, int]) -> OCRLine:
    if isinstance(raw_line, (str, bytes)) or not isinstance(raw_line, Sequence):
        raise EasyOCRBoundaryError("EasyOCR line must be a sequence")
    if len(raw_line) != 3:
        raise EasyOCRBoundaryError(
            f"EasyOCR line must contain box, text and score, got {len(raw_line)}"
        )
    raw_box, text, score = raw_line
    if not isinstance(text, str):
        raise EasyOCRBoundaryError("EasyOCR text must be a string")
    box = np.asarray(raw_box, dtype=float)
    if box.shape != (4, 2):
        raise EasyOCRBoundaryError(f"EasyOCR box must have shape (4, 2), got {box.shape}")
    width, height = image_size
    try:
        points = (
            _normalize_point(box[0], width, height),
            _normalize_point(box[1], width, height),
            _normalize_point(box[2], width, height),
            _normalize_point(box[3], width, height),
        )
        return OCRLine(box=points, text=text, confidence=_to_float(score, "confidence"))
    except EasyOCRBoundaryError:
        raise
    except (TypeError, ValueError) as exc:
        raise EasyOCRBoundaryError(f"invalid EasyOCR line: {exc}") from exc


def _normalize_point(point: np.ndarray, width: int, height: int) -> OCRPoint:
    return OCRPoint(
        x=min(max(_to_float(point[0], "x coordinate"), 0.0), float(width)),
        y=min(max(_to_float(point[1], "y coordinate"), 0.0), float(height)),
    )


def _to_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float, np.number)):
        raise EasyOCRBoundaryError(f"EasyOCR {label} must be numeric")
    return float(cast(SupportsFloat, value))
