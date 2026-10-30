"""Engine-independent OCR observations in screenshot coordinates."""

from __future__ import annotations

import math
from typing import Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OCRPoint(BaseModel):
    """One point in the source screenshot coordinate system."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)

    @field_validator("x", "y")
    @classmethod
    def _validate_finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("OCR coordinates must be finite")
        return value


class OCRLine(BaseModel):
    """One recognized text line and its four-point source-image box."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    box: tuple[OCRPoint, OCRPoint, OCRPoint, OCRPoint]
    text: str
    confidence: float = Field(ge=0, le=1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The axis-aligned box around this line's four points."""
        xs = [point.x for point in self.box]
        ys = [point.y for point in self.box]
        return min(xs), min(ys), max(xs), max(ys)

    @field_validator("confidence")
    @classmethod
    def _validate_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("OCR confidence must be finite")
        return value


class OCRTiming(BaseModel):
    """OCR engine and wall-clock durations in milliseconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_ms: float | None = Field(default=None, ge=0)
    classification_ms: float | None = Field(default=None, ge=0)
    recognition_ms: float | None = Field(default=None, ge=0)
    engine_total_ms: float = Field(ge=0)
    wall_ms: float = Field(ge=0)

    @field_validator(
        "detection_ms", "classification_ms", "recognition_ms", "engine_total_ms", "wall_ms"
    )
    @classmethod
    def _validate_finite_duration(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value):
            raise ValueError("OCR durations must be finite")
        return value


class OCRResult(BaseModel):
    """One engine's independent OCR reading of a complete screenshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_identity: str = Field(min_length=1)
    lines: tuple[OCRLine, ...]
    timing: OCRTiming


class OCRReader(Protocol):
    """Boundary implemented by a whole-screen OCR engine."""

    @property
    def identity(self) -> str: ...

    def recognize(self, image: Image.Image) -> OCRResult: ...
