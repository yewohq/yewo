"""Evidence records and capture services."""

from yewo.observations.errors import ObservationCaptureError
from yewo.observations.fingerprint import (
    build_screen_fingerprint,
    fingerprint_distance,
)
from yewo.observations.records import (
    CoordinateScale,
    DeviceOrientation,
    Observation,
    PresentationKey,
    PixelRegion,
    RedactionProvenance,
    RedactionRegion,
    ScreenRegion,
    ScreenFingerprint,
    ScreenshotRedactionState,
    ScreenshotReference,
)
from yewo.observations.matching import ScreenComparisonPolicy, ScreenMatcher
from yewo.observations.redaction import (
    RedactionResult,
    RegionScreenshotRedactor,
    ScreenshotRedactor,
)
from yewo.observations.service import ObservationCapturePolicy, ObservationService
from yewo.observations.store import ObservationStore, ScreenshotStore

__all__ = [
    "CoordinateScale",
    "DeviceOrientation",
    "Observation",
    "ObservationCapturePolicy",
    "ObservationCaptureError",
    "ObservationService",
    "ObservationStore",
    "PresentationKey",
    "PixelRegion",
    "RedactionProvenance",
    "RedactionRegion",
    "RedactionResult",
    "RegionScreenshotRedactor",
    "ScreenRegion",
    "ScreenComparisonPolicy",
    "ScreenFingerprint",
    "ScreenMatcher",
    "ScreenshotRedactor",
    "ScreenshotRedactionState",
    "ScreenshotReference",
    "ScreenshotStore",
    "build_screen_fingerprint",
    "fingerprint_distance",
]
