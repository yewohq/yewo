"""Cross-platform application failure classification."""

from yewo.devices.diagnostics.records import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
    classify_application_failures,
)

__all__ = [
    "ApplicationDiagnosticClassification",
    "ApplicationFailureKind",
    "classify_application_failures",
]
