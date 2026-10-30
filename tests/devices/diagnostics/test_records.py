"""Tests for platform-neutral diagnostic conclusions."""

import pytest

from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
    classify_application_failures,
)


@pytest.mark.parametrize(
    ("failure_kinds", "expected"),
    (
        ((), ApplicationDiagnosticClassification.NO_APPLICATION_FAILURE_OBSERVED),
        ((ApplicationFailureKind.CRASH,), ApplicationDiagnosticClassification.APPLICATION_CRASH),
        ((ApplicationFailureKind.ANR,), ApplicationDiagnosticClassification.APPLICATION_ANR),
        (
            (ApplicationFailureKind.CRASH, ApplicationFailureKind.ANR),
            ApplicationDiagnosticClassification.MULTIPLE_APPLICATION_FAILURES,
        ),
        (
            (ApplicationFailureKind.OTHER_TERMINATION,),
            ApplicationDiagnosticClassification.OTHER_APPLICATION_TERMINATION,
        ),
        (
            (ApplicationFailureKind.UNCLASSIFIED,),
            ApplicationDiagnosticClassification.UNCLASSIFIED_APPLICATION_TERMINATION,
        ),
    ),
)
def test_failure_classification_preserves_distinct_outcomes(
    failure_kinds: tuple[ApplicationFailureKind, ...],
    expected: ApplicationDiagnosticClassification,
) -> None:
    assert classify_application_failures(failure_kinds) is expected
