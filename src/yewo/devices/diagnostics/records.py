"""Platform-neutral application failure vocabulary."""

from collections.abc import Iterable
from enum import StrEnum


class ApplicationFailureKind(StrEnum):
    """A failure or termination established by platform evidence."""

    CRASH = "crash"
    ANR = "anr"
    OTHER_TERMINATION = "other_termination"
    UNCLASSIFIED = "unclassified"


class ApplicationDiagnosticClassification(StrEnum):
    """The diagnostic conclusion for one source at one checkpoint."""

    NO_APPLICATION_FAILURE_OBSERVED = "no_application_failure_observed"
    APPLICATION_CRASH = "application_crash"
    APPLICATION_ANR = "application_anr"
    MULTIPLE_APPLICATION_FAILURES = "multiple_application_failures"
    OTHER_APPLICATION_TERMINATION = "other_application_termination"
    UNCLASSIFIED_APPLICATION_TERMINATION = "unclassified_application_termination"
    DIAGNOSTICS_UNAVAILABLE = "diagnostics_unavailable"
    DIAGNOSTICS_INCOMPLETE = "diagnostics_incomplete"
    YEWO_INFRASTRUCTURE_FAILURE = "yewo_infrastructure_failure"


def classify_application_failures(
    failure_kinds: Iterable[ApplicationFailureKind],
) -> ApplicationDiagnosticClassification:
    """Return one deterministic conclusion without hiding mixed failures."""
    kinds = frozenset(failure_kinds)
    if not kinds:
        return ApplicationDiagnosticClassification.NO_APPLICATION_FAILURE_OBSERVED
    if ApplicationFailureKind.CRASH in kinds and ApplicationFailureKind.ANR in kinds:
        return ApplicationDiagnosticClassification.MULTIPLE_APPLICATION_FAILURES
    if ApplicationFailureKind.ANR in kinds:
        return ApplicationDiagnosticClassification.APPLICATION_ANR
    if ApplicationFailureKind.CRASH in kinds:
        return ApplicationDiagnosticClassification.APPLICATION_CRASH
    if ApplicationFailureKind.OTHER_TERMINATION in kinds:
        return ApplicationDiagnosticClassification.OTHER_APPLICATION_TERMINATION
    return ApplicationDiagnosticClassification.UNCLASSIFIED_APPLICATION_TERMINATION
