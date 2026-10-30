"""Application identity contracts."""

from yewo.applications.errors import ApplicationError, ApplicationInspectionError
from yewo.applications.inspection import ApplicationArtifactInspector
from yewo.applications.presence import ApplicationPresence
from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallationChange,
    ApplicationInstallMode,
    ApplicationRequest,
)

__all__ = [
    "ApplicationDescriptor",
    "ApplicationError",
    "ApplicationArtifactInspector",
    "ApplicationInstallMode",
    "ApplicationInstallationChange",
    "ApplicationInspectionError",
    "ApplicationPresence",
    "ApplicationRequest",
]
