"""Resolve caller intent into one inspected application identity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from yewo.applications.errors import ApplicationInspectionError
from yewo.applications.inspection import ApplicationArtifactInspector
from yewo.applications.records import ApplicationDescriptor, ApplicationRequest
from yewo.devices import DevicePlatform


class ApplicationResolver(Protocol):
    """Resolve an application request without installing the application."""

    def resolve(self, request: ApplicationRequest) -> ApplicationDescriptor:
        """Return trusted identity for an artifact or requested installed app."""
        ...


class RequestedApplicationResolver:
    """Inspect artifacts and preserve explicit reuse identity."""

    def __init__(
        self,
        inspectors: Mapping[DevicePlatform, ApplicationArtifactInspector],
    ) -> None:
        self._inspectors = inspectors

    def resolve(self, request: ApplicationRequest) -> ApplicationDescriptor:
        """Resolve one request without changing device state."""
        artifact_path = request.artifact_path
        if artifact_path is None:
            if request.application_id is None:
                raise ApplicationInspectionError(
                    "An application request has no artifact or application ID."
                )
            return ApplicationDescriptor(
                platform=request.platform,
                application_id=request.application_id,
            )
        inspector = self._inspectors.get(request.platform)
        if inspector is None:
            raise ApplicationInspectionError(
                f"No artifact inspector is configured for {request.platform.value}."
            )
        descriptor = inspector.inspect(artifact_path)
        if descriptor.platform is not request.platform:
            raise ApplicationInspectionError(
                "The inspected application platform does not match the request."
            )
        if (
            request.application_id is not None
            and descriptor.application_id != request.application_id
        ):
            raise ApplicationInspectionError(
                "The inspected application ID does not match the requested ID."
            )
        return descriptor
