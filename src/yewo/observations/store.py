"""Atomic persistence for screenshot evidence."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.paths import ARTIFACTS_DIRECTORY
from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    new_record_identifier,
)
from yewo.artifacts.store import AtomicArtifactWriter
from yewo.devices.screen import png_size
from yewo.observations.errors import ObservationCaptureError
from yewo.observations.records import (
    Observation,
    RedactionProvenance,
    ScreenshotRedactionState,
    ScreenshotReference,
)


class ScreenshotStore:
    """Persist PNG bytes under one isolated artifact root."""

    def __init__(
        self,
        root: Path,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self.root = root
        self._identifier_factory = identifier_factory
        self._writer = AtomicArtifactWriter(root)

    def persist_raw(
        self,
        png: bytes,
        *,
        sensitivity: ArtifactSensitivity,
        retention: ArtifactRetention,
    ) -> ScreenshotReference:
        """Persist unredacted PNG bytes and return their verified reference."""
        return self._persist(
            png,
            redaction_state=ScreenshotRedactionState.RAW,
            sensitivity=sensitivity,
            retention=retention,
        )

    def persist_redacted(
        self,
        png: bytes,
        *,
        sensitivity: ArtifactSensitivity,
        retention: ArtifactRetention,
        redaction: RedactionProvenance,
    ) -> ScreenshotReference:
        """Persist approved redacted PNG bytes and return their reference."""
        return self._persist(
            png,
            redaction_state=ScreenshotRedactionState.REDACTED,
            sensitivity=sensitivity,
            retention=retention,
            redaction=redaction,
        )

    def discard(self, reference: ScreenshotReference) -> None:
        """Remove an unchanged screenshot created by an incomplete capture."""
        path = self.root / reference.path
        try:
            path.resolve().relative_to(self.root.resolve())
            if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != reference.sha256:
                raise ObservationCaptureError(
                    f"Refused to remove changed screenshot: {reference.artifact_id}"
                )
            path.unlink()
        except ObservationCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise ObservationCaptureError(
                f"Could not remove incomplete screenshot: {reference.artifact_id}"
            ) from error

    def _persist(
        self,
        png: bytes,
        *,
        redaction_state: ScreenshotRedactionState,
        sensitivity: ArtifactSensitivity,
        retention: ArtifactRetention,
        redaction: RedactionProvenance | None = None,
    ) -> ScreenshotReference:
        dimensions = png_size(png)
        if dimensions is None:
            raise ObservationCaptureError("The screenshot store received invalid PNG data.")
        artifact_id = self._identifier_factory()
        relative_path = ARTIFACTS_DIRECTORY / "screenshots" / f"{artifact_id}.png"
        width, height = dimensions
        try:
            reference = ScreenshotReference(
                artifact_id=artifact_id,
                sha256=sha256(png).hexdigest(),
                width_px=width,
                height_px=height,
                path=relative_path,
                redaction_state=redaction_state,
                redaction=redaction,
                sensitivity=sensitivity,
                retention=retention,
            )
        except ValidationError as error:
            raise ObservationCaptureError(
                "The screenshot identifier or metadata is invalid."
            ) from error
        try:
            self._writer.write(relative_path, png)
        except ArtifactWriteError as error:
            raise ObservationCaptureError(str(error)) from error
        return reference


class ObservationStore:
    """Persist complete observation records under one artifact root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._writer = AtomicArtifactWriter(root)

    def persist(self, observation: Observation) -> Path:
        """Persist one canonical JSON observation without overwriting evidence."""
        for screenshot in (
            observation.raw_screenshot,
            observation.redacted_screenshot,
        ):
            if screenshot is not None:
                self._verify_screenshot(screenshot)
        relative_path = (
            ARTIFACTS_DIRECTORY / "observations" / f"{observation.observation_id}.json"
        )
        content = f"{observation.model_dump_json(indent=2)}\n".encode("utf-8")
        try:
            self._writer.write(relative_path, content)
        except ArtifactWriteError as error:
            raise ObservationCaptureError(str(error)) from error
        return relative_path

    def _verify_screenshot(self, reference: ScreenshotReference) -> None:
        screenshot_path = self.root / reference.path
        try:
            if screenshot_path.is_symlink():
                raise ObservationCaptureError(
                    "An observation screenshot reference cannot be a symbolic link."
                )
            screenshot_path.resolve().relative_to(self.root.resolve())
            content = screenshot_path.read_bytes()
        except ObservationCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise ObservationCaptureError(
                f"Could not verify observation screenshot: {reference.artifact_id}"
            ) from error
        if sha256(content).hexdigest() != reference.sha256:
            raise ObservationCaptureError(
                f"Observation screenshot digest mismatch: {reference.artifact_id}"
            )
        if png_size(content) != (reference.width_px, reference.height_px):
            raise ObservationCaptureError(
                f"Observation screenshot dimensions mismatch: {reference.artifact_id}"
            )
