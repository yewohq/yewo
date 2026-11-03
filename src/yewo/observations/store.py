"""Atomic persistence for screenshot evidence."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path

from pydantic import ValidationError

from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.paths import ARTIFACTS_DIRECTORY
from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    new_record_identifier,
)
from yewo.artifacts.store import AppendOnlyJsonlWriter, AtomicArtifactWriter
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

    def load(self, reference: ScreenshotReference) -> bytes:
        """Load screenshot bytes after verifying path, digest, and dimensions."""
        path = self.root / reference.path
        try:
            path.resolve().relative_to(self.root.resolve())
            if path.is_symlink() or not path.is_file():
                raise ObservationCaptureError("A screenshot must be a regular file.")
            content = path.read_bytes()
        except ObservationCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise ObservationCaptureError(
                f"Could not load screenshot: {reference.artifact_id}"
            ) from error
        if sha256(content).hexdigest() != reference.sha256:
            raise ObservationCaptureError(
                f"Screenshot digest mismatch: {reference.artifact_id}"
            )
        if png_size(content) != (reference.width_px, reference.height_px):
            raise ObservationCaptureError(
                f"Screenshot dimensions mismatch: {reference.artifact_id}"
            )
        return content

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
        self._writer = AppendOnlyJsonlWriter(root)

    def persist(self, observation: Observation) -> Path:
        """Append one canonical observation without overwriting evidence."""
        for screenshot in (
            observation.raw_screenshot,
            observation.redacted_screenshot,
        ):
            if screenshot is not None:
                self._verify_screenshot(screenshot)
        relative_path = ARTIFACTS_DIRECTORY / "observations.jsonl"
        try:
            self._writer.append(
                relative_path,
                observation.model_dump(mode="json"),
                identity_field="observation_id",
            )
        except ArtifactWriteError as error:
            raise ObservationCaptureError(str(error)) from error
        return relative_path

    def load_all(self) -> tuple[Observation, ...]:
        """Load and verify every persisted observation in ledger order."""
        relative_path = ARTIFACTS_DIRECTORY / "observations.jsonl"
        path = self.root / relative_path
        if not path.exists():
            return ()
        observations: list[Observation] = []
        observation_ids: set[str] = set()
        try:
            if path.is_symlink() or not path.is_file():
                raise ObservationCaptureError("The observation ledger must be a regular file.")
            content = path.read_bytes()
            if content and not content.endswith(b"\n"):
                raise ObservationCaptureError(
                    "The observation ledger has an incomplete final line."
                )
            for line_number, line in enumerate(content.splitlines(), start=1):
                if not line:
                    raise ObservationCaptureError(
                        f"The observation ledger has an empty line: {line_number}."
                    )
                observation = Observation.model_validate_json(line)
                if observation.observation_id in observation_ids:
                    raise ObservationCaptureError(
                        "The observation ledger repeats an observation identifier."
                    )
                for screenshot in (
                    observation.raw_screenshot,
                    observation.redacted_screenshot,
                ):
                    if screenshot is not None:
                        self._verify_screenshot(screenshot)
                observations.append(observation)
                observation_ids.add(observation.observation_id)
        except ObservationCaptureError:
            raise
        except (OSError, ValidationError) as error:
            raise ObservationCaptureError("The observation ledger is invalid.") from error
        return tuple(observations)

    def load_selected(self, observation_ids: set[str]) -> tuple[Observation, ...]:
        """Load and verify only named observations from the append-only ledger."""
        if not observation_ids:
            return ()
        relative_path = ARTIFACTS_DIRECTORY / "observations.jsonl"
        path = self.root / relative_path
        selected: list[Observation] = []
        found: set[str] = set()
        try:
            if path.is_symlink() or not path.is_file():
                raise ObservationCaptureError("The observation ledger must be a regular file.")
            content = path.read_bytes()
            if content and not content.endswith(b"\n"):
                raise ObservationCaptureError(
                    "The observation ledger has an incomplete final line."
                )
            for line_number, line in enumerate(content.splitlines(), start=1):
                if not line:
                    raise ObservationCaptureError(
                        f"The observation ledger has an empty line: {line_number}."
                    )
                payload = json.loads(line)
                observation_id = payload.get("observation_id")
                if observation_id not in observation_ids:
                    continue
                if observation_id in found:
                    raise ObservationCaptureError(
                        "The observation ledger repeats a requested observation identifier."
                    )
                observation = Observation.model_validate(payload)
                for screenshot in (
                    observation.raw_screenshot,
                    observation.redacted_screenshot,
                ):
                    if screenshot is not None:
                        self._verify_screenshot(screenshot)
                selected.append(observation)
                found.add(observation.observation_id)
        except ObservationCaptureError:
            raise
        except (OSError, TypeError, json.JSONDecodeError, ValidationError) as error:
            raise ObservationCaptureError("The observation ledger is invalid.") from error
        missing = observation_ids - found
        if missing:
            raise ObservationCaptureError(
                "The observation ledger is missing requested observations: "
                + ", ".join(sorted(missing))
            )
        return tuple(selected)

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
