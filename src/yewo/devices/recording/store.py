"""Atomic persistence for checkpointed screen recordings."""

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
from yewo.artifacts.store import AppendOnlyJsonlWriter, AtomicArtifactWriter
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import (
    ScreenRecordingArtifactReference,
    ScreenRecordingBoundary,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingCheckpointReceipt,
    ScreenRecordingRedactionProvenance,
    ScreenRecordingRedactionState,
    ScreenRecordingRemoteCleanupReceipt,
)


class ScreenRecordingStore:
    """Persist immutable recording evidence beneath one artifact root."""

    def __init__(
        self,
        root: Path,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self.root = root
        self._identifier_factory = identifier_factory
        self._writer = AtomicArtifactWriter(root)
        self._events = AppendOnlyJsonlWriter(root)

    def persist_boundary(self, boundary: ScreenRecordingBoundary) -> Path:
        """Persist the context that starts one recording interval."""
        relative_path = self._recording_directory(boundary.recording_id) / "boundary.json"
        self._write_record(relative_path, boundary.model_dump_json(indent=2))
        return relative_path

    def persist_artifact(
        self,
        source: Path,
        *,
        boundary: ScreenRecordingBoundary,
        segment_index: int,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
        capture_method: ScreenRecordingCaptureMethod,
        duration_seconds: float,
        requested_maximum_seconds: int,
        effective_maximum_seconds: int,
        maximum_stop_seconds: float,
        maximum_pull_seconds: float,
        reached_effective_limit: bool,
        redaction_state: ScreenRecordingRedactionState,
        redaction: ScreenRecordingRedactionProvenance | None,
        sensitivity: ArtifactSensitivity,
        retention: ArtifactRetention,
    ) -> ScreenRecordingArtifactReference:
        """Persist one nonempty MP4 segment and return its immutable reference."""
        if not _is_mp4(source):
            raise ScreenRecordingError("A recording artifact must contain an MP4 container.")
        artifact_id = self._identifier_factory()
        relative_path = self._recording_directory(boundary.recording_id) / (
            f"{segment_index:06d}-{checkpoint_kind.value}-{artifact_id}.mp4"
        )
        try:
            reference = ScreenRecordingArtifactReference(
                artifact_id=artifact_id,
                recording_id=boundary.recording_id,
                segment_index=segment_index,
                checkpoint_kind=checkpoint_kind,
                checkpoint_id=checkpoint_id,
                sha256=_file_sha256(source),
                path=relative_path,
                byte_count=source.stat().st_size,
                duration_seconds=duration_seconds,
                requested_maximum_seconds=requested_maximum_seconds,
                effective_maximum_seconds=effective_maximum_seconds,
                maximum_stop_seconds=maximum_stop_seconds,
                maximum_pull_seconds=maximum_pull_seconds,
                reached_effective_limit=reached_effective_limit,
                capture_method=capture_method,
                redaction_state=redaction_state,
                redaction=redaction,
                sensitivity=sensitivity,
                retention=retention,
            )
            self._writer.write_file(relative_path, source)
        except (ArtifactWriteError, ValidationError) as error:
            raise ScreenRecordingError("Could not persist the recording artifact.") from error
        return reference

    def persist_receipt(self, receipt: ScreenRecordingCheckpointReceipt) -> Path:
        """Verify the referenced segment and persist its checkpoint receipt."""
        if receipt.artifact is not None:
            self._verify_artifact(receipt.artifact)
        relative_path = self._recording_directory(receipt.recording_id) / "events.jsonl"
        self._append_event(
            relative_path,
            event_id=receipt.receipt_id,
            event_type="checkpoint",
            record=receipt.model_dump(mode="json"),
        )
        return relative_path

    def persist_remote_cleanup(
        self,
        receipt: ScreenRecordingRemoteCleanupReceipt,
    ) -> Path:
        """Persist append-only evidence that remote recovery is no longer required."""
        relative_path = self._recording_directory(receipt.recording_id) / "events.jsonl"
        self._append_event(
            relative_path,
            event_id=receipt.cleanup_id,
            event_type="remote_cleanup",
            record=receipt.model_dump(mode="json"),
        )
        return relative_path

    def discard(self, reference: ScreenRecordingArtifactReference) -> None:
        """Remove one unchanged segment created by an incomplete checkpoint."""
        artifact_path = self.root / reference.path
        try:
            artifact_path.resolve().relative_to(self.root.resolve())
            if artifact_path.is_symlink():
                raise ScreenRecordingError(
                    f"Refused to remove a changed recording: {reference.artifact_id}"
                )
            if _file_sha256(artifact_path) != reference.sha256:
                raise ScreenRecordingError(
                    f"Refused to remove a changed recording: {reference.artifact_id}"
                )
            artifact_path.unlink()
        except ScreenRecordingError:
            raise
        except (OSError, ValueError) as error:
            raise ScreenRecordingError(
                f"Could not remove an incomplete recording: {reference.artifact_id}"
            ) from error

    def _verify_artifact(self, reference: ScreenRecordingArtifactReference) -> None:
        artifact_path = self.root / reference.path
        try:
            if artifact_path.is_symlink():
                raise ScreenRecordingError("A recording reference cannot be a symbolic link.")
            artifact_path.resolve().relative_to(self.root.resolve())
        except ScreenRecordingError:
            raise
        except (OSError, ValueError) as error:
            raise ScreenRecordingError(
                f"Could not verify recording artifact: {reference.artifact_id}"
            ) from error
        if _file_sha256(artifact_path) != reference.sha256:
            raise ScreenRecordingError(
                f"Recording artifact digest mismatch: {reference.artifact_id}"
            )
        if artifact_path.stat().st_size != reference.byte_count:
            raise ScreenRecordingError(f"Recording artifact size mismatch: {reference.artifact_id}")

    def _write_record(self, relative_path: Path, content: str) -> None:
        try:
            self._writer.write(relative_path, f"{content}\n".encode())
        except ArtifactWriteError as error:
            raise ScreenRecordingError("Could not persist the recording record.") from error

    def _append_event(
        self,
        relative_path: Path,
        *,
        event_id: str,
        event_type: str,
        record: dict[str, object],
    ) -> None:
        try:
            self._events.append(
                relative_path,
                {**record, "event_id": event_id, "event_type": event_type},
                identity_field="event_id",
            )
        except ArtifactWriteError as error:
            raise ScreenRecordingError("Could not persist the recording record.") from error

    @staticmethod
    def _recording_directory(recording_id: str) -> Path:
        return ARTIFACTS_DIRECTORY / "recordings" / recording_id


def _is_mp4(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[4:8] == b"ftyp"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ScreenRecordingError("Could not read the recording artifact source.") from error
    return digest.hexdigest()
