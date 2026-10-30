"""Atomic persistence for bounded device-log evidence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
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
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.records import (
    DeviceLogArtifactKind,
    DeviceLogArtifactReference,
    DeviceLogBoundary,
    DeviceLogDrainReceipt,
    DeviceLogSource,
)
from yewo.devices.diagnostics import ApplicationFailureKind


class DeviceLogStore:
    """Persist immutable device-log evidence beneath one artifact root."""

    def __init__(
        self,
        root: Path,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self.root = root
        self._identifier_factory = identifier_factory
        self._writer = AtomicArtifactWriter(root)
        self._ledger = AppendOnlyJsonlWriter(root)

    def persist_artifact(
        self,
        content: bytes,
        *,
        boundary_id: str,
        drain_id: str,
        drain_index: int,
        source: DeviceLogSource,
        entry_count: int,
        kind: DeviceLogArtifactKind,
        source_name: str | None,
        redaction_policy_id: str,
        source_redaction_count: int,
        sensitivity: ArtifactSensitivity,
        retention: ArtifactRetention,
        failure_kind: ApplicationFailureKind | None = None,
        occurred_at: datetime | None = None,
        platform_reason: str | None = None,
        platform_code: str | None = None,
    ) -> DeviceLogArtifactReference:
        """Persist one nonempty, redacted device-log artifact."""
        if not content or entry_count < 1:
            raise DeviceLogCaptureError("A device-log artifact cannot be empty.")
        artifact_id = self._identifier_factory()
        suffix = _artifact_suffix(kind, source_name)
        relative_path = (
            ARTIFACTS_DIRECTORY
            / "device-logs"
            / boundary_id
            / (f"{drain_index:06d}-{drain_id}-{source.artifact_stem}-{artifact_id}{suffix}")
        )
        try:
            reference = DeviceLogArtifactReference(
                artifact_id=artifact_id,
                boundary_id=boundary_id,
                drain_id=drain_id,
                log_type=source.log_type,
                kind=kind,
                source_name=source_name,
                failure_kind=failure_kind,
                occurred_at=occurred_at,
                platform_reason=platform_reason,
                platform_code=platform_code,
                sha256=sha256(content).hexdigest(),
                path=relative_path,
                byte_count=len(content),
                entry_count=entry_count,
                redaction_policy_id=redaction_policy_id,
                source_redaction_count=source_redaction_count,
                sensitivity=sensitivity,
                retention=retention,
            )
            self._writer.write(relative_path, content)
        except (ArtifactWriteError, ValidationError) as error:
            raise DeviceLogCaptureError("Could not persist the device-log artifact.") from error
        return reference

    def discard(self, reference: DeviceLogArtifactReference) -> None:
        """Remove one unchanged artifact created by an incomplete drain."""
        artifact_path = self.root / reference.path
        try:
            artifact_path.resolve().relative_to(self.root.resolve())
            if artifact_path.is_symlink():
                raise DeviceLogCaptureError(
                    f"Refused to remove a changed device-log artifact: {reference.artifact_id}"
                )
            content = artifact_path.read_bytes()
            if sha256(content).hexdigest() != reference.sha256:
                raise DeviceLogCaptureError(
                    f"Refused to remove a changed device-log artifact: {reference.artifact_id}"
                )
            artifact_path.unlink()
            self._remove_empty_boundary_directory(reference.boundary_id)
        except DeviceLogCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise DeviceLogCaptureError(
                f"Could not remove an incomplete device-log artifact: {reference.artifact_id}"
            ) from error

    def persist_boundary(self, boundary: DeviceLogBoundary) -> Path:
        """Persist the context and source state that start one log interval."""
        relative_path = ARTIFACTS_DIRECTORY / "device-logs" / boundary.boundary_id / "boundary.json"
        self._write_record(relative_path, boundary.model_dump_json(indent=2))
        return relative_path

    def discard_boundary(self, boundary: DeviceLogBoundary) -> None:
        """Remove one unchanged boundary whose startup receipt did not persist."""
        relative_path = ARTIFACTS_DIRECTORY / "device-logs" / boundary.boundary_id / "boundary.json"
        expected_content = f"{boundary.model_dump_json(indent=2)}\n".encode()
        boundary_path = self.root / relative_path
        try:
            boundary_path.resolve().relative_to(self.root.resolve())
            if boundary_path.is_symlink() or boundary_path.read_bytes() != expected_content:
                raise DeviceLogCaptureError(
                    f"Refused to remove a changed log boundary: {boundary.boundary_id}"
                )
            boundary_path.unlink()
            self._remove_empty_boundary_directory(boundary.boundary_id)
        except DeviceLogCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise DeviceLogCaptureError(
                f"Could not remove an incomplete log boundary: {boundary.boundary_id}"
            ) from error

    def persist_receipt(self, receipt: DeviceLogDrainReceipt) -> Path:
        """Verify referenced segments and persist one drain receipt."""
        for source_receipt in receipt.sources:
            for artifact in source_receipt.artifacts:
                self._verify_artifact(artifact)
        relative_path = (
            ARTIFACTS_DIRECTORY / "device-logs" / receipt.boundary_id / "drains.jsonl"
        )
        try:
            self._ledger.append(
                relative_path,
                receipt.model_dump(mode="json"),
                identity_field="drain_id",
            )
        except ArtifactWriteError as error:
            raise DeviceLogCaptureError("Could not persist the device-log record.") from error
        return relative_path

    def _verify_artifact(self, reference: DeviceLogArtifactReference) -> None:
        artifact_path = self.root / reference.path
        try:
            if artifact_path.is_symlink():
                raise DeviceLogCaptureError(
                    "A device-log artifact reference cannot be a symbolic link."
                )
            artifact_path.resolve().relative_to(self.root.resolve())
            content = artifact_path.read_bytes()
        except DeviceLogCaptureError:
            raise
        except (OSError, ValueError) as error:
            raise DeviceLogCaptureError(
                f"Could not verify device-log artifact: {reference.artifact_id}"
            ) from error
        if sha256(content).hexdigest() != reference.sha256:
            raise DeviceLogCaptureError(
                f"Device-log artifact digest mismatch: {reference.artifact_id}"
            )
        if len(content) != reference.byte_count:
            raise DeviceLogCaptureError(
                f"Device-log artifact size mismatch: {reference.artifact_id}"
            )

    def _write_record(self, relative_path: Path, json_content: str) -> None:
        try:
            self._writer.write(relative_path, f"{json_content}\n".encode())
        except ArtifactWriteError as error:
            raise DeviceLogCaptureError("Could not persist the device-log record.") from error

    def _remove_empty_boundary_directory(self, boundary_id: str) -> None:
        boundary_path = self.root / ARTIFACTS_DIRECTORY / "device-logs" / boundary_id
        try:
            boundary_path.rmdir()
        except OSError:
            return


def _artifact_suffix(kind: DeviceLogArtifactKind, source_name: str | None) -> str:
    if kind is DeviceLogArtifactKind.TEXT_STREAM:
        return ".log"
    suffix = Path(source_name or "").suffix.casefold()
    return suffix if suffix in {".ips", ".json", ".log", ".txt"} else ".log"
