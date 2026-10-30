"""Own exact remote temporary-file paths for one automation session."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from pydantic import ValidationError

from yewo.applications.records import ApplicationDescriptor
from yewo.artifacts.records import new_record_identifier, validate_record_identifier
from yewo.devices.inventory import DeviceDescriptor
from yewo.devices.remote_files.appium import RemoteFileRemover
from yewo.devices.remote_files.errors import (
    RemoteTemporaryFileCleanupError,
    RemoteTemporaryFileError,
)
from yewo.devices.remote_files.records import (
    RemoteTemporaryFileReference,
    remote_temporary_path,
)


class RemoteTemporaryFileLease:
    """Reserve and clean only paths owned by one automation session."""

    def __init__(
        self,
        remover: RemoteFileRemover,
        device: DeviceDescriptor,
        application: ApplicationDescriptor,
        namespace_id: str,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        if device.platform is not application.platform:
            raise ValueError("The remote-file device and application must share a platform.")
        try:
            self.namespace_id = validate_record_identifier(namespace_id)
        except ValidationError as error:
            raise RemoteTemporaryFileError(
                "The remote temporary-file namespace is invalid."
            ) from error
        self._remover = remover
        self._device = device
        self._application = application
        self._identifier_factory = identifier_factory
        self._references: dict[str, RemoteTemporaryFileReference] = {}
        self._name_keys: set[str] = set()
        self._closed = False
        self._close_attempted = False

    @classmethod
    def open(
        cls,
        remover: RemoteFileRemover,
        device: DeviceDescriptor,
        application: ApplicationDescriptor,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> Self:
        """Create one unique remote temporary-file namespace."""
        try:
            namespace_id = validate_record_identifier(identifier_factory())
        except ValidationError as error:
            raise RemoteTemporaryFileError(
                "The remote temporary-file namespace is invalid."
            ) from error
        return cls(
            remover,
            device,
            application,
            namespace_id,
            identifier_factory=identifier_factory,
        )

    @property
    def closed(self) -> bool:
        """Return whether final cleanup was attempted."""
        return self._closed

    @property
    def close_attempted(self) -> bool:
        """Return whether cleanup has run at least once."""
        return self._close_attempted

    @property
    def cleanup_complete(self) -> bool:
        """Return whether every cleanup obligation was released."""
        return not self._references

    @property
    def references(self) -> tuple[RemoteTemporaryFileReference, ...]:
        """Return paths that still have an active cleanup obligation."""
        return tuple(self._references.values())

    def owns(self, reference: RemoteTemporaryFileReference) -> bool:
        """Return whether this lease owns the exact reference."""
        return self._references.get(reference.file_id) == reference

    def reserve(self, name: str) -> RemoteTemporaryFileReference:
        """Register cleanup before a caller starts writing a remote file."""
        if self._closed:
            raise RemoteTemporaryFileError(
                "A closed remote temporary-file lease cannot reserve a path."
            )
        name_key = name.casefold()
        if name_key in self._name_keys:
            raise RemoteTemporaryFileError(
                "A remote temporary-file name is already reserved in this session."
            )
        try:
            file_id = validate_record_identifier(self._identifier_factory())
            reference = RemoteTemporaryFileReference(
                file_id=file_id,
                namespace_id=self.namespace_id,
                platform=self._device.platform,
                target_id=self._device.target_id,
                application_id=self._application.application_id,
                name=name,
                remote_path=remote_temporary_path(
                    self._device.platform,
                    self._application.application_id,
                    self.namespace_id,
                    name,
                ),
            )
        except (ValidationError, ValueError) as error:
            raise RemoteTemporaryFileError(
                "The remote temporary-file reservation is invalid."
            ) from error
        if file_id in self._references:
            raise RemoteTemporaryFileError(
                "A remote temporary-file identifier is already reserved."
            )
        self._references[file_id] = reference
        self._name_keys.add(name_key)
        return reference

    def release(self, reference: RemoteTemporaryFileReference) -> bool:
        """Remove one reserved file now and clear its cleanup obligation."""
        tracked = self._references.get(reference.file_id)
        if tracked is None:
            return False
        if tracked != reference:
            raise RemoteTemporaryFileError(
                "The remote temporary-file reference does not match its reservation."
            )
        try:
            self._remover.remove_file(reference.remote_path)
        except RemoteTemporaryFileError:
            raise
        except Exception as error:  # noqa: BLE001 - injected platform boundary
            raise RemoteTemporaryFileError("The remote temporary-file remover failed.") from error
        del self._references[reference.file_id]
        self._name_keys.remove(reference.name.casefold())
        return True

    def close(self) -> None:
        """Attempt every outstanding cleanup in reverse reservation order."""
        if self._closed and not self._references:
            return
        self._close_attempted = True
        self._closed = True
        failed_paths: list[str] = []
        failure_errors: list[RemoteTemporaryFileError] = []
        for reference in reversed(self.references):
            try:
                self.release(reference)
            except RemoteTemporaryFileError as error:
                failed_paths.append(reference.remote_path)
                failure_errors.append(error)
        if failure_errors:
            raise RemoteTemporaryFileCleanupError(tuple(failed_paths)) from failure_errors[0]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()
