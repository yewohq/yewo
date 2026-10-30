"""Owned and integrity-checked media push and pull."""

from hashlib import sha256

from yewo.devices.media.errors import (
    MediaTransferError,
    MediaTransferRollbackError,
)
from yewo.devices.media.policy import MediaTransferPolicy
from yewo.devices.media.records import RemoteMediaReference
from yewo.devices.media.transport import MediaTransport
from yewo.devices.remote_files import (
    RemoteTemporaryFileError,
    RemoteTemporaryFileLease,
)


class MediaTransferService:
    """Transfer bounded media through run-owned remote paths."""

    def __init__(
        self,
        transport: MediaTransport,
        remote_files: RemoteTemporaryFileLease,
        *,
        policy: MediaTransferPolicy | None = None,
    ) -> None:
        self._transport = transport
        self._remote_files = remote_files
        self._policy = policy or MediaTransferPolicy()

    def push(self, name: str, content: bytes) -> RemoteMediaReference:
        """Push bytes after registering their remote cleanup obligation."""
        self._validate_content(content)
        remote_file = self._remote_files.reserve(name)
        try:
            self._transport.push(remote_file.remote_path, content)
        except Exception as error:  # noqa: BLE001 - injected transport boundary
            transfer_error = (
                error
                if isinstance(error, MediaTransferError)
                else MediaTransferError("The media transport failed during push.")
            )
            try:
                self._remote_files.release(remote_file)
            except RemoteTemporaryFileError as cleanup_error:
                rollback_error = MediaTransferRollbackError(
                    "A failed media push left an uncertain remote file."
                )
                rollback_error.add_note(f"Remote cleanup also failed: {cleanup_error}")
                raise rollback_error from transfer_error
            if transfer_error is error:
                raise transfer_error from None
            raise transfer_error from error
        return RemoteMediaReference(
            remote_file=remote_file,
            sha256=sha256(content).hexdigest(),
            byte_count=len(content),
        )

    def pull(self, reference: RemoteMediaReference) -> bytes:
        """Pull owned media and match it to its original content identity."""
        if not self._remote_files.owns(reference.remote_file):
            raise MediaTransferError("The media reference is not owned by this session.")
        try:
            content = self._transport.pull(
                reference.remote_file.remote_path,
                self._policy.maximum_bytes,
            )
        except MediaTransferError:
            raise
        except Exception as error:  # noqa: BLE001 - injected transport boundary
            raise MediaTransferError("The media transport failed during pull.") from error
        self._validate_content(content)
        if len(content) != reference.byte_count or sha256(content).hexdigest() != reference.sha256:
            raise MediaTransferError("The pulled media does not match its push identity.")
        return content

    def release(self, reference: RemoteMediaReference) -> bool:
        """Remove pushed media before the session cleanup boundary."""
        try:
            return self._remote_files.release(reference.remote_file)
        except RemoteTemporaryFileError as error:
            raise MediaTransferError("The pushed media could not be removed.") from error

    def _validate_content(self, content: bytes) -> None:
        if not content:
            raise MediaTransferError("Media content must not be empty.")
        if len(content) > self._policy.maximum_bytes:
            raise MediaTransferError("Media content exceeds the configured byte limit.")
