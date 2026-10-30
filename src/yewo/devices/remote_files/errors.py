"""Failures raised by remote temporary-file ownership."""


class RemoteTemporaryFileError(RuntimeError):
    """A remote temporary-file operation failed."""


class RemoteTemporaryFileCleanupError(RemoteTemporaryFileError):
    """One or more run-owned remote files remain on the device."""

    def __init__(self, failed_paths: tuple[str, ...]) -> None:
        self.failed_paths = failed_paths
        super().__init__(
            f"Could not remove {len(failed_paths)} run-owned remote temporary file(s)."
        )
