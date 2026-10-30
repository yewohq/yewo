"""Failures raised by bounded media transfer."""


class MediaTransferError(RuntimeError):
    """A media push, pull, or integrity check failed."""


class MediaTransferRollbackError(MediaTransferError):
    """A failed media push also left an uncertain remote file."""
