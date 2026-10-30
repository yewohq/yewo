"""Failures raised by shared artifact persistence."""


class ArtifactWriteError(RuntimeError):
    """An immutable artifact could not be persisted safely."""
