"""Failures raised by screen-recording boundaries."""


class ScreenRecordingError(RuntimeError):
    """A recorder, checkpoint, or recording artifact operation failed."""


class ScreenRecordingRollbackError(ScreenRecordingError):
    """Incomplete recording evidence could not be removed safely."""
