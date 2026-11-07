"""Failures raised by canonical report persistence."""


class CanonicalRunRecordWriteError(RuntimeError):
    """The canonical run record could not be persisted."""


class ReportBundleWriteError(RuntimeError):
    """One or more derived report outputs could not be persisted."""
