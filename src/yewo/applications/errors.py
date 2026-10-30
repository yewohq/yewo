"""Failures reported while identifying an application artifact."""


class ApplicationError(RuntimeError):
    """An application operation could not complete."""


class ApplicationInspectionError(ApplicationError):
    """An artifact did not provide trustworthy application identity."""
