"""Describe a run failure and the evidence it left behind."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from yewo.artifacts import ARTIFACTS_DIRECTORY
from yewo.host import LOCK_FILE_SUFFIX


_MAXIMUM_CHAINED_ERRORS = 5


def error_chain(error: BaseException) -> str:
    """Render the chained error type names without their messages."""
    names: list[str] = []
    current: BaseException | None = error
    while current is not None and len(names) < _MAXIMUM_CHAINED_ERRORS:
        names.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return "<-".join(names)


def validation_detail(error: BaseException) -> str:
    """Render validation failures without copying validated inputs into the trace."""
    if not isinstance(error, ValidationError):
        return ""
    failures = (
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)
    )
    return f" detail={' | '.join(failures)}"


def has_partial_evidence(root: Path) -> bool:
    """Return whether a run left any evidence file beneath its artifact root."""
    assets = root / ARTIFACTS_DIRECTORY
    try:
        return assets.is_dir() and any(
            path.is_file() and path.suffix != LOCK_FILE_SUFFIX for path in assets.rglob("*")
        )
    except OSError:
        return False
