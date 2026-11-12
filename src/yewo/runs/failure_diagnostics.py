"""Describe a run failure and the evidence it left behind."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from yewo.artifacts import ARTIFACTS_DIRECTORY
from yewo.extensions import ExtensionDiagnosticError
from yewo.host import LOCK_FILE_SUFFIX


_MAXIMUM_CHAINED_ERRORS = 5


def _chained_errors(error: BaseException) -> list[BaseException]:
    """Walk one bounded cause chain, so every renderer below reads the same errors."""
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(chain) < _MAXIMUM_CHAINED_ERRORS:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def error_chain(error: BaseException) -> str:
    """Render the chained error type names without their messages."""
    return "<-".join(type(item).__name__ for item in _chained_errors(error))


def authored_detail(error: BaseException) -> str:
    """Render the messages Yewo wrote itself, skipping any a dependency produced.

    A type name alone does not identify a failure. Yewo raises one error class from many places
    for different reasons, so the trace names a class without naming the check that refused the
    run. A Yewo message describes the broken expectation and at most names configuration. A
    message from outside Yewo may quote whatever the failing call handled. Callers decide where
    this is safe. A few Yewo messages do carry run detail, so device selection reports its class
    alone and uses its own refusal text instead.
    """
    messages = [
        message
        for item in _chained_errors(error)
        if (type(item).__module__.startswith("yewo.") or isinstance(item, ExtensionDiagnosticError))
        and (message := str(item).strip())
    ]
    if not messages:
        return ""
    return f" said={' <- '.join(dict.fromkeys(messages))}"


def validation_detail(error: BaseException) -> str:
    """Render validation failures without copying the checked inputs into the trace."""
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
