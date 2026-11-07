"""Verified application state established before scenario execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from yewo.artifacts import RecordIdentifier, new_record_identifier


ApplicationPrecondition = Literal["clear_data", "restart", "preserve"] | None


class ApplicationPreconditionBoundary(Protocol):
    """Application operations used to establish one scenario start state."""

    def clear_application_data(self) -> None: ...

    def activate(self) -> None: ...

    def relaunch(self) -> None: ...

    def is_foreground(self) -> bool: ...


class ApplicationPreconditionReceipt(BaseModel):
    """Read-back evidence for one requested scenario application state."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    receipt_id: RecordIdentifier
    requested_state: Literal["clear_data", "restart", "preserve"]
    verified: bool
    reason: str = Field(min_length=1, max_length=2_000)


class ApplicationPreconditionCoordinator:
    """Apply one application precondition and verify foreground state."""

    def __init__(
        self,
        application: ApplicationPreconditionBoundary,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._application = application
        self._identifier_factory = identifier_factory

    def establish(
        self,
        requested_state: ApplicationPrecondition,
    ) -> ApplicationPreconditionReceipt:
        """Apply the requested state without exposing raw driver errors."""
        normalized_state = requested_state or "preserve"
        try:
            if normalized_state == "clear_data":
                self._application.clear_application_data()
                self._application.activate()
            elif normalized_state == "restart":
                self._application.relaunch()
            verified = self._application.is_foreground()
        except Exception:  # noqa: BLE001 - injected device boundary
            return self._receipt(
                normalized_state,
                False,
                "Yewo could not establish the requested application state.",
            )
        return self._receipt(
            normalized_state,
            verified,
            (
                "Read-back verified the requested application state."
                if verified
                else "Read-back did not verify a foreground application."
            ),
        )

    def _receipt(
        self,
        requested_state: Literal["clear_data", "restart", "preserve"],
        verified: bool,
        reason: str,
    ) -> ApplicationPreconditionReceipt:
        return ApplicationPreconditionReceipt(
            receipt_id=self._identifier_factory(),
            requested_state=requested_state,
            verified=verified,
            reason=reason,
        )
