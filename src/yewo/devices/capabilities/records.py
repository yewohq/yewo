"""Records for verified and restorable device-capability transactions."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.project_config.capabilities import CapabilityKind, DeviceCapabilitySpec


class CapabilitySupportState(StrEnum):
    """Whether a target can execute and restore one capability request."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class CapabilityWorkDisposition(StrEnum):
    """How scenario work ended while a capability transaction was active."""

    NOT_RUN = "not_run"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CapabilityTransactionState(StrEnum):
    """Terminal state of one capability transaction."""

    APPLY_FAILED = "apply_failed"
    RESTORED = "restored"
    RESTORATION_FAILED = "restoration_failed"


class CapabilitySupportResult(BaseModel):
    """Explicit support decision made before device mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    capability: CapabilityKind
    state: CapabilitySupportState
    reason: str = Field(min_length=1, max_length=2_000)


class CapabilityStateObservation(BaseModel):
    """One normalized device-state reading and its evidence source."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    value: str = Field(min_length=1, max_length=2_000)
    source: str = Field(min_length=1, max_length=500)


class CapabilityApplicationReceipt(BaseModel):
    """Read-back evidence for the requested device-state change."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    requested_state: str = Field(min_length=1, max_length=2_000)
    observed_state: CapabilityStateObservation | None = None
    verified: bool
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _validate_verified_state(self) -> Self:
        if self.verified and self.observed_state is None:
            raise ValueError("A verified capability application needs observed state.")
        return self


class CapabilityRestorationReceipt(BaseModel):
    """Read-back evidence that the original device state was restored."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    requested_state: CapabilityStateObservation
    observed_state: CapabilityStateObservation | None = None
    verified: bool
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _validate_verified_state(self) -> Self:
        if self.verified and self.observed_state is None:
            raise ValueError("A verified restoration needs observed state.")
        return self


class CapabilityTransactionReceipt(BaseModel):
    """Complete evidence for one applied and restored capability request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: RecordIdentifier
    target_id: str = Field(min_length=1, max_length=500)
    request: DeviceCapabilitySpec
    support: CapabilitySupportResult
    original_state: CapabilityStateObservation
    application: CapabilityApplicationReceipt
    work_disposition: CapabilityWorkDisposition
    restoration: CapabilityRestorationReceipt
    state: CapabilityTransactionState

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> Self:
        expected = (
            CapabilityTransactionState.APPLY_FAILED
            if not self.application.verified
            else CapabilityTransactionState.RESTORED
            if self.restoration.verified
            else CapabilityTransactionState.RESTORATION_FAILED
        )
        if self.state is not expected:
            raise ValueError("Capability transaction state does not match its receipts.")
        if self.support.state is not CapabilitySupportState.SUPPORTED:
            raise ValueError("A capability transaction receipt requires supported execution.")
        return self
