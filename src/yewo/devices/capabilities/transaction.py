"""Common transaction boundary for temporary device-state changes."""

from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Protocol, TypeVar, cast

from yewo.artifacts import new_record_identifier
from yewo.devices.capabilities.errors import (
    CapabilityApplicationError,
    CapabilityRestorationError,
)
from yewo.devices.capabilities.records import (
    CapabilityApplicationReceipt,
    CapabilityRestorationReceipt,
    CapabilityStateObservation,
    CapabilitySupportResult,
    CapabilitySupportState,
    CapabilityTransactionReceipt,
    CapabilityTransactionState,
    CapabilityWorkDisposition,
)
from yewo.project_config.capabilities import (
    CapabilitySpec,
    DeviceCapabilitySpec,
)


RequestT = TypeVar("RequestT", bound=CapabilitySpec)
RequestT_contra = TypeVar("RequestT_contra", bound=CapabilitySpec, contravariant=True)


class DeviceCapabilityAdapter(Protocol[RequestT_contra]):
    """Capability-specific state, mutation, verification, and restoration."""

    def support(self, request: RequestT_contra) -> CapabilitySupportResult: ...

    def read_state(self, request: RequestT_contra) -> CapabilityStateObservation: ...

    def validate_original_state(
        self,
        request: RequestT_contra,
        original_state: CapabilityStateObservation,
    ) -> CapabilitySupportResult: ...

    def requested_state(self, request: RequestT_contra) -> str: ...

    def apply(self, request: RequestT_contra) -> None: ...

    def application_verified(
        self,
        request: RequestT_contra,
        observed_state: CapabilityStateObservation,
    ) -> bool: ...

    def restore(
        self,
        request: RequestT_contra,
        original_state: CapabilityStateObservation,
    ) -> None: ...

    def restoration_verified(
        self,
        request: RequestT_contra,
        original_state: CapabilityStateObservation,
        observed_state: CapabilityStateObservation,
    ) -> bool: ...


@dataclass(frozen=True)
class CapabilityPreparation(Generic[RequestT]):
    """Support decision and an optional ready transaction."""

    support: CapabilitySupportResult
    transaction: CapabilityTransaction[RequestT] | None = None


class CapabilityTransactionCoordinator(Generic[RequestT]):
    """Resolve support and prepare one restorable capability transaction."""

    def __init__(
        self,
        target_id: str,
        adapter: DeviceCapabilityAdapter[RequestT],
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        if not target_id.strip():
            raise ValueError("A capability coordinator needs a target identifier.")
        self._target_id = target_id.strip()
        self._adapter = adapter
        self._identifier_factory = identifier_factory

    def prepare(self, request: RequestT) -> CapabilityPreparation[RequestT]:
        """Read support and original state without mutating the device."""
        support = self._adapter.support(request)
        if support.state is not CapabilitySupportState.SUPPORTED:
            return CapabilityPreparation(support=support)
        try:
            original_state = self._adapter.read_state(request)
        except Exception:  # noqa: BLE001 - injected device boundary
            return CapabilityPreparation(
                support=CapabilitySupportResult(
                    capability=support.capability,
                    state=CapabilitySupportState.UNAVAILABLE,
                    reason="Yewo could not read the original device state.",
                )
            )
        restorable = self._adapter.validate_original_state(request, original_state)
        if restorable.state is not CapabilitySupportState.SUPPORTED:
            return CapabilityPreparation(support=restorable)
        return CapabilityPreparation(
            support=restorable,
            transaction=CapabilityTransaction(
                transaction_id=self._identifier_factory(),
                target_id=self._target_id,
                request=request,
                support=restorable,
                original_state=original_state,
                adapter=self._adapter,
            ),
        )


class CapabilityTransaction(Generic[RequestT]):
    """Context-managed mutation with verified restoration on every exit path."""

    def __init__(
        self,
        *,
        transaction_id: str,
        target_id: str,
        request: RequestT,
        support: CapabilitySupportResult,
        original_state: CapabilityStateObservation,
        adapter: DeviceCapabilityAdapter[RequestT],
    ) -> None:
        self._transaction_id = transaction_id
        self._target_id = target_id
        self._request = request
        self._support = support
        self._original_state = original_state
        self._adapter = adapter
        self._application: CapabilityApplicationReceipt | None = None
        self._receipt: CapabilityTransactionReceipt | None = None
        self._entered = False

    @property
    def receipt(self) -> CapabilityTransactionReceipt | None:
        """Return the terminal receipt after application failure or context exit."""
        return self._receipt

    def __enter__(self) -> CapabilityTransaction[RequestT]:
        if self._entered:
            raise RuntimeError("A capability transaction cannot be entered more than once.")
        self._entered = True
        requested_state = self._adapter.requested_state(self._request)
        try:
            self._adapter.apply(self._request)
            observed_state = self._adapter.read_state(self._request)
            verified = self._adapter.application_verified(self._request, observed_state)
            self._application = CapabilityApplicationReceipt(
                requested_state=requested_state,
                observed_state=observed_state,
                verified=verified,
                reason=(
                    "Read-back verified the requested capability state."
                    if verified
                    else "Read-back did not verify the requested capability state."
                ),
            )
            if not verified:
                raise RuntimeError("Capability read-back did not match the request.")
        except BaseException as error:  # noqa: BLE001 - cleanup must cover cancellation
            if self._application is None:
                self._application = CapabilityApplicationReceipt(
                    requested_state=requested_state,
                    verified=False,
                    reason="Yewo could not apply and verify the requested capability state.",
                )
            restoration = self._restore()
            work_disposition = (
                CapabilityWorkDisposition.CANCELLED
                if isinstance(error, CancelledError)
                else CapabilityWorkDisposition.NOT_RUN
            )
            self._receipt = self._build_receipt(
                work_disposition,
                restoration,
            )
            if not isinstance(error, Exception):
                if not restoration.verified:
                    error.add_note(restoration.reason)
                raise
            raise CapabilityApplicationError(self._receipt) from error
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del error_type, traceback
        work_disposition = (
            CapabilityWorkDisposition.COMPLETED
            if error is None
            else CapabilityWorkDisposition.CANCELLED
            if isinstance(error, CancelledError)
            else CapabilityWorkDisposition.FAILED
        )
        restoration = self._restore()
        self._receipt = self._build_receipt(work_disposition, restoration)
        if restoration.verified:
            return False
        if error is not None:
            error.add_note(restoration.reason)
            return False
        raise CapabilityRestorationError(self._receipt)

    def _restore(self) -> CapabilityRestorationReceipt:
        try:
            self._adapter.restore(self._request, self._original_state)
            observed_state = self._adapter.read_state(self._request)
            verified = self._adapter.restoration_verified(
                self._request,
                self._original_state,
                observed_state,
            )
            return CapabilityRestorationReceipt(
                requested_state=self._original_state,
                observed_state=observed_state,
                verified=verified,
                reason=(
                    "Read-back verified restoration of the original state."
                    if verified
                    else "Read-back did not verify restoration of the original state."
                ),
            )
        except Exception:  # noqa: BLE001 - injected device boundary
            return CapabilityRestorationReceipt(
                requested_state=self._original_state,
                verified=False,
                reason="Yewo could not restore and verify the original device state.",
            )

    def _build_receipt(
        self,
        work_disposition: CapabilityWorkDisposition,
        restoration: CapabilityRestorationReceipt,
    ) -> CapabilityTransactionReceipt:
        application = self._application
        if application is None:
            raise AssertionError("A terminal transaction needs an application receipt.")
        state = (
            CapabilityTransactionState.APPLY_FAILED
            if not application.verified
            else CapabilityTransactionState.RESTORED
            if restoration.verified
            else CapabilityTransactionState.RESTORATION_FAILED
        )
        return CapabilityTransactionReceipt(
            transaction_id=self._transaction_id,
            target_id=self._target_id,
            request=cast(DeviceCapabilitySpec, self._request),
            support=self._support,
            original_state=self._original_state,
            application=application,
            work_disposition=work_disposition,
            restoration=restoration,
            state=state,
        )
