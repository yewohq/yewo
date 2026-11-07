"""Tests for the shared capability transaction boundary."""

from asyncio import CancelledError

import pytest

from yewo.devices.capabilities import (
    CapabilityApplicationError,
    CapabilityRestorationError,
    CapabilityStateObservation,
    CapabilitySupportResult,
    CapabilitySupportState,
    CapabilityTransactionCoordinator,
    CapabilityTransactionState,
    CapabilityWorkDisposition,
)
from yewo.project_config import CapabilityKind
from yewo.project_config.capabilities import AppStateCapabilitySpec


class _CapabilityAdapter:
    def __init__(
        self,
        *,
        support_state: CapabilitySupportState = CapabilitySupportState.SUPPORTED,
        read_fails: bool = False,
        apply_fails: bool = False,
        apply_cancels: bool = False,
        restoration_fails: bool = False,
    ) -> None:
        self.support_state = support_state
        self.read_fails = read_fails
        self.apply_fails = apply_fails
        self.apply_cancels = apply_cancels
        self.restoration_fails = restoration_fails
        self.state = "original"
        self.read_count = 0
        self.apply_count = 0
        self.restore_count = 0

    def support(self, request: AppStateCapabilitySpec) -> CapabilitySupportResult:
        del request
        return CapabilitySupportResult(
            capability=CapabilityKind.APP_STATE,
            state=self.support_state,
            reason="The fake adapter returned its configured support state.",
        )

    def read_state(self, request: AppStateCapabilitySpec) -> CapabilityStateObservation:
        del request
        self.read_count += 1
        if self.read_fails:
            raise RuntimeError("private device failure")
        return CapabilityStateObservation(value=self.state, source="fake device")

    def validate_original_state(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
    ) -> CapabilitySupportResult:
        del request, original_state
        return CapabilitySupportResult(
            capability=CapabilityKind.APP_STATE,
            state=CapabilitySupportState.SUPPORTED,
            reason="The fake original state is restorable.",
        )

    def requested_state(self, request: AppStateCapabilitySpec) -> str:
        del request
        return "changed"

    def apply(self, request: AppStateCapabilitySpec) -> None:
        del request
        self.apply_count += 1
        self.state = "changed"
        if self.apply_cancels:
            raise CancelledError
        if self.apply_fails:
            raise RuntimeError("private apply failure")

    def application_verified(
        self,
        request: AppStateCapabilitySpec,
        observed_state: CapabilityStateObservation,
    ) -> bool:
        del request
        return observed_state.value == "changed"

    def restore(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
    ) -> None:
        del request
        self.restore_count += 1
        if self.restoration_fails:
            raise RuntimeError("private restoration failure")
        self.state = original_state.value

    def restoration_verified(
        self,
        request: AppStateCapabilitySpec,
        original_state: CapabilityStateObservation,
        observed_state: CapabilityStateObservation,
    ) -> bool:
        del request
        return observed_state == original_state


REQUEST = AppStateCapabilitySpec(action="background_resume")


def _prepare(adapter: _CapabilityAdapter):
    return CapabilityTransactionCoordinator(
        "device-1",
        adapter,
        identifier_factory=lambda: "transaction-1",
    ).prepare(REQUEST)


def test_unsupported_capability_returns_before_read_or_mutation() -> None:
    adapter = _CapabilityAdapter(support_state=CapabilitySupportState.UNSUPPORTED)

    preparation = _prepare(adapter)

    assert preparation.support.state is CapabilitySupportState.UNSUPPORTED
    assert preparation.transaction is None
    assert adapter.read_count == 0
    assert adapter.apply_count == 0


def test_unreadable_original_state_is_unavailable_without_mutation() -> None:
    adapter = _CapabilityAdapter(read_fails=True)

    preparation = _prepare(adapter)

    assert preparation.support.state is CapabilitySupportState.UNAVAILABLE
    assert preparation.transaction is None
    assert adapter.apply_count == 0
    assert "private" not in preparation.support.reason


def test_successful_work_records_application_and_restoration() -> None:
    adapter = _CapabilityAdapter()
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with transaction:
        assert adapter.state == "changed"

    receipt = transaction.receipt
    assert receipt is not None
    assert receipt.application.verified
    assert receipt.work_disposition is CapabilityWorkDisposition.COMPLETED
    assert receipt.restoration.verified
    assert receipt.state is CapabilityTransactionState.RESTORED
    assert adapter.state == "original"


def test_work_failure_is_preserved_after_verified_restoration() -> None:
    adapter = _CapabilityAdapter()
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(LookupError, match="scenario failure"), transaction:
        raise LookupError("scenario failure")

    assert transaction.receipt is not None
    assert transaction.receipt.work_disposition is CapabilityWorkDisposition.FAILED
    assert transaction.receipt.restoration.verified


def test_cancellation_is_recorded_and_propagated_after_restoration() -> None:
    adapter = _CapabilityAdapter()
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(CancelledError), transaction:
        raise CancelledError

    assert transaction.receipt is not None
    assert transaction.receipt.work_disposition is CapabilityWorkDisposition.CANCELLED
    assert transaction.receipt.restoration.verified


def test_partial_apply_failure_restores_and_returns_a_safe_receipt() -> None:
    adapter = _CapabilityAdapter(apply_fails=True)
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(CapabilityApplicationError) as captured, transaction:
        raise AssertionError("work must not start")

    receipt = captured.value.receipt
    assert receipt.state is CapabilityTransactionState.APPLY_FAILED
    assert receipt.work_disposition is CapabilityWorkDisposition.NOT_RUN
    assert receipt.restoration.verified
    assert "private" not in receipt.application.reason
    assert adapter.state == "original"


def test_cancellation_during_apply_restores_and_propagates() -> None:
    adapter = _CapabilityAdapter(apply_cancels=True)
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(CancelledError), transaction:
        raise AssertionError("work must not start")

    assert transaction.receipt is not None
    assert transaction.receipt.work_disposition is CapabilityWorkDisposition.CANCELLED
    assert transaction.receipt.state is CapabilityTransactionState.APPLY_FAILED
    assert transaction.receipt.restoration.verified


def test_restoration_failure_is_terminal_after_successful_work() -> None:
    adapter = _CapabilityAdapter(restoration_fails=True)
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(CapabilityRestorationError) as captured, transaction:
        pass

    assert captured.value.receipt.state is CapabilityTransactionState.RESTORATION_FAILED
    assert not captured.value.receipt.restoration.verified


def test_restoration_failure_does_not_hide_the_work_failure() -> None:
    adapter = _CapabilityAdapter(restoration_fails=True)
    transaction = _prepare(adapter).transaction
    assert transaction is not None

    with pytest.raises(LookupError, match="scenario failure") as captured, transaction:
        raise LookupError("scenario failure")

    assert transaction.receipt is not None
    assert transaction.receipt.state is CapabilityTransactionState.RESTORATION_FAILED
    assert captured.value.__notes__ == [
        "Yewo could not restore and verify the original device state."
    ]
