"""Deterministic settlement of one dispatched action."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from yewo.agent.memory.control_identity import ControlIdentity
from yewo.agent.memory.screen_identity import ScreenComparison, compare_screens
from yewo.identity import new_uuid


class SettlementStatus(str, Enum):
    DURABLE_PROGRESS = "durable_progress"
    TRANSIENT_OSCILLATION = "transient_oscillation"
    NO_VISIBLE_CHANGE = "no_visible_change"
    DISPATCH_FAILED = "dispatch_failed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class SettlementReceipt:
    """Secret-free result of settling one authorized action."""

    settlement_id: str
    control_identity: ControlIdentity
    source_phash: str | None
    destination_phash: str | None
    status: SettlementStatus
    reason: str
    observation_ids: tuple[str, ...] = ()
    expected_destination_phash: str | None = None
    source_comparison: ScreenComparison | None = None
    destination_comparison: ScreenComparison | None = None


@dataclass
class _SettlementTransaction:
    settlement_id: str
    control_identity: ControlIdentity
    source_phash: str | None
    source_observation_id: str = ""
    destination_phash: str | None = None
    observation_ids: list[str] = field(default_factory=list)
    compatible_observations: int = 0
    dispatch_recorded: bool = False
    expected_destination_phash: str | None = None


class SettlementController:
    """Retain action evidence until progress becomes durable or disproved."""

    def __init__(self) -> None:
        self._active: _SettlementTransaction | None = None
        self._receipts: list[SettlementReceipt] = []

    @property
    def receipts(self) -> tuple[SettlementReceipt, ...]:
        return tuple(self._receipts)

    @property
    def active_settlement_id(self) -> str | None:
        return self._active.settlement_id if self._active is not None else None

    def begin_action(
        self,
        control_identity: ControlIdentity,
        source_phash: str | None,
        source_observation_id: str = "",
        expected_destination_phash: str | None = None,
    ) -> SettlementReceipt | None:
        previous = self._settle_before_next_action(control_identity, source_phash)
        self._active = _SettlementTransaction(
            settlement_id=new_uuid(),
            control_identity=control_identity,
            source_phash=source_phash,
            source_observation_id=source_observation_id,
            expected_destination_phash=expected_destination_phash,
        )
        return previous

    def record_dispatch(
        self,
        control_identity: ControlIdentity,
        source_phash: str | None,
        destination_phash: str | None,
        *,
        failed: bool,
    ) -> SettlementReceipt | None:
        transaction = self._active
        if transaction is None or not transaction.control_identity.matches(control_identity):
            transaction = _SettlementTransaction(
                settlement_id=new_uuid(),
                control_identity=control_identity,
                source_phash=source_phash,
            )
            self._active = transaction
        transaction.dispatch_recorded = True
        transaction.destination_phash = destination_phash
        if failed:
            return self._close(
                SettlementStatus.DISPATCH_FAILED,
                "The device rejected or failed the dispatched action.",
            )
        if source_phash is None or destination_phash is None:
            return self._close(
                SettlementStatus.UNVERIFIED,
                "The action lacked source or destination visual evidence.",
            )
        source_comparison = compare_screens(source_phash, destination_phash)
        if source_comparison is ScreenComparison.UNKNOWN:
            return self._close(
                SettlementStatus.UNVERIFIED,
                "The action lacked comparable source and destination visual evidence.",
                source_comparison=source_comparison,
            )
        if source_comparison is ScreenComparison.SAME:
            return self._close(
                SettlementStatus.NO_VISIBLE_CHANGE,
                "The settled screenshot remained compatible with the source.",
                source_comparison=source_comparison,
            )
        if transaction.expected_destination_phash is not None:
            destination_comparison = compare_screens(
                destination_phash, transaction.expected_destination_phash
            )
            if destination_comparison is not ScreenComparison.SAME:
                return self._close(
                    SettlementStatus.UNVERIFIED,
                    "The navigation destination did not match its observed postcondition.",
                    source_comparison=source_comparison,
                    destination_comparison=destination_comparison,
                )
            transaction.compatible_observations = 1
        return None

    def observe(
        self, page_hash: str | None, observation_id: str = ""
    ) -> SettlementReceipt | None:
        transaction = self._active
        if transaction is None or not transaction.dispatch_recorded or page_hash is None:
            return None
        if observation_id:
            transaction.observation_ids.append(observation_id)
        source_comparison = compare_screens(page_hash, transaction.source_phash)
        if source_comparison is ScreenComparison.SAME:
            return self._close(
                SettlementStatus.TRANSIENT_OSCILLATION,
                "The application returned to the source without another action.",
                source_comparison=source_comparison,
            )
        destination_comparison = (
            compare_screens(page_hash, transaction.expected_destination_phash)
            if transaction.expected_destination_phash is not None
            else compare_screens(page_hash, transaction.destination_phash)
        )
        if destination_comparison is not ScreenComparison.SAME:
            transaction.compatible_observations = 0
            return None
        transaction.compatible_observations += 1
        if transaction.compatible_observations < 2:
            return None
        return self._close(
            SettlementStatus.DURABLE_PROGRESS,
            "The destination remained stable across independent observations.",
            source_comparison=source_comparison,
            destination_comparison=destination_comparison,
        )

    def close_unverified(self, reason: str) -> SettlementReceipt | None:
        if self._active is None:
            return None
        return self._close(SettlementStatus.UNVERIFIED, reason)

    def _settle_before_next_action(
        self,
        next_control: ControlIdentity,
        source_phash: str | None,
    ) -> SettlementReceipt | None:
        transaction = self._active
        if transaction is None:
            return None
        source_comparison = compare_screens(source_phash, transaction.source_phash)
        if source_comparison is ScreenComparison.SAME:
            return self._close(
                SettlementStatus.TRANSIENT_OSCILLATION,
                "The next action began from the original source state.",
                source_comparison=source_comparison,
            )
        return self._close(
            SettlementStatus.UNVERIFIED,
            "Another action began before durable progress was established; prior repetition evidence remains active.",
            source_comparison=source_comparison,
        )

    def _close(
        self,
        status: SettlementStatus,
        reason: str,
        *,
        source_comparison: ScreenComparison | None = None,
        destination_comparison: ScreenComparison | None = None,
    ) -> SettlementReceipt:
        transaction = self._active
        if transaction is None:
            raise RuntimeError("No settlement transaction is active.")
        observation_ids = tuple(
            identifier
            for identifier in (
                transaction.source_observation_id,
                *transaction.observation_ids,
            )
            if identifier
        )
        receipt = SettlementReceipt(
            settlement_id=transaction.settlement_id,
            control_identity=transaction.control_identity,
            source_phash=transaction.source_phash,
            destination_phash=transaction.destination_phash,
            status=status,
            reason=reason,
            observation_ids=observation_ids,
            expected_destination_phash=transaction.expected_destination_phash,
            source_comparison=source_comparison,
            destination_comparison=destination_comparison,
        )
        self._receipts.append(receipt)
        self._active = None
        return receipt
