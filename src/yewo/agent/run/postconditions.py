"""Action-derived facts that must hold before another action may run.

These verifications are local. The planner cannot grade its own action. The
screen auditor must not receive a secret to prove that the device entered it.
A postcondition names a declared value. It records only whether current-run
evidence agreed with that value.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import unicodedata
from typing import Callable, Literal, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.device.focus import FocusedFieldReading
from yewo.agent.perception.elements import GUIElement
from yewo.agent.run.input_transaction import (
    FocusReceipt,
    InputTransactionReceipt,
    InputTransactionStatus,
)
from yewo.identity import new_uuid
from yewo.utils.secrets import AppValueRendering
from yewo.agent.run.recovery import (
    RecoveryCause,
    RecoveryTransaction,
)

logger = logging.getLogger(__name__)


class ValueResolver(Protocol):
    def name_of(self, text: Optional[str]) -> Optional[str]: ...


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class VerificationClassification(str, Enum):
    AGENT_ACTION = "agent_action"
    APPLICATION = "application"
    EVIDENCE = "evidence"


Normalizer = Literal["whitespace", "hyphens", "case", "currency", "grouping"]


class ValueVerificationPolicy(BaseModel):
    """The transformations and masking representation a journey permits."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["auto", "exact", "normalized", "masked"] = "auto"
    normalizers: tuple[Normalizer, ...] = ()
    visible_suffix: int = Field(default=0, ge=0)
    masked_length: Optional[int] = Field(default=None, ge=1)


class EnteredValuePostcondition(BaseModel):
    """A typed action must leave the current run's declared value in its field."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["entered_value"] = "entered_value"
    value_name: str = Field(min_length=1)
    target: Optional[str] = None
    comparison: Literal["auto", "exact", "normalized", "masked"] = "auto"
    normalizers: tuple[Normalizer, ...] = ()
    visible_suffix: int = Field(default=0, ge=0)
    masked_length: Optional[int] = Field(default=None, ge=1)


class PostconditionVerification(BaseModel):
    """Secret-free evidence used by the execution barrier and final verdict."""

    kind: Literal["entered_value"] = "entered_value"
    value_name: str
    status: VerificationStatus
    classification: Optional[VerificationClassification] = None
    reason: str
    app_rendering: AppValueRendering = AppValueRendering.UNKNOWN
    transaction_id: Optional[str] = None
    context_retained: Optional[bool] = None


@dataclass(frozen=True)
class PendingEnteredValue:
    """The transient value and target retained only until the next observation."""

    specification: EnteredValuePostcondition
    expected: str
    coordinates: Coordinates
    entry_receipt: Optional[str] = None
    transaction_id: str = ""
    focus: FocusReceipt = FocusReceipt()
    source_observation_id: Optional[str] = None
    source_context_id: Optional[str] = None
    source_phash: Optional[str] = None


@dataclass(frozen=True)
class TextEntryReceipt:
    """The focused element value read immediately after the device typed."""

    observed: Optional[str]
    focus: FocusReceipt = FocusReceipt()


def specification_for(
    action: TypeAction,
    names: ValueResolver,
    verification_policies: Optional[Mapping[str, ValueVerificationPolicy | str]] = None,
) -> Optional[EnteredValuePostcondition]:
    """Derive a reusable verification from a typed declared value."""
    value_name = names.name_of(action.text)
    if value_name is None:
        return None
    policy = verification_policy((verification_policies or {}).get(value_name))
    return EnteredValuePostcondition(
        value_name=value_name,
        target=action.target,
        comparison=policy.mode,
        normalizers=policy.normalizers,
        visible_suffix=policy.visible_suffix,
        masked_length=policy.masked_length,
    )


def pending_for(
    action: TypeAction,
    names: ValueResolver,
    values: dict[str, str],
    entry_receipt: Optional[str],
    specification: Optional[EnteredValuePostcondition] = None,
    verification_policies: Optional[Mapping[str, ValueVerificationPolicy | str]] = None,
    focus: Optional[FocusReceipt] = None,
    source_observation_id: Optional[str] = None,
    source_context_id: Optional[str] = None,
    source_phash: Optional[str] = None,
) -> Optional[PendingEnteredValue]:
    """Build the current-run verification without saving the value in the path."""
    contract = specification or specification_for(action, names, verification_policies)
    if contract is None:
        return None
    expected = values.get(contract.value_name)
    if expected is None:
        return None
    return PendingEnteredValue(
        specification=contract,
        expected=expected,
        coordinates=action.coordinates,
        entry_receipt=entry_receipt,
        transaction_id=new_uuid(),
        focus=focus or FocusReceipt(),
        source_observation_id=source_observation_id,
        source_context_id=source_context_id,
        source_phash=source_phash,
    )


def verification_policy(
    value: Optional[ValueVerificationPolicy | str],
) -> ValueVerificationPolicy:
    """Adapt the legacy string spelling at the path API boundary."""
    if isinstance(value, ValueVerificationPolicy):
        return value
    if value == "masked":
        return ValueVerificationPolicy(mode="masked")
    if value == "normalized":
        return ValueVerificationPolicy(mode="normalized")
    if value == "exact":
        return ValueVerificationPolicy(mode="exact")
    return ValueVerificationPolicy()


def verify(
    pending: PendingEnteredValue,
    elements: Sequence[GUIElement],
    focused_field: Optional[FocusedFieldReading] = None,
) -> PostconditionVerification:
    """Verify one settled entry. Use the dispatch receipt for attribution.

    A value found elsewhere on the screen is not accepted.  It could be an
    example, a previous value, or unrelated content.  OCR is evidence only
    from an element that still contains the action point.
    """
    retained_field = (
        focused_field
        if focused_field is not None
        and focused_field.bounds is not None
        and _bounds_contain(focused_field.bounds, pending.coordinates)
        else None
    )
    settled_value = retained_field.value if retained_field is not None else None
    candidates = [
        element.text.strip()
        for element in elements
        if element.text.strip() and _contains(element, pending.coordinates)
    ]
    context_retained = retained_field is not None or bool(candidates)
    app_rendering = _app_rendering(pending, retained_field, candidates)
    if settled_value is not None and _matches(pending, settled_value):
        return _result(
            pending,
            VerificationStatus.PASSED,
            "The settled field confirmed it.",
            app_rendering=app_rendering,
            context_retained=True,
        )
    if any(
        _matches(pending, line.strip())
        for candidate in candidates
        for line in candidate.splitlines()
    ):
        return _result(
            pending,
            VerificationStatus.PASSED,
            "Target-local text confirmed it.",
            app_rendering=app_rendering,
            context_retained=True,
        )
    if not context_retained:
        return _result(
            pending,
            VerificationStatus.UNVERIFIED,
            "The original field or form context was not retained after text entry.",
            VerificationClassification.EVIDENCE,
            app_rendering,
            context_retained=False,
        )
    receipt_matches = pending.entry_receipt is not None and _matches(
        pending, pending.entry_receipt
    )
    if settled_value is not None or candidates:
        if receipt_matches:
            return _result(
                pending,
                VerificationStatus.FAILED,
                "The app changed a correctly entered value after the entry action.",
                VerificationClassification.APPLICATION,
                app_rendering,
                context_retained=True,
            )
        if pending.entry_receipt is not None:
            return _result(
                pending,
                VerificationStatus.FAILED,
                "The entry action did not deliver the intended declared value.",
                VerificationClassification.AGENT_ACTION,
                app_rendering,
                context_retained=True,
            )
        return _result(
            pending,
            VerificationStatus.UNVERIFIED,
            "The field differed, but Yewo could not attribute the mismatch safely.",
            VerificationClassification.EVIDENCE,
            app_rendering,
            context_retained=True,
        )
    return _result(
        pending,
        VerificationStatus.UNVERIFIED,
        "No settled field value or target-local text was available after the entry action.",
        VerificationClassification.EVIDENCE,
        app_rendering,
        context_retained=context_retained,
    )


def verify_context(
    snapshot,
    ctx,
    read_focused_field: Optional[Callable[[], FocusedFieldReading]] = None,
    protect_focused_region: Optional[
        Callable[[str, tuple[int, int, int, int], AppValueRendering], None]
    ] = None,
) -> Optional[PostconditionVerification]:
    """Settle the previous action before audit, cache selection, or planning."""
    pending = ctx.pending_entered_value
    if pending is None:
        return None
    ctx.pending_entered_value = None
    focused_field = read_focused_field() if read_focused_field is not None else None
    verification = verify(pending, snapshot.gui_elements, focused_field)
    if (
        focused_field is not None
        and focused_field.bounds is not None
        and protect_focused_region is not None
    ):
        protect_focused_region(
            pending.expected,
            focused_field.bounds,
            verification.app_rendering,
        )
    ctx.postcondition_verifications.append(verification)
    ctx.input_transactions.append(
        _input_transaction_receipt(pending, verification, snapshot)
    )
    ctx.postcondition_blocked = verification.classification in (
        VerificationClassification.APPLICATION,
        VerificationClassification.EVIDENCE,
    )
    ctx.postcondition_recovery_required = (
        verification.classification is VerificationClassification.AGENT_ACTION
    )
    _settle_recovery(ctx, pending, verification)
    logger.info(
        "Entered-value postcondition for %s: %s.",
        verification.value_name,
        verification.status.value,
    )
    return verification


def close_pending_input(ctx, reason: str) -> Optional[PostconditionVerification]:
    """Persist an interrupted input transaction without inventing an observation."""
    pending = ctx.pending_entered_value
    if pending is None:
        return None
    ctx.pending_entered_value = None
    verification = _result(
        pending,
        VerificationStatus.UNVERIFIED,
        reason,
        VerificationClassification.EVIDENCE,
    )
    ctx.postcondition_verifications.append(verification)
    ctx.input_transactions.append(
        InputTransactionReceipt(
            transaction_id=pending.transaction_id,
            value_name=pending.specification.value_name,
            status=InputTransactionStatus.UNVERIFIED,
            value_verified=False,
            context_retained=False,
            reason=reason,
            focus=pending.focus,
            source_observation_id=pending.source_observation_id,
            source_context_id=pending.source_context_id,
            source_phash=pending.source_phash,
        )
    )
    return verification


def _settle_recovery(ctx, pending, verification: PostconditionVerification) -> None:
    """Open or close the transaction; the planner never grades this evidence."""
    active = ctx.active_recovery
    if verification.status is VerificationStatus.PASSED:
        if active is not None and active.value_name == verification.value_name:
            active.recover(verification.reason)
            ctx.active_recovery = None
            ctx.postcondition_recovery_required = False
        return

    if active is not None and active.value_name == verification.value_name:
        if verification.classification is VerificationClassification.AGENT_ACTION:
            return
        active.fail(verification.reason)
        ctx.active_recovery = None
        return

    if verification.classification is not VerificationClassification.AGENT_ACTION:
        return

    current_step = ctx.steps.current.n if ctx.steps.current is not None else 1
    transaction = RecoveryTransaction.open(
        failed_step=max(1, current_step - 1),
        opened_step=current_step,
        original_intent=pending.specification.target or (
            f"Enter the declared value named {pending.specification.value_name}"
        ),
        value_name=pending.specification.value_name,
        failure_reason=verification.reason,
        cause=RecoveryCause.GROUNDING_DRIFT,
    )
    ctx.recovery_transactions.append(transaction)
    ctx.active_recovery = transaction


def _contains(element: GUIElement, point: Coordinates) -> bool:
    box = element.bounding_box
    return box.x_min <= point.x <= box.x_max and box.y_min <= point.y <= box.y_max


MASKS = frozenset("*•●·▪■")
HYPHENS = frozenset("-‐‑‒–—")
GROUPING = frozenset(",'_\u00a0")


def _bounds_contain(bounds: tuple[int, int, int, int], point: Coordinates) -> bool:
    left, top, right, bottom = bounds
    return left <= point.x <= right and top <= point.y <= bottom


def _normalize(value: str, normalizers: Sequence[Normalizer]) -> str:
    normalized = value
    if "whitespace" in normalizers:
        normalized = "".join(character for character in normalized if not character.isspace())
    if "hyphens" in normalizers:
        normalized = "".join(character for character in normalized if character not in HYPHENS)
    if "currency" in normalizers:
        normalized = "".join(
            character for character in normalized if unicodedata.category(character) != "Sc"
        )
    if "grouping" in normalizers:
        normalized = "".join(character for character in normalized if character not in GROUPING)
    if "case" in normalizers:
        normalized = normalized.casefold()
    return normalized


def _matches(pending: PendingEnteredValue, observed: str) -> bool:
    expected = _normalize(pending.expected, pending.specification.normalizers)
    normalized_observed = _normalize(observed, pending.specification.normalizers)
    if normalized_observed == expected:
        return True
    if pending.specification.comparison == "normalized":
        return False
    if pending.specification.comparison not in ("auto", "masked"):
        return False
    compact = "".join(
        character for character in normalized_observed if not character.isspace()
    )
    required_length = pending.specification.masked_length or len(expected)
    if len(compact) != required_length or not compact:
        return False
    suffix_length = pending.specification.visible_suffix
    if suffix_length == 0:
        return all(character in MASKS for character in compact)
    if suffix_length > len(expected) or suffix_length >= len(compact):
        return False
    suffix = expected[-suffix_length:]
    prefix = compact[:-suffix_length]
    return compact.endswith(suffix) and all(character in MASKS for character in prefix)


def _is_masked_representation(pending: PendingEnteredValue, observed: str) -> bool:
    """Whether observed characters are the permitted replacement representation."""
    expected = _normalize(pending.expected, pending.specification.normalizers)
    normalized_observed = _normalize(observed, pending.specification.normalizers)
    if normalized_observed == expected:
        return False
    compact = "".join(
        character for character in normalized_observed if not character.isspace()
    )
    required_length = pending.specification.masked_length or len(expected)
    if len(compact) != required_length or not compact:
        return False
    suffix_length = pending.specification.visible_suffix
    if suffix_length == 0:
        return all(character in MASKS for character in compact)
    if suffix_length > len(expected) or suffix_length >= len(compact):
        return False
    return compact.endswith(expected[-suffix_length:]) and all(
        character in MASKS for character in compact[:-suffix_length]
    )


def _app_rendering(
    pending: PendingEnteredValue,
    focused_field: Optional[FocusedFieldReading],
    candidates: Sequence[str],
) -> AppValueRendering:
    """Classify display evidence without retaining the declared characters."""
    if focused_field is not None and focused_field.is_secure is True:
        return AppValueRendering.MASKED
    observed_values = list(candidates)
    if focused_field is not None and focused_field.value is not None:
        observed_values.append(focused_field.value)
    if any(_is_masked_representation(pending, observed) for observed in observed_values):
        return AppValueRendering.MASKED
    if any(
        _normalize(observed, pending.specification.normalizers)
        == _normalize(pending.expected, pending.specification.normalizers)
        for observed in candidates
    ):
        return AppValueRendering.EXPOSED
    if (
        focused_field is not None
        and focused_field.is_secure is False
        and focused_field.value is not None
        and _normalize(focused_field.value, pending.specification.normalizers)
        == _normalize(pending.expected, pending.specification.normalizers)
    ):
        return AppValueRendering.EXPOSED
    return AppValueRendering.UNKNOWN


def _result(
    pending: PendingEnteredValue,
    status: VerificationStatus,
    reason: str,
    classification: Optional[VerificationClassification] = None,
    app_rendering: AppValueRendering = AppValueRendering.UNKNOWN,
    context_retained: Optional[bool] = None,
) -> PostconditionVerification:
    return PostconditionVerification(
        value_name=pending.specification.value_name,
        status=status,
        classification=classification,
        reason=reason,
        app_rendering=app_rendering,
        transaction_id=pending.transaction_id or None,
        context_retained=context_retained,
    )


def _input_transaction_receipt(
    pending: PendingEnteredValue,
    verification: PostconditionVerification,
    snapshot,
) -> InputTransactionReceipt:
    retained = verification.context_retained is True
    verified = verification.status is VerificationStatus.PASSED
    if verified and retained:
        status = InputTransactionStatus.COMPLETED
    elif not retained:
        status = InputTransactionStatus.CONTEXT_LOST
    elif verification.status is VerificationStatus.FAILED:
        status = InputTransactionStatus.VALUE_FAILED
    else:
        status = InputTransactionStatus.UNVERIFIED
    return InputTransactionReceipt(
        transaction_id=pending.transaction_id,
        value_name=verification.value_name,
        status=status,
        value_verified=verified,
        context_retained=retained,
        reason=verification.reason,
        focus=pending.focus,
        source_observation_id=pending.source_observation_id,
        destination_observation_id=getattr(snapshot, "observation_id", None),
        source_context_id=pending.source_context_id,
        source_phash=pending.source_phash,
        destination_phash=getattr(snapshot, "current_phash", None),
    )
