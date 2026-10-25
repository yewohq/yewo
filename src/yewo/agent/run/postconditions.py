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
from yewo.utils.secrets import AppValueRendering

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


@dataclass(frozen=True)
class PendingEnteredValue:
    """The transient value and target retained only until the next observation."""

    specification: EnteredValuePostcondition
    expected: str
    coordinates: Coordinates
    entry_receipt: Optional[str] = None


@dataclass(frozen=True)
class TextEntryReceipt:
    """The focused element value read immediately after the device typed."""

    observed: Optional[str]


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
    settled_value = focused_field.value if focused_field is not None else None
    target_point = _focused_centre(focused_field) or pending.coordinates
    candidates = [
        element.text.strip()
        for element in elements
        if element.text.strip() and _contains(element, target_point)
    ]
    app_rendering = _app_rendering(pending, focused_field, candidates)
    if settled_value is not None and _matches(pending, settled_value):
        return _result(
            pending,
            VerificationStatus.PASSED,
            "The settled field confirmed it.",
            app_rendering=app_rendering,
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
            )
        if pending.entry_receipt is not None:
            return _result(
                pending,
                VerificationStatus.FAILED,
                "The entry action did not deliver the intended declared value.",
                VerificationClassification.AGENT_ACTION,
                app_rendering,
            )
        return _result(
            pending,
            VerificationStatus.UNVERIFIED,
            "The field differed, but Yewo could not attribute the mismatch safely.",
            VerificationClassification.EVIDENCE,
            app_rendering,
        )
    return _result(
        pending,
        VerificationStatus.UNVERIFIED,
        "No settled field value or target-local text was available after the entry action.",
        VerificationClassification.EVIDENCE,
        app_rendering,
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
    ctx.postcondition_blocked = verification.classification in (
        VerificationClassification.APPLICATION,
        VerificationClassification.EVIDENCE,
    )
    ctx.postcondition_recovery_required = (
        verification.classification is VerificationClassification.AGENT_ACTION
    )
    logger.info(
        "Entered-value postcondition for %s: %s.",
        verification.value_name,
        verification.status.value,
    )
    return verification


def _contains(element: GUIElement, point: Coordinates) -> bool:
    box = element.bounding_box
    return box.x_min <= point.x <= box.x_max and box.y_min <= point.y <= box.y_max


MASKS = frozenset("*•●·▪■")
HYPHENS = frozenset("-‐‑‒–—")
GROUPING = frozenset(",'_\u00a0")


def _focused_centre(focused_field: Optional[FocusedFieldReading]) -> Optional[Coordinates]:
    if focused_field is None or focused_field.bounds is None:
        return None
    left, top, right, bottom = focused_field.bounds
    return Coordinates(x=(left + right) // 2, y=(top + bottom) // 2)


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
) -> PostconditionVerification:
    return PostconditionVerification(
        value_name=pending.specification.value_name,
        status=status,
        classification=classification,
        reason=reason,
        app_rendering=app_rendering,
    )
