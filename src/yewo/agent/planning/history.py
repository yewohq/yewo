"""Screen-scoped planner history and bounded prompt rendering."""

from __future__ import annotations

import json
from typing import Annotated, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation.diagnostics import ActionApplicationFailure, planner_failure_context
from yewo.agent.actuation import (
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
    ActionRefusal,
    DeclaredValueReference,
    DispatchDisposition,
    is_correctable_rejection,
    DispatchReceipt,
    LiteralText,
)
from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundedPoint, GroundingCorrection, GroundingEvidence, PixelPoint
from yewo.models import TextModelContent
from yewo.observations import Observation


RECENT_PLANNING_HISTORY_LIMIT = 14
PLANNING_HISTORY_ANCHOR_STEPS = 8
EARLIER_HISTORY_SUMMARY_LIMIT = 20
EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT = 6_000
GROUNDING_CORRECTION_MINIMUM_DISTANCE_FRACTION = 0.02
GROUNDING_CORRECTION_MINIMUM_DISTANCE_PX = 8
MAXIMUM_SCREEN_FACT_CHARACTERS = 1_000

ScreenFactText = Annotated[
    str,
    Field(min_length=1, max_length=MAXIMUM_SCREEN_FACT_CHARACTERS),
]


class PlanningHistorySummary(TypedDict):
    """Bounded older-history entries and their explicit omission count."""

    entries: list[dict[str, object]]
    omitted_groups: int


class PlanningScreenReference(BaseModel):
    """A compact screen identity for one planner-history edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: RecordIdentifier
    platform_context: str | None = Field(default=None, max_length=200)
    keyboard_visible: bool | None = None
    screen_fingerprint: str = Field(pattern=r"^[0-9a-f]{16}$")


class PlanningScreenFact(BaseModel):
    """One deterministic fact scoped to the screen where Yewo learned it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    screen: PlanningScreenReference
    fact: ScreenFactText
    origin_proposal_id: RecordIdentifier | None = None
    is_durable_constraint: bool = False
    supersedes_recovery_guidance: bool = False


class GroundedTargetPoint(BaseModel):
    """A dispatched target point and its source-screenshot dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point: PixelPoint
    screenshot_width_px: int = Field(gt=0)
    screenshot_height_px: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_point_bounds(self) -> Self:
        if self.point.x >= self.screenshot_width_px or self.point.y >= self.screenshot_height_px:
            raise ValueError("The grounded target point must be within the source screenshot.")
        return self

    def correction(self) -> GroundingCorrection:
        """Build the explicit minimum-distance rule for one correction."""
        minimum_dimension = min(self.screenshot_width_px, self.screenshot_height_px)
        return GroundingCorrection(
            previous_point=self.point,
            minimum_distance_px=max(
                GROUNDING_CORRECTION_MINIMUM_DISTANCE_PX,
                round(minimum_dimension * GROUNDING_CORRECTION_MINIMUM_DISTANCE_FRACTION),
            ),
        )


class PlanningHistoryEntry(BaseModel):
    """One previous proposal, its screen edge, and trusted runtime feedback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: ActionProposal
    source_screen: PlanningScreenReference | None = None
    destination_screen: PlanningScreenReference | None = None
    refusal: ActionRefusal | None = None
    receipt: DispatchReceipt | None = None
    outcome: ActionOutcome | None = None
    application_failure: ActionApplicationFailure | None = None
    grounded_target_point: GroundedTargetPoint | None = None

    @model_validator(mode="after")
    def _validate_feedback(self) -> Self:
        feedback = tuple(item for item in (self.refusal, self.receipt, self.outcome) if item)
        if len(feedback) != 1:
            raise ValueError("Planning history requires exactly one feedback record.")
        if self.source_screen is not None and (
            self.source_screen.observation_id != self.proposal.source_observation_id
        ):
            raise ValueError("The history source screen does not match the proposal.")
        self._validate_feedback_owner()
        self._validate_destination_screen()
        has_point_feedback = (
            self.outcome is not None
            and self.outcome.disposition
            in {
                ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
                ActionOutcomeDisposition.AMBIGUOUS,
            }
        ) or (
            self.receipt is not None
            and self.receipt.disposition is DispatchDisposition.TARGET_REJECTED
        )
        if self.grounded_target_point is not None and (
            not has_point_feedback or self.proposal.target_expression is None
        ):
            raise ValueError("A grounded target point requires targeted no-progress feedback.")
        return self

    def _validate_feedback_owner(self) -> None:
        if self.refusal is not None and self.refusal.proposal_id != self.proposal.proposal_id:
            raise ValueError("The action refusal does not match the proposal.")
        if self.receipt is not None:
            if not is_correctable_rejection(self.receipt.disposition):
                raise ValueError("Only a correctable dispatch rejection is planning feedback.")
            if (
                self.receipt.proposal_id != self.proposal.proposal_id
                or self.receipt.action is not self.proposal.action
            ):
                raise ValueError("The dispatch rejection does not match the proposal.")
        if self.outcome is not None and self.outcome.action is not self.proposal.action:
            raise ValueError("The action outcome does not match the proposal action.")
        if self.outcome is not None and (
            self.outcome.source_observation_id != self.proposal.source_observation_id
        ):
            raise ValueError("The action outcome does not match the proposal observation.")

    def _validate_destination_screen(self) -> None:
        if self.outcome is None and self.destination_screen is not None:
            raise ValueError("Only an action outcome can have a destination screen.")
        if (
            self.outcome is not None
            and self.destination_screen is not None
            and (self.destination_screen.observation_id != self.outcome.settled_observation_id)
        ):
            raise ValueError("The history destination screen does not match the outcome.")


def planning_screen_reference(observation: Observation) -> PlanningScreenReference:
    """Build the screen identity used to scope planner history and facts."""
    return PlanningScreenReference(
        observation_id=observation.observation_id,
        platform_context=observation.platform_context,
        keyboard_visible=observation.keyboard_visible,
        screen_fingerprint=observation.screen_fingerprint.value,
    )


def grounding_correction_for_history(
    proposal: ActionProposal,
    history: list[PlanningHistoryEntry],
) -> GroundingCorrection | None:
    """Resolve one declared correction from trusted no-progress history."""
    failed_proposal_id = proposal.grounding_correction_for_proposal_id
    for entry in reversed(history):
        if entry.proposal.proposal_id != failed_proposal_id:
            continue
        point = entry.grounded_target_point
        return point.correction() if point is not None else None
    return None


def build_planning_history_entry(
    proposal: ActionProposal,
    outcome: ActionOutcome,
    target_grounding: GroundingEvidence | None,
    source_observation: Observation,
    destination_observation: Observation | None,
    application_failure: ActionApplicationFailure | None = None,
) -> PlanningHistoryEntry:
    """Build one screen-scoped planner transition with trusted point evidence."""
    point_evidence = _grounded_target_point(target_grounding, outcome)
    return PlanningHistoryEntry(
        proposal=proposal,
        source_screen=planning_screen_reference(source_observation),
        destination_screen=(
            planning_screen_reference(destination_observation)
            if outcome.settled_observation_id is not None and destination_observation is not None
            else None
        ),
        outcome=outcome,
        application_failure=application_failure,
        grounded_target_point=point_evidence,
    )


def build_dispatch_rejection_history_entry(
    proposal: ActionProposal,
    receipt: DispatchReceipt,
    target_grounding: GroundingEvidence | None,
    source_observation: Observation,
) -> PlanningHistoryEntry:
    """Preserve correctable dispatch feedback and relevant point evidence."""
    return PlanningHistoryEntry(
        proposal=proposal,
        source_screen=planning_screen_reference(source_observation),
        receipt=receipt,
        grounded_target_point=(
            _grounded_target_point(target_grounding)
            if receipt.disposition is DispatchDisposition.TARGET_REJECTED
            else None
        ),
    )


def build_refusal_history_entry(
    proposal: ActionProposal,
    refusal: ActionRefusal,
    source_observation: Observation,
) -> PlanningHistoryEntry:
    """Attach a pre-dispatch refusal to the screen that owns it."""
    return PlanningHistoryEntry(
        proposal=proposal,
        source_screen=planning_screen_reference(source_observation),
        refusal=refusal,
    )


def grounding_correction_failure_reason(
    proposal: ActionProposal,
    outcome: ActionOutcome,
) -> str | None:
    """Return the terminal reason for an ineffective bounded correction."""
    if (
        proposal.grounding_correction_for_proposal_id is not None
        and outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
    ):
        return "The grounding correction produced no observable change."
    return None


def planning_history_context(
    entry: PlanningHistoryEntry,
    *,
    include_text: bool = True,
) -> dict[str, object]:
    """Render one detailed, screen-scoped recent-history edge."""
    proposal = entry.proposal
    context: dict[str, object] = _without_empty_values(
        {
            "proposal_id": proposal.proposal_id,
            "source_screen": _screen_context(entry.source_screen),
            "destination_screen": _screen_context(entry.destination_screen),
            "action": proposal.action.value,
            "grounding_correction_for_proposal_id": (proposal.grounding_correction_for_proposal_id),
            "interaction_refinement_for_proposal_id": (
                proposal.interaction_refinement_for_proposal_id
            ),
            "target_expression": proposal.target_expression,
            "destination_expression": proposal.destination_expression,
            "duration_seconds": proposal.duration_seconds,
            "direction": proposal.direction.value if proposal.direction is not None else None,
            "distance": proposal.distance.value if proposal.distance is not None else None,
        }
    )
    if include_text and proposal.text is not None:
        context["text"] = _text_context(proposal.text)
    context["feedback"] = _feedback_context(entry, include_reason=True)
    if entry.application_failure is not None:
        context["application_failure"] = planner_failure_context(
            entry.application_failure, include_summary=True
        )
    if entry.grounded_target_point is not None:
        context["grounded_target_point"] = entry.grounded_target_point.model_dump(mode="json")
    return context


def split_planning_history(
    history: tuple[PlanningHistoryEntry, ...],
) -> tuple[tuple[PlanningHistoryEntry, ...], tuple[PlanningHistoryEntry, ...]]:
    """Split history into the summarised older part and the quoted recent window.

    The boundary moves one anchor at a time so the quoted blocks stay byte-stable between steps,
    which is what keeps the prompt prefix cacheable. The window therefore holds between
    RECENT_PLANNING_HISTORY_LIMIT and one anchor more.
    """
    drift = len(history) - RECENT_PLANNING_HISTORY_LIMIT
    if drift <= 0:
        return (), history
    start = drift // PLANNING_HISTORY_ANCHOR_STEPS * PLANNING_HISTORY_ANCHOR_STEPS
    return history[:start], history[start:]


def trusted_history_content(
    history: tuple[PlanningHistoryEntry, ...],
    *,
    include_text: bool = True,
) -> tuple[TextModelContent, ...]:
    """Summarise older evidence first, then one byte-stable block per recent step."""
    older, recent = split_planning_history(history)
    blocks: list[TextModelContent] = []
    if older:
        blocks.append(
            TextModelContent(
                text="Earlier history summary:\n"
                + json.dumps(planning_history_summary(older), sort_keys=True, separators=(",", ":"))
            )
        )
    blocks.extend(
        TextModelContent(
            text="Trusted history event:\n"
            + json.dumps(
                planning_history_context(entry, include_text=include_text),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for entry in recent
    )
    return tuple(blocks)


def planning_history_summary(entries: tuple[PlanningHistoryEntry, ...]) -> PlanningHistorySummary:
    """Group older evidence and report how many distinct groups were omitted."""
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        context = _compact_history_context(entry)
        key = json.dumps(context, sort_keys=True, separators=(",", ":"))
        existing_entry = grouped.get(key)
        if existing_entry is None:
            grouped[key] = {**context, "count": 1}
        else:
            count = existing_entry["count"]
            if not isinstance(count, int):
                raise TypeError("A compact history count must be an integer.")
            existing_entry["count"] = count + 1
    selected_newest_first: list[dict[str, object]] = []
    grouped_summaries = list(grouped.values())
    for summary in reversed(grouped_summaries[-EARLIER_HISTORY_SUMMARY_LIMIT:]):
        prospective = [*selected_newest_first, summary]
        rendered = {
            "entries": list(reversed(prospective)),
            "omitted_groups": len(grouped_summaries) - len(prospective),
        }
        if len(json.dumps(rendered, sort_keys=True, separators=(",", ":"))) > (
            EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT
        ):
            break
        selected_newest_first = prospective
    return {
        "entries": list(reversed(selected_newest_first)),
        "omitted_groups": len(grouped_summaries) - len(selected_newest_first),
    }


def _compact_history_context(entry: PlanningHistoryEntry) -> dict[str, object]:
    proposal = entry.proposal
    context: dict[str, object] = {"action": proposal.action.value}
    if entry.source_screen is not None:
        context["source_screen"] = _screen_context(entry.source_screen, include_fingerprint=True)
    if entry.destination_screen is not None:
        context["destination_screen"] = _screen_context(
            entry.destination_screen, include_fingerprint=True
        )
    for field_name, value in (
        ("target_expression", proposal.target_expression),
        ("destination_expression", proposal.destination_expression),
        ("direction", proposal.direction.value if proposal.direction is not None else None),
        ("distance", proposal.distance.value if proposal.distance is not None else None),
    ):
        if value is not None:
            context[field_name] = value
    context["feedback"] = _feedback_context(entry, include_reason=False)
    if entry.application_failure is not None:
        context["application_failure"] = planner_failure_context(
            entry.application_failure, include_summary=False
        )
    return context


def _feedback_context(entry: PlanningHistoryEntry, *, include_reason: bool) -> dict[str, object]:
    if entry.refusal is not None:
        context: dict[str, object] = {
            "kind": "refusal",
            "code": entry.refusal.code.value,
        }
        if include_reason:
            context["reason"] = entry.refusal.reason
        return context
    if entry.receipt is not None:
        return {
            "kind": "receipt",
            "disposition": entry.receipt.disposition.value,
            "reason": entry.receipt.reason,
        }
    if entry.outcome is None:
        raise ValueError("Planning history has no feedback.")
    context = {
        "kind": "outcome",
        "disposition": entry.outcome.disposition.value,
        "signals": sorted(signal.value for signal in entry.outcome.signals),
    }
    if entry.outcome.scroll_exhausted is not None:
        context["scroll_exhausted"] = entry.outcome.scroll_exhausted
    if include_reason and entry.outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
        context["reason"] = entry.outcome.reason
    return context


def _screen_context(
    screen: PlanningScreenReference | None,
    *,
    include_fingerprint: bool = False,
) -> dict[str, object] | None:
    """Render screen identity without identifiers the planner cannot resolve."""
    if screen is None:
        return None
    context: dict[str, object] = {
        "platform_context": screen.platform_context,
        "keyboard_visible": screen.keyboard_visible,
    }
    if include_fingerprint:
        context["screen_fingerprint"] = screen.screen_fingerprint
    return context


def correctable_grounding_proposals(
    history: tuple[PlanningHistoryEntry, ...],
) -> dict[str, tuple[str, str | None]]:
    """Map each failed grounded target that still allows one correction attempt."""
    corrected = {
        entry.proposal.grounding_correction_for_proposal_id
        for entry in history
        if entry.proposal.grounding_correction_for_proposal_id is not None
    }
    return {
        entry.proposal.proposal_id: (
            entry.proposal.action.value,
            entry.proposal.target_expression,
        )
        for entry in history
        if entry.grounded_target_point is not None
        and entry.proposal.grounding_correction_for_proposal_id is None
        and entry.proposal.proposal_id not in corrected
    }


def planning_screen_fact_context(fact: PlanningScreenFact) -> dict[str, object]:
    """Render one screen-scoped fact without identifiers the planner cannot use."""
    return {
        "screen": _screen_context(fact.screen),
        "fact": fact.fact,
        "supersedes_recovery_guidance": fact.supersedes_recovery_guidance,
    }


def _without_empty_values(context: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in context.items() if value is not None}


def _text_context(text: LiteralText | DeclaredValueReference) -> dict[str, object]:
    if isinstance(text, DeclaredValueReference):
        return {"source": text.source, "declared_value_name": text.value_name}
    return {"source": text.source, "literal_value_retained": False}


def _grounded_target_point(
    target_grounding: GroundingEvidence | None,
    outcome: ActionOutcome | None = None,
) -> GroundedTargetPoint | None:
    if target_grounding is None or not isinstance(target_grounding.result.location, GroundedPoint):
        return None
    if outcome is not None and outcome.disposition not in {
        ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        ActionOutcomeDisposition.AMBIGUOUS,
    }:
        return None
    screenshot = target_grounding.request.screenshot
    return GroundedTargetPoint(
        point=target_grounding.result.location.pixel,
        screenshot_width_px=screenshot.width_px,
        screenshot_height_px=screenshot.height_px,
    )


__all__ = [
    "correctable_grounding_proposals",
    "EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT",
    "EARLIER_HISTORY_SUMMARY_LIMIT",
    "GroundedTargetPoint",
    "PlanningHistoryEntry",
    "PlanningHistorySummary",
    "PlanningScreenFact",
    "PlanningScreenReference",
    "build_planning_history_entry",
    "build_refusal_history_entry",
    "build_dispatch_rejection_history_entry",
    "grounding_correction_failure_reason",
    "planning_history_context",
    "planning_history_summary",
    "planning_screen_reference",
]
