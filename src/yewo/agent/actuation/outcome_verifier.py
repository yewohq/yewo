"""Judge whether the outcome one action predicted is the outcome the screen shows."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from yewo.agent.actuation.action_catalog import ActionKind, ActionOutcomeSignal
from yewo.agent.actuation.outcomes import ActionOutcome, ActionOutcomeDisposition
from yewo.agent.actuation.outcome_verifier_prompt import (
    OUTCOME_VERIFIER_SYSTEM_PROMPT,
    OUTCOME_VERIFIER_USER_PROMPT,
)
from yewo.agent.actuation.records import AuthorizedAction
from yewo.agent.actuation.target_marker import draw_target_marker
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)
from yewo.observations import Observation, ScreenshotReference
from yewo.observations.errors import ObservationCaptureError
from yewo.observations.store import ScreenshotStore


# Matched to the other reasoning roles. The answer itself is one verdict and one sentence, but
# private reasoning draws on the same allowance, and a truncated object re-sends both screenshots
# on every validation attempt.
OUTCOME_VERIFIER_MAX_COMPLETION_TOKENS = 8_192


class OutcomeVerdict(StrEnum):
    """Whether the screen shows the outcome the planner predicted."""

    HAPPENED = "happened"
    NOT_HAPPENED = "not_happened"
    INDETERMINATE = "indeterminate"


class OutcomeVerification(BaseModel):
    """One model judgement about a predicted outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: OutcomeVerdict
    reason: str = Field(min_length=1, max_length=1_000)


class OutcomeAssessment(BaseModel):
    """A verdict bound to the action it judged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_id: RecordIdentifier
    authorization_id: RecordIdentifier
    verdict: OutcomeVerdict
    reason: str = Field(min_length=1)


class OutcomeVerifier(Protocol):
    """Decide whether a dispatched action produced what it predicted."""

    def verify(
        self,
        action: AuthorizedAction,
        before_screenshot: ScreenshotReference,
        after: Observation,
    ) -> OutcomeAssessment: ...


class ModelOutcomeVerifier:
    """Compare the dispatch-time screen against the settled screen with a dedicated profile."""

    def __init__(
        self,
        model_client: ModelClient,
        model_profile_id: str,
        screenshot_store: ScreenshotStore,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._model_client = model_client
        self._model_profile_id = model_profile_id
        self._screenshot_store = screenshot_store
        self._identifier_factory = identifier_factory

    def verify(
        self,
        action: AuthorizedAction,
        before_screenshot: ScreenshotReference,
        after: Observation,
    ) -> OutcomeAssessment:
        """Return an assessment without treating unavailable evidence as action failure."""
        assessment_id = self._identifier_factory()
        marked = self._marked_source(action, before_screenshot)
        context = {
            "action": action.action.value,
            "expected_outcome": action.expected_outcome,
            "target_marked": marked is not before_screenshot,
        }
        request = ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=self._model_profile_id,
            role=ModelRole.OUTCOME_VERIFIER,
            max_completion_tokens=OUTCOME_VERIFIER_MAX_COMPLETION_TOKENS,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text=OUTCOME_VERIFIER_SYSTEM_PROMPT),),
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text=OUTCOME_VERIFIER_USER_PROMPT),),
                    cache_prefix_end=True,
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(
                        TextModelContent(
                            text="Dispatched action:\n"
                            + json.dumps(context, sort_keys=True, separators=(",", ":"))
                        ),
                        TextModelContent(text="The screen before the action:"),
                        ImageModelContent(screenshot=marked),
                        TextModelContent(text="The screen after the action settled:"),
                        ImageModelContent(screenshot=after.screenshot),
                    ),
                ),
            ),
            response_contract=build_model_response_contract(
                "outcome-verification-v1",
                OutcomeVerification.model_json_schema(),
            ),
            retention=after.screenshot.retention,
        )
        response = self._model_client.complete(request, OutcomeVerification)
        if response.disposition is not ModelCallDisposition.COMPLETED:
            return self._indeterminate(
                assessment_id,
                action,
                response.reason or "The outcome-verifier model call did not complete.",
            )
        try:
            verification = OutcomeVerification.model_validate(response.structured_output)
        except ValidationError:
            return self._indeterminate(
                assessment_id,
                action,
                "The outcome-verifier response was invalid.",
            )
        return OutcomeAssessment(
            assessment_id=assessment_id,
            authorization_id=action.authorization_id,
            verdict=verification.verdict,
            reason=verification.reason,
        )

    def _marked_source(
        self,
        action: AuthorizedAction,
        before_screenshot: ScreenshotReference,
    ) -> ScreenshotReference:
        """Persist a copy of the source screen carrying the point the action was aimed at.

        A separate artifact, because a stored screenshot is verified against the digest its
        reference records and marking the stored bytes would make that reference unreadable.
        """
        if action.target is None or action.action is ActionKind.WAIT:
            return before_screenshot
        try:
            marked = draw_target_marker(
                self._screenshot_store.load(before_screenshot),
                action.target,
            )
            return self._screenshot_store.persist(
                marked,
                retention=before_screenshot.retention,
            )
        except (ObservationCaptureError, OSError):
            return before_screenshot

    @staticmethod
    def _indeterminate(
        assessment_id: str,
        action: AuthorizedAction,
        reason: str,
    ) -> OutcomeAssessment:
        return OutcomeAssessment(
            assessment_id=assessment_id,
            authorization_id=action.authorization_id,
            verdict=OutcomeVerdict.INDETERMINATE,
            reason=reason,
        )


OUTCOME_VERIFICATION_MODEL_REQUESTS = 1


def needs_outcome_verification(outcome: ActionOutcome) -> bool:
    """Whether pixel and platform evidence left the predicted outcome unsettled.

    A bare pixel difference says the screen is not identical. It does not say the action did what
    it predicted, and an ambiguous dispatch says nothing either way. Every other disposition
    already rests on evidence a second opinion cannot improve.
    """
    if outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS:
        return True
    return (
        outcome.disposition is ActionOutcomeDisposition.OBSERVED
        and outcome.signals == frozenset({ActionOutcomeSignal.PIXELS_CHANGED})
    )


def apply_outcome_assessment(
    outcome: ActionOutcome,
    assessment: OutcomeAssessment,
) -> ActionOutcome:
    """Weaken an outcome the screens contradict, and never strengthen one.

    The verifier reads the same screenshots the runtime already holds, so it can show that a
    change was not the predicted one. It cannot establish an effect that device evidence did not
    record, and promoting on its word alone would let a model outrank the device.
    """
    if assessment.verdict is OutcomeVerdict.INDETERMINATE:
        return outcome
    if outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS:
        # The disposition already says the evidence is unsettled. What the planner lacks is any
        # account of the two screens, so the verdict is worth keeping even where it changes
        # nothing about how the runtime treats the step.
        return outcome.model_copy(
            update={"reason": _format_verdict_attribution(assessment) + outcome.reason}
        )
    if (
        outcome.disposition is not ActionOutcomeDisposition.OBSERVED
        or assessment.verdict is not OutcomeVerdict.NOT_HAPPENED
    ):
        return outcome
    return outcome.model_copy(
        update={
            "disposition": ActionOutcomeDisposition.AMBIGUOUS,
            "reason": (
                "The screen changed, but not as the action predicted. "
                "A model comparing the two screens reports: " + assessment.reason
            ),
        }
    )


def _format_verdict_attribution(assessment: OutcomeAssessment) -> str:
    """Name the verdict's source, because the planner treats an outcome reason as device fact.

    The disposition still rests on device evidence. Stating the verdict unattributed would put a
    model's reading of two screenshots into the planner's trusted runtime feedback.
    """
    opening = (
        "A model comparing the two screens reports that the predicted outcome happened. "
        if assessment.verdict is OutcomeVerdict.HAPPENED
        else "A model comparing the two screens reports that the predicted outcome did not happen. "
    )
    return opening + assessment.reason + " "


__all__ = [
    "OUTCOME_VERIFIER_MAX_COMPLETION_TOKENS",
    "ModelOutcomeVerifier",
    "OutcomeAssessment",
    "OutcomeVerdict",
    "OutcomeVerification",
    "OutcomeVerifier",
    "apply_outcome_assessment",
    "needs_outcome_verification",
]
