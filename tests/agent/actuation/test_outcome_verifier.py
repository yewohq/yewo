"""The post-dispatch verifier may weaken a pixel-only claim and never strengthen evidence."""

from yewo.agent.actuation import ActionOutcomeSignal
from yewo.agent.actuation.outcome_verifier import (
    OutcomeAssessment,
    OutcomeVerdict,
    apply_outcome_assessment,
    needs_outcome_verification,
)
from yewo.agent.actuation.outcomes import ActionOutcome, ActionOutcomeDisposition
from yewo.artifacts import ArtifactRetention
from yewo.agent.actuation import ActionKind


def _outcome(disposition: ActionOutcomeDisposition, *signals: ActionOutcomeSignal) -> ActionOutcome:
    return ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.TAP,
        source_observation_id="obs_1",
        settled_observation_id="obs_2",
        disposition=disposition,
        signals=frozenset(signals),
        reason="The settled observation contains action-outcome signals.",
        retention=ArtifactRetention.RUN,
    )


def _assessment(verdict: OutcomeVerdict) -> OutcomeAssessment:
    return OutcomeAssessment(
        assessment_id="assessment-1",
        authorization_id="authorization-1",
        verdict=verdict,
        reason="The Home tab is still selected and no Favourites list is visible.",
    )


def test_a_bare_pixel_difference_is_sent_for_a_second_opinion() -> None:
    # Pixels differing is not evidence the action did what it predicted.
    assert needs_outcome_verification(
        _outcome(ActionOutcomeDisposition.OBSERVED, ActionOutcomeSignal.PIXELS_CHANGED)
    )


def test_an_unsettled_dispatch_is_sent_for_a_second_opinion() -> None:
    # Text that may not have landed, or a dispatch the device could not confirm, leaves the
    # prediction open in exactly the way a reader of the two screens can close.
    assert needs_outcome_verification(_outcome(ActionOutcomeDisposition.AMBIGUOUS))


def test_a_state_change_is_evidence_enough_on_its_own() -> None:
    assert not needs_outcome_verification(
        _outcome(
            ActionOutcomeDisposition.OBSERVED,
            ActionOutcomeSignal.PIXELS_CHANGED,
            ActionOutcomeSignal.CONTEXT_CHANGED,
        )
    )


def test_an_unchanged_screen_needs_no_second_opinion() -> None:
    assert not needs_outcome_verification(_outcome(ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE))


def test_a_contradicted_pixel_change_stops_being_an_observed_outcome() -> None:
    observed = _outcome(ActionOutcomeDisposition.OBSERVED, ActionOutcomeSignal.PIXELS_CHANGED)

    revised = apply_outcome_assessment(observed, _assessment(OutcomeVerdict.NOT_HAPPENED))

    assert revised.disposition is ActionOutcomeDisposition.AMBIGUOUS
    assert "not as the action predicted" in revised.reason
    assert "Home tab is still selected" in revised.reason


def test_a_confirmed_outcome_is_left_exactly_as_the_device_reported_it() -> None:
    observed = _outcome(ActionOutcomeDisposition.OBSERVED, ActionOutcomeSignal.PIXELS_CHANGED)

    assert apply_outcome_assessment(observed, _assessment(OutcomeVerdict.HAPPENED)) == observed


def test_an_unreadable_pair_cannot_weaken_the_device_record() -> None:
    observed = _outcome(ActionOutcomeDisposition.OBSERVED, ActionOutcomeSignal.PIXELS_CHANGED)

    assert apply_outcome_assessment(observed, _assessment(OutcomeVerdict.INDETERMINATE)) == observed


def test_the_verifier_cannot_promote_an_outcome_the_device_did_not_record() -> None:
    # A model reading screenshots must not outrank device evidence. It may describe the screens.
    ambiguous = _outcome(ActionOutcomeDisposition.AMBIGUOUS)

    revised = apply_outcome_assessment(ambiguous, _assessment(OutcomeVerdict.HAPPENED))

    assert revised.disposition is ActionOutcomeDisposition.AMBIGUOUS
    assert revised.signals == ambiguous.signals


def test_the_verifier_sends_both_screens_and_its_profile_can_carry_them() -> None:
    # The request is built from two screenshots, so a profile that admits one image would have
    # the role rejecting its own request at the provider boundary.
    from yewo.models import ModelRole
    from yewo.models.profiles import minimum_image_count_for_role

    images_in_request = 2

    assert minimum_image_count_for_role(ModelRole.OUTCOME_VERIFIER) >= images_in_request


def test_an_unsettled_outcome_keeps_what_the_screens_showed() -> None:
    # The gate pays for this call, so its answer has to reach the planner even where the
    # disposition cannot move.
    ambiguous = _outcome(ActionOutcomeDisposition.AMBIGUOUS)

    revised = apply_outcome_assessment(ambiguous, _assessment(OutcomeVerdict.NOT_HAPPENED))

    assert revised.disposition is ActionOutcomeDisposition.AMBIGUOUS
    # The planner reads an outcome reason as trusted runtime feedback, so a model's reading of
    # two screenshots has to name itself rather than arrive as device fact.
    assert revised.reason.startswith("A model comparing the two screens reports")
    assert "Home tab is still selected" in revised.reason
    assert ambiguous.reason in revised.reason


def test_an_unsettled_outcome_records_a_confirmed_prediction_too() -> None:
    ambiguous = _outcome(ActionOutcomeDisposition.AMBIGUOUS)

    revised = apply_outcome_assessment(ambiguous, _assessment(OutcomeVerdict.HAPPENED))

    assert revised.reason.startswith("A model comparing the two screens reports")
    assert "happened" in revised.reason


def test_an_unsettled_outcome_is_told_nothing_when_the_screens_are_unreadable() -> None:
    # Indeterminate means the pair could not settle the question. Reporting it as a negative
    # finding would invent evidence against the action.
    ambiguous = _outcome(ActionOutcomeDisposition.AMBIGUOUS)

    revised = apply_outcome_assessment(ambiguous, _assessment(OutcomeVerdict.INDETERMINATE))

    assert revised.reason == ambiguous.reason
