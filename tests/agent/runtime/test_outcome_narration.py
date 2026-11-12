"""What the run says about an outcome a second reader looked at."""

from pathlib import Path

from yewo.agent.actuation import (
    ActionKind,
    ActionSource,
    AuthorizedAction,
    SideEffectClass,
)
from yewo.agent.actuation.outcome_verifier import OutcomeAssessment, OutcomeVerdict
from yewo.agent.runtime.settlement import _verified_prediction
from yewo.agent.actuation.records import PixelPoint
from yewo.artifacts import ArtifactRetention
from yewo.observations import Observation, ScreenFingerprint, ScreenshotReference
from yewo.run_narration import RunNarrator, narration_scope

from agent.runtime.test_agent import _mode_policy, _observation
from agent.runtime.test_ambiguous_outcomes import _ambiguous_focus


class _Verifier:
    def __init__(self, verdict: OutcomeVerdict) -> None:
        self._verdict = verdict
        self.calls = 0

    def verify(
        self,
        action: AuthorizedAction,
        before_screenshot: ScreenshotReference,
        after: Observation,
    ) -> OutcomeAssessment:
        self.calls += 1
        return OutcomeAssessment(
            assessment_id="assessment-1",
            authorization_id="authorization-ambiguous",
            verdict=self._verdict,
            reason="Both screens show the field holding the typed value.",
        )


def _authorization(screen: Observation) -> AuthorizedAction:
    return AuthorizedAction(
        authorization_id="authorization-ambiguous",
        proposal_id="proposal-focus",
        source_observation_id=screen.observation_id,
        source_screenshot_sha256=screen.screenshot.sha256,
        application_id="dev.yewo.controlled",
        action=ActionKind.FOCUS,
        source=ActionSource.PLANNER,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        expected_outcome="The field takes focus.",
        target=PixelPoint(x=100, y=200),
        retention=ArtifactRetention.RUN,
    )


def _narration(
    tmp_path: Path,
    verdict: OutcomeVerdict,
    baseline: ScreenFingerprint | None = None,
) -> str:
    screen = _observation("obs_1", "0000000000000000")
    with narration_scope(RunNarrator(tmp_path)):
        _verified_prediction(
            _Verifier(verdict),
            _authorization(screen),
            screen,
            screen,
            _ambiguous_focus(screen),
            7,
            _mode_policy(),
            0,
            baseline,
        )
    log = tmp_path / "runtime/yewo-session.log"
    # A skipped verification narrates nothing, so the file need not exist.
    return log.read_text() if log.exists() else ""


def test_confirmed_outcome_is_not_missing(tmp_path: Path) -> None:
    """An ambiguous outcome is revised for every verdict, this one included.

    Keying the message on whether the outcome object changed therefore reported the predicted
    outcome missing in the one case where the reader had just confirmed it.
    """
    narration = _narration(tmp_path, OutcomeVerdict.HAPPENED)

    assert "did not produce" not in narration
    assert "Step 7" in narration


def test_contradicted_outcome_is_announced(tmp_path: Path) -> None:
    """The message the fix must not silence."""
    narration = _narration(tmp_path, OutcomeVerdict.NOT_HAPPENED)

    assert "Step 7 did not produce the outcome it predicted." in narration


def test_a_screen_that_moved_before_dispatch_is_not_verified(tmp_path: Path) -> None:
    """The only `before` image available is the one planning read.

    Planning, grounding and authorization each take model calls. When the dispatch baseline
    says the screen changed across them, that image describes a transition the action did not
    make, and a verdict about it would be about the wrong pair of screens.
    """
    screen = _observation("obs_1", "0000000000000000")
    moved = screen.presentation_key.fingerprint.model_copy(
        update={"masked_pixels_sha256": "b" * 64}
    )

    narration = _narration(tmp_path, OutcomeVerdict.NOT_HAPPENED, moved)

    assert "did not produce" not in narration


def test_a_screen_that_held_still_is_verified(tmp_path: Path) -> None:
    """The same call with a baseline matching the planning screen."""
    screen = _observation("obs_1", "0000000000000000")

    narration = _narration(
        tmp_path, OutcomeVerdict.NOT_HAPPENED, screen.presentation_key.fingerprint
    )

    assert "Step 7 did not produce the outcome it predicted." in narration


def test_a_run_out_of_model_requests_does_not_spend_one_on_verification() -> None:
    """Capacity is reserved before the call, as every other model call reserves it.

    Counting only answered calls lets a run pass the ceiling its own budget report states.
    """
    screen = _observation("obs_1", "0000000000000000")
    verifier = _Verifier(OutcomeVerdict.NOT_HAPPENED)
    outcome = _ambiguous_focus(screen)

    revised, requests = _verified_prediction(
        verifier,
        _authorization(screen),
        screen,
        screen,
        outcome,
        7,
        _mode_policy(maximum_model_requests=4),
        4,
    )

    assert verifier.calls == 0
    assert requests == 0
    assert revised is outcome


def test_a_run_with_one_model_request_left_still_verifies() -> None:
    """The reservation must not refuse the request the ceiling still allows."""
    screen = _observation("obs_1", "0000000000000000")
    verifier = _Verifier(OutcomeVerdict.NOT_HAPPENED)

    _, requests = _verified_prediction(
        verifier,
        _authorization(screen),
        screen,
        screen,
        _ambiguous_focus(screen),
        7,
        _mode_policy(maximum_model_requests=4),
        3,
    )

    assert verifier.calls == 1
    assert requests == 1
