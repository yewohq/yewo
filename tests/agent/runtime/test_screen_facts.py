"""Screen-scoped deterministic planning facts."""

from yewo.agent.actuation import ActionKind, ActionRefusal, ActionRefusalCode
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.history import MAXIMUM_SCREEN_FACT_CHARACTERS
from yewo.agent.runtime.no_progress import ActionNoProgressTracker

from agent.actuation.support import proposal
from agent.runtime.test_agent import _observation


def test_a_long_conditional_refusal_is_bounded_as_a_screen_fact() -> None:
    tracker = ActionNoProgressTracker(ScreenComparisonPolicy(maximum_phash_distance=0))
    screen = _observation("obs_1", "0000000000000000")
    tracker.start(screen)
    advance = proposal(
        ActionKind.TAP,
        proposal_id="proposal-advance",
        source_observation_id=screen.observation_id,
        target_expression="the save control",
    )

    tracker.record_refusal(
        advance,
        screen,
        ActionRefusal(
            proposal_id=advance.proposal_id,
            code=ActionRefusalCode.COMPLETED_WORK_REGRESSION,
            reason="Detailed verifier explanation. " * 100,
        ),
    )

    fact = tracker.screen_facts(screen)[-1].fact
    assert len(fact) == MAXIMUM_SCREEN_FACT_CHARACTERS
    assert fact.startswith("Resolve this before you repeat tap:")
