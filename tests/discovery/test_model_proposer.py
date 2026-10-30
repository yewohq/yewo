"""Where the journey's intent meets a real screen, and what it may not decide.

The model is faked at the `LLMClient` boundary production already has. What is
under test is what reaches the prompt, what is refused, and what the spend
report charges.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from perception_fakes import detected, frame, read

from yewo.agent.actuation.action_models import Coordinates, PlannerTapAction
from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgets,
    DiscoveryRun,
    JourneyGoal,
    SideEffectClass,
)
from yewo.discovery.model_proposer import (
    MAXIMUM_COST_MICRO_USD,
    MODEL_CALLS_PER_PROPOSAL,
    UNNAMED_CONTROL,
    ModelDiscoveryProposer,
    ProposalSpend,
    ControlChoice,
    GoalJudgement,
)
from yewo.discovery.runtime import DiscoveryActionCandidate
from yewo.utils.secrets import SecretScrubber

SCREEN = (400, 800)
GOAL = JourneyGoal(goal_id="goal-1", statement="A user places an order.")


class _Perception:
    def __init__(self, current_frame):
        self.current_frame = current_frame


def _proposer(answers, current_frame=None, secrets=None):
    client = SimpleNamespace(generate_response=Mock(side_effect=list(answers)))
    scrubber = SecretScrubber()
    if secrets:
        scrubber.learn_secrets(secrets)
    spend = ProposalSpend()
    proposer = ModelDiscoveryProposer(
        client,
        _Perception(current_frame),
        scrubber,
        spend,
    )
    return proposer, client, spend


def _candidate(x=100, y=130, goal_ids=("goal-1",)):
    action = PlannerTapAction(coordinates=Coordinates(x=x, y=y), reason="explore")
    return DiscoveryActionCandidate(
        source_observation_id="observation-1",
        source_screen_state_id="state-1",
        control_identity=f"control-{x}-{y}",
        relevant_goal_ids=goal_ids,
        action=action,
        side_effect=SideEffectClass.REVERSIBLE,
        reversible=True,
        source_phash="0" * 16,
        screen_size=SCREEN,
    )


def _run(goals=(GOAL,)):
    return SimpleNamespace(goals=goals)


def _screen(*elements, lines=()):
    return frame(elements=tuple(elements), text_lines=tuple(lines))


def _prompt(client, call=0):
    return client.generate_response.call_args_list[call].kwargs["messages"][0].content


def test_the_chosen_control_is_named_by_its_frontier_identity():
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, _ = _proposer([ControlChoice(control=1)], current)
    candidate = _candidate()

    receipt = proposer.select(_run(), (candidate,))

    assert receipt.selected_frontier_id == candidate.frontier_id


def test_the_open_goal_and_every_offered_control_reach_the_prompt():
    current = _screen(
        detected((20, 100, 180, 160), text="Basket"),
        detected((20, 300, 180, 360), text="Settings"),
    )
    proposer, client, _ = _proposer([ControlChoice(control=1)], current)

    proposer.select(_run(), (_candidate(100, 130), _candidate(100, 330)))

    prompt = _prompt(client)
    assert "A user places an order." in prompt
    assert "1. Basket" in prompt
    assert "2. Settings" in prompt


def test_a_control_number_the_proposer_never_offered_is_refused():
    """The controller re-checks too. This refuses before spending an action."""
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, _ = _proposer([ControlChoice(control=7)], current)

    receipt = proposer.select(_run(), (_candidate(),))

    assert receipt.selected_frontier_id is None


def test_a_model_choosing_nothing_selects_nothing():
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, _ = _proposer([ControlChoice(control=None)], current)

    receipt = proposer.select(_run(), (_candidate(),))

    assert receipt.selected_frontier_id is None
    assert not receipt.provider_failed


def test_a_provider_failure_selects_nothing_rather_than_guessing():
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, _ = _proposer([RuntimeError("the provider refused")], current)

    receipt = proposer.select(_run(), (_candidate(),))

    assert receipt.selected_frontier_id is None
    assert receipt.provider_failed


def test_a_candidate_the_current_frame_cannot_place_stops_the_proposal():
    """A point in no detection means this frame is not the frame that named it."""
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, client, _ = _proposer([ControlChoice(control=1)], current)

    receipt = proposer.select(_run(), (_candidate(x=390, y=790),))

    assert receipt.selected_frontier_id is None
    client.generate_response.assert_not_called()


def test_no_frame_at_all_stops_the_proposal_without_a_model_call():
    proposer, client, _ = _proposer([ControlChoice(control=1)], None)

    assert proposer.select(_run(), (_candidate(),)).selected_frontier_id is None
    client.generate_response.assert_not_called()


def test_a_settled_goal_is_not_offered_for_choice():
    settled = GOAL.activate().close(CapabilityState.REACHED, ("observation-1",))
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, client, _ = _proposer([ControlChoice(control=1)], current)

    receipt = proposer.select(_run(goals=(settled,)), (_candidate(),))

    assert receipt.selected_frontier_id is None
    client.generate_response.assert_not_called()


def test_a_band_wrapping_a_title_does_not_borrow_its_name():
    """An app bar reads as itself, and the heading inside it keeps its name.

    The bar read as the heading, because the heading was the smallest
    detection around the bar's centre. The heading answered the journey, so
    the model chose the bar, the tap landed on text, and the screen did not
    move.
    """
    current = _screen(
        detected((0, 52, 400, 116), name="View"),
        detected((33, 72, 167, 100), name="Text", text="Trainings Schedule"),
    )
    proposer, client, _ = _proposer([ControlChoice(control=2)], current)

    proposer.select(_run(), (_candidate(x=200, y=84), _candidate(x=100, y=86)))

    prompt = _prompt(client)
    assert f"1. {UNNAMED_CONTROL}" in prompt
    assert "2. Trainings Schedule" in prompt


def test_a_control_with_no_text_is_still_offered_as_one():
    current = _screen(detected((20, 100, 180, 160)))
    proposer, client, _ = _proposer([ControlChoice(control=1)], current)

    proposer.select(_run(), (_candidate(),))

    assert f"1. {UNNAMED_CONTROL}" in _prompt(client)


def test_a_declared_secret_on_screen_does_not_reach_the_prompt():
    current = _screen(
        detected((20, 100, 180, 160), text="4242424242424242"),
        lines=(read((20, 100, 180, 160), "4242424242424242"),),
    )
    proposer, client, _ = _proposer(
        [ControlChoice(control=1)],
        current,
        secrets={"card": "4242424242424242"},
    )

    proposer.select(_run(), (_candidate(),))

    prompt = _prompt(client)
    assert "4242424242424242" not in prompt
    assert "{{card}}" in prompt


def test_a_judged_goal_is_returned_by_its_own_identity():
    current = _screen(lines=(read((20, 40, 300, 80), "Order confirmed"),))
    proposer, _, _ = _proposer([GoalJudgement(satisfied_goals=[1])])

    assert proposer.satisfied_goal_ids((GOAL,), current) == ("goal-1",)


def test_the_platform_clock_is_not_part_of_the_question():
    """A clock in the prompt makes two readings of one screen two questions.

    Nothing is then cached between them, and no two runs of one journey can
    be compared. A live run put the time in the model's own reasoning.
    """
    current = _screen(
        lines=(
            read((60, 4, 150, 20), "8:46"),
            read((20, 40, 300, 80), "Order confirmed"),
        )
    )
    proposer, client, _ = _proposer([GoalJudgement(satisfied_goals=[1])])

    proposer.satisfied_goal_ids((GOAL,), current, ((0, 0, 400, 24),))

    prompt = _prompt(client)
    assert "8:46" not in prompt
    assert "Order confirmed" in prompt


def test_a_goal_number_outside_the_list_is_discarded():
    current = _screen(lines=(read((20, 40, 300, 80), "Order confirmed"),))
    proposer, _, _ = _proposer([GoalJudgement(satisfied_goals=[0, 2, 1, 1])])

    assert proposer.satisfied_goal_ids((GOAL,), current) == ("goal-1",)


def test_a_screen_that_reads_as_nothing_is_not_judged():
    proposer, client, _ = _proposer([GoalJudgement(satisfied_goals=[1])])

    assert proposer.satisfied_goal_ids((GOAL,), _screen()) == ()
    client.generate_response.assert_not_called()


def test_no_open_goal_is_not_judged():
    current = _screen(lines=(read((20, 40, 300, 80), "Order confirmed"),))
    proposer, client, _ = _proposer([GoalJudgement(satisfied_goals=[1])])

    assert proposer.satisfied_goal_ids((), current) == ()
    client.generate_response.assert_not_called()


def test_a_provider_failure_judges_nothing():
    current = _screen(lines=(read((20, 40, 300, 80), "Order confirmed"),))
    proposer, _, _ = _proposer([RuntimeError("the provider refused")])

    assert proposer.satisfied_goal_ids((GOAL,), current) == ()


def test_the_judging_call_is_charged_on_the_next_proposal():
    """Otherwise judging would be model spend no budget ever saw."""
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, spend = _proposer(
        [GoalJudgement(satisfied_goals=[]), ControlChoice(control=1)],
        current,
    )
    spend.record(SimpleNamespace(cost=0.001))

    proposer.satisfied_goal_ids((GOAL,), _screen(lines=(read((0, 0, 10, 10), "x"),)))
    spend.record(SimpleNamespace(cost=0.002))
    receipt = proposer.select(_run(), (_candidate(),))

    assert receipt.model_calls == 2
    assert receipt.cost_micro_usd == 3000


def test_overspend_is_reported_rather_than_clamped_to_the_ceiling():
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, spend = _proposer([ControlChoice(control=1)], current)
    for _ in range(9):
        spend.record(SimpleNamespace(cost=1.0))

    receipt = proposer.select(_run(), (_candidate(),))

    assert receipt.model_calls == 9 > MODEL_CALLS_PER_PROPOSAL
    assert receipt.cost_micro_usd == 9_000_000 > MAXIMUM_COST_MICRO_USD


def test_budgeting_rejects_a_proposal_report_above_its_ceiling():
    from yewo.discovery import budgeting

    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, spend = _proposer([ControlChoice(control=1)], current)
    spend.record(SimpleNamespace(cost=1.0))
    receipt = proposer.select(_run(), (_candidate(),))

    _, error = budgeting.record_proposal(_budgeted_run(), receipt, proposer)

    assert error is DiscoveryBlocker.INFRASTRUCTURE


def test_spend_a_run_ended_before_reporting_is_still_drained():
    """A run ending on the judgement that met its goal has no next proposal."""
    proposer, _, spend = _proposer([])
    spend.record(SimpleNamespace(cost=0.003))

    assert proposer.drain_unreported_spend() == (1, 3000)
    assert proposer.drain_unreported_spend() == (0, 0)


def test_spend_is_drained_so_one_call_is_charged_once():
    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, _, spend = _proposer(
        [ControlChoice(control=1), ControlChoice(control=1)], current
    )
    spend.record(SimpleNamespace(cost=0.001))

    first = proposer.select(_run(), (_candidate(),))
    second = proposer.select(_run(), (_candidate(),))

    assert (first.model_calls, first.cost_micro_usd) == (1, 1000)
    assert (second.model_calls, second.cost_micro_usd) == (0, 0)


def test_a_call_reporting_no_cost_is_counted_without_inventing_one():
    proposer, _, spend = _proposer([])
    spend.record(SimpleNamespace(cost=None))

    assert spend.drain() == (1, 0)


def test_a_candidate_whose_action_has_no_point_is_refused():
    """It cannot be read off a screen, so no choice against it addresses it."""
    from yewo.agent.actuation.actions import WaitAction

    current = _screen(detected((20, 100, 180, 160), text="Basket"))
    proposer, client, _ = _proposer([ControlChoice(control=1)], current)
    pointless = _candidate()
    pointless = type(pointless)(
        **{
            **{field: getattr(pointless, field) for field in pointless.__dataclass_fields__},
            "action": WaitAction(duration=1, reason="wait"),
        }
    )

    assert proposer.select(_run(), (pointless,)).selected_frontier_id is None
    client.generate_response.assert_not_called()


def _budgeted_run():
    """A run whose ceilings are the proposer's own, for charging one receipt."""
    return SimpleNamespace(
        usage=DiscoveryRun.model_fields["usage"].default_factory(),
        budgets=DiscoveryBudgets(
            observations=10,
            actions=8,
            model_proposals=4,
            resets=2,
            duration_ms=60_000,
            state_depth=6,
            cost_micro_usd=20_000,
            side_effect_actions=8,
            attempts_per_transition=2,
        ),
    )
