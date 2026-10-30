"""Discovery controller decisions that leave or terminate the frontier."""

from dataclasses import replace

import pytest

from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryTerminal,
    FrontierState,
    ObservedControl,
)
from yewo.discovery.controller import DiscoveryController
from yewo.discovery.runtime import (
    DiscoveryObservationBatch,
    DiscoveryProposalReceipt,
    DiscoveryResetReceipt,
)
from yewo.agent.memory.replay_store import RelativeBounds

from discovery_runtime_fakes import (
    DESTINATION_PHASH,
    SOURCE_PHASH,
    FirstCandidateProposer,
    SuccessfulRuntime,
    UnknownCandidateProposer,
    _run,
    _candidate,
    _observation,
    _source_batch,
)


class _DecliningProposer(FirstCandidateProposer):
    def select(self, run, candidates):
        del run, candidates
        return DiscoveryProposalReceipt(
            selected_frontier_id=None, model_calls=1, cost_micro_usd=50
        )

    @staticmethod
    def drain_unreported_spend():
        return 0, 0


class _ResettingRuntime(SuccessfulRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resets = 0

    def execute(self, authorization, candidate):
        execution = super().execute(authorization, candidate)
        return replace(execution, capability_evidence=())

    def reset(self):
        self.resets += 1
        return DiscoveryResetReceipt(success=True, observation=self.source)


class _DecliningAfterOneStep(FirstCandidateProposer):
    def __init__(self):
        self.calls = 0

    def select(self, run, candidates):
        del run
        self.calls += 1
        selected_frontier_id = (
            candidates[0].frontier_id if self.calls == 1 else None
        )
        return DiscoveryProposalReceipt(
            selected_frontier_id=selected_frontier_id,
            model_calls=1,
            cost_micro_usd=50,
        )

    @staticmethod
    def drain_unreported_spend():
        return 0, 0


class _PendingSpendProposer(FirstCandidateProposer):
    def __init__(self, model_calls, micro_usd):
        self.pending_spend = (model_calls, micro_usd)

    def drain_unreported_spend(self):
        pending_spend = self.pending_spend
        self.pending_spend = (0, 0)
        return pending_spend


class _FailedProvider(FirstCandidateProposer):
    def select(self, run, candidates):
        del run, candidates
        return DiscoveryProposalReceipt(
            selected_frontier_id=None,
            model_calls=1,
            cost_micro_usd=50,
            provider_failed=True,
        )


class _DeclineFirstWindow(FirstCandidateProposer):
    def __init__(self):
        self.window_sizes = []

    def select(self, run, candidates):
        del run
        self.window_sizes.append(len(candidates))
        return DiscoveryProposalReceipt(
            selected_frontier_id=(
                None if len(self.window_sizes) == 1 else candidates[0].frontier_id
            ),
            model_calls=1,
            cost_micro_usd=50,
        )


def _batch(observation_id, state, phash, candidates):
    """One frame. The first candidate is named, so it is the one offered first."""
    controls = tuple(
        ObservedControl(
            control_identity=candidate.control_identity,
            role="button",
            bounds=RelativeBounds(left=0, top=700, right=100, bottom=800),
            text_digest="a" * 64 if position == 0 else "",
            actionable=True,
        )
        for position, candidate in enumerate(candidates)
    )
    return DiscoveryObservationBatch(
        observation=_observation(observation_id, state, phash, controls),
        candidates=candidates,
    )


class _ReturnsHomeWithoutOneControl(SuccessfulRuntime):
    """Relaunches to a start screen one of its earlier controls is missing from.

    A row scrolled away reads this way, and so did every offer named under
    the fingerprint scheme. The run stands on that offer's screen and still
    cannot take it.
    """

    def __init__(self, setup_sha, source, home, goal_id, menu_candidate):
        super().__init__(setup_sha, source, goal_id)
        self.home = home
        self.menu_candidate = menu_candidate

    def execute(self, authorization, candidate):
        execution = super().execute(authorization, candidate)
        return replace(
            execution,
            capability_evidence=(),
            destination=_batch(
                "destination-observation",
                "destination-state",
                DESTINATION_PHASH,
                (self.menu_candidate,),
            ),
        )

    def reset(self):
        return DiscoveryResetReceipt(success=True, observation=self.home)


def test_an_unreachable_offer_does_not_write_off_the_work_behind_it(
    header, budgets
):
    """One stranded offer made its own screen read as work already in hand.

    The run then stopped looking for a route, and closed the offers on every
    other screen with it. Here the stranded offer retires on its own, and the
    verified route to the screen that still holds work is taken.

    The first tap asks for a reset, so the run reaches home with the menu's
    offer still open. That is the state the frontier was written off from.
    """
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    opens_menu = _candidate(goal_id, x=200)
    scrolled_away = _candidate(goal_id, x=600)
    menu_offer = _candidate(
        goal_id,
        "destination-observation",
        source_state="destination-state",
        x=400,
    )
    runtime = _ReturnsHomeWithoutOneControl(
        run.header.compatibility.setup_contract_sha256,
        _batch(
            "source-observation",
            "source-state",
            SOURCE_PHASH,
            (replace(opens_menu, reset_after=True), scrolled_away),
        ),
        _batch(
            "home-observation",
            "source-state",
            SOURCE_PHASH,
            (replace(opens_menu, source_observation_id="home-observation"),),
        ),
        goal_id,
        menu_offer,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, _DecliningAfterOneStep()
    )

    states = {entry.control_identity: entry.state for entry in result.run.frontier}
    assert states[scrolled_away.control_identity] is (
        FrontierState.NAVIGATION_UNAVAILABLE
    )
    assert states[menu_offer.control_identity] is FrontierState.DECLINED
    assert runtime.execute_calls == 2


def _three_candidate_batch(goal_id):
    candidates = tuple(
        _candidate(goal_id, x=200 + position * 200, y=1600)
        for position in range(3)
    )
    controls = tuple(
        ObservedControl(
            control_identity=candidate.control_identity,
            role="button",
            bounds=RelativeBounds(
                left=position * 200,
                top=700,
                right=position * 200 + 100,
                bottom=800,
            ),
            text_digest=str(position + 1) * 64,
            actionable=True,
        )
        for position, candidate in enumerate(candidates)
    )
    return DiscoveryObservationBatch(
        observation=_observation(
            "source-observation",
            "source-state",
            "0" * 16,
            controls,
        ),
        candidates=candidates,
    )


def test_a_declined_screen_sends_the_run_home_instead_of_ending_it(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = _ResettingRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    DiscoveryController(run).run_to_terminal(runtime, _DecliningAfterOneStep())

    assert runtime.resets >= 1


def test_a_declined_root_screen_reports_the_frontier_spent(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = _ResettingRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(runtime, _DecliningProposer())

    assert runtime.resets == 0
    assert result.run.goals[0].state is CapabilityState.UNREACHABLE
    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_INCOMPLETE


def test_a_decline_retires_only_its_bounded_proposal_window(header, budgets):
    limited = budgets.model_copy(update={"proposal_window_size": 2})
    run = _run(header, limited)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _three_candidate_batch(goal_id),
        goal_id,
    )
    proposer = _DeclineFirstWindow()

    result = DiscoveryController(run).run_to_terminal(runtime, proposer)

    assert proposer.window_sizes == [2, 1]
    assert [entry.state for entry in result.run.frontier].count(
        FrontierState.DECLINED
    ) == 2
    assert [entry.state for entry in result.run.frontier].count(
        FrontierState.VERIFIED
    ) == 1


def test_an_unoffered_control_is_ambiguous(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, UnknownCandidateProposer()
    )

    assert result.run.goals[0].state is CapabilityState.AMBIGUOUS
    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_INCOMPLETE
    assert result.run.usage.actions == 0
    assert runtime.execute_calls == 0


def test_a_provider_failure_is_not_recorded_as_a_declined_frontier(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(runtime, _FailedProvider())

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.run.frontier[0].state is FrontierState.AVAILABLE


def test_final_spend_above_the_proposer_ceiling_is_an_infrastructure_block(
    header, budgets
):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, _PendingSpendProposer(model_calls=1, micro_usd=101)
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE


@pytest.mark.parametrize(
    ("budget_update", "pending_spend", "dimension"),
    (
        (
            {"model_proposals": 1},
            (1, 0),
            DiscoveryBudgetDimension.MODEL_PROPOSALS,
        ),
        (
            {"cost_micro_usd": 100},
            (0, 60),
            DiscoveryBudgetDimension.COST_MICRO_USD,
        ),
    ),
)
def test_final_spend_cannot_overrun_the_run_budget(
    header, budgets, budget_update, pending_spend, dimension
):
    limited = budgets.model_copy(update=budget_update)
    run = _run(header, limited)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime,
        _PendingSpendProposer(
            model_calls=pending_spend[0], micro_usd=pending_spend[1]
        ),
    )

    assert result.run.terminal.exhausted_budget is dimension
