from dataclasses import replace

import pytest

from yewo.agent.run.settlement import SettlementStatus
from yewo.discovery.contracts import (
    CapabilityCandidate,
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryTerminal,
    SideEffectClass,
)
from yewo.discovery.controller import DiscoveryController
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryFinding,
    DiscoveryObservationBatch,
)

from discovery_runtime_fakes import (
    SOURCE_PHASH,
    FirstCandidateProposer,
    SuccessfulRuntime,
    _candidate,
    _observation,
    _run,
    _source_batch,
)

def test_controller_accepts_only_authorized_settled_local_evidence(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
    assert result.run.goals[0].state is CapabilityState.ROUTE_VERIFIED
    assert len(result.run.transitions) == 1
    assert result.run.transitions[0].authorization_receipt_id
    assert result.run.usage.actions == 1
    assert result.run.usage.side_effect_actions == 0
    assert runtime.execute_calls == 1
    assert result.findings == ()


def test_policy_refuses_unsafe_frontier_before_model_or_dispatch(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    candidate = replace(
        _candidate(goal_id),
        side_effect=SideEffectClass.EXTERNAL_COMMUNICATION,
        reversible=False,
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id, candidate=candidate),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.POLICY
    assert result.run.goals[0].state is CapabilityState.BLOCKED_BY_POLICY
    assert result.run.usage.model_proposals == 0
    assert runtime.execute_calls == 0


def test_discovery_findings_remain_outside_scenario_and_run_verdicts(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    observation = _observation("source-observation", "source-state", SOURCE_PHASH)
    source = DiscoveryObservationBatch(
        observation=observation,
        capability_evidence=(
            CapabilityEvidenceReceipt(
                goal_id=goal_id,
                observation_id=observation.observation_id,
                kind="observed",
                predicate_id="schedule-list-visible-v1",
            ),
        ),
        findings=(
            DiscoveryFinding(
                kind="universal_crash_freedom",
                severity="info",
                observation_id=observation.observation_id,
                description="The observed screen remained responsive.",
            ),
        ),
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        source,
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
    assert len(result.findings) == 1
    assert "findings" not in type(result.run).model_fields


def test_candidate_identity_cannot_be_reset_by_a_fresh_observation_id(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    first = _candidate(goal_id, "observation-1")
    second = replace(first, source_observation_id="observation-2")

    assert first.frontier_id == second.frontier_id
    assert first.attempt_identity == second.attempt_identity


def test_malformed_runtime_candidate_becomes_typed_infrastructure_block(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    source = _source_batch(goal_id)
    malformed = replace(source.candidates[0], source_screen_state_id="wrong-state")
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        replace(source, candidates=(malformed,)),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert runtime.execute_calls == 0


def test_runtime_depth_is_an_independent_typed_budget_terminal(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id, depth=budgets.state_depth + 1),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED
    assert result.run.terminal.exhausted_budget is DiscoveryBudgetDimension.STATE_DEPTH
    assert runtime.execute_calls == 0


@pytest.mark.parametrize(
    ("budget_update", "candidate_update", "dimension"),
    (
        ({"actions": 0}, {}, DiscoveryBudgetDimension.ACTIONS),
        ({"observations": 1}, {}, DiscoveryBudgetDimension.OBSERVATIONS),
        (
            {"side_effect_actions": 0},
            {"side_effect": SideEffectClass.REVERSIBLE},
            DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS,
        ),
        ({"model_proposals": 0}, {}, DiscoveryBudgetDimension.MODEL_PROPOSALS),
        ({"cost_micro_usd": 0}, {}, DiscoveryBudgetDimension.COST_MICRO_USD),
    ),
)
def test_independent_budget_stops_before_dispatch(
    header, budgets, budget_update, candidate_update, dimension
):
    limited = budgets.model_copy(update=budget_update)
    run = _run(header, limited)
    goal_id = run.goals[0].goal_id
    candidate = replace(_candidate(goal_id), **candidate_update)
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id, candidate=candidate),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.exhausted_budget is dimension
    assert result.run.usage.actions == 0
    assert runtime.execute_calls == 0


def test_duration_budget_is_checked_without_an_action(header, budgets):
    run = _run(header, budgets.model_copy(update={"duration_ms": 1}))
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )
    readings = iter((0.0, 0.002))

    result = DiscoveryController(run, clock=lambda: next(readings)).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.exhausted_budget is DiscoveryBudgetDimension.DURATION_MS
    assert runtime.execute_calls == 0


def test_interruption_precedes_discovery_work_after_setup(header, budgets):
    run = _run(header, budgets)
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(run.goals[0].goal_id),
        run.goals[0].goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime,
        FirstCandidateProposer(),
        interrupted=lambda: True,
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_INTERRUPTED
    assert result.run.usage.observations == 0
    assert runtime.execute_calls == 0


def test_runtime_execution_exception_is_a_typed_infrastructure_block(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id

    class Runtime(SuccessfulRuntime):
        def execute(self, authorization, candidate):
            del authorization, candidate
            self.execute_calls += 1
            raise RuntimeError("lost Appium session")

    runtime = Runtime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.run.usage.actions == 1
    assert runtime.execute_calls == 1


def test_unsettled_edge_is_not_accepted_and_reset_failure_is_typed(header, budgets):
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id

    class Runtime(SuccessfulRuntime):
        def execute(self, authorization, candidate):
            execution = super().execute(authorization, candidate)
            return replace(
                execution,
                settlement=replace(
                    execution.settlement,
                    status=SettlementStatus.NO_VISIBLE_CHANGE,
                ),
            )

    runtime = Runtime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.RESET
    assert result.run.transitions == ()
    assert result.run.goals[0].state is CapabilityState.NOT_EVALUATED


def test_reset_has_an_independent_zero_budget(header, budgets):
    limited = budgets.model_copy(update={"resets": 0})
    run = _run(header, limited)
    goal_id = run.goals[0].goal_id

    class Runtime(SuccessfulRuntime):
        def execute(self, authorization, candidate):
            execution = super().execute(authorization, candidate)
            return replace(
                execution,
                settlement=replace(
                    execution.settlement,
                    status=SettlementStatus.NO_VISIBLE_CHANGE,
                ),
            )

    runtime = Runtime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.exhausted_budget is DiscoveryBudgetDimension.RESETS
    assert result.run.usage.resets == 0


def test_scenario_value_in_finding_is_refused_at_runtime_boundary(header, budgets):
    run = _run(header, budgets)
    observation = _observation("source-observation", "source-state", SOURCE_PHASH)
    source = DiscoveryObservationBatch(
        observation=observation,
        findings=(
            DiscoveryFinding(
                kind="diagnostic",
                severity="info",
                observation_id=observation.observation_id,
                description="The field contained secret-scenario-value.",
            ),
        ),
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        source,
        run.goals[0].goal_id,
    )

    result = DiscoveryController(
        run,
        scenario_owned_values=("secret-scenario-value",),
    ).run_to_terminal(runtime, FirstCandidateProposer())

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.findings == ()


def test_incompatible_setup_blocks_before_observation(header, budgets):
    run = _run(header, budgets)
    runtime = SuccessfulRuntime("f" * 64, _source_batch(run.goals[0].goal_id), "unused")

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.SETUP
    assert result.run.usage.observations == 0


def test_what_a_run_reached_is_kept_even_when_its_goal_is_not(header, budgets):
    """A partial run's screens are evidence. They are not a requirement."""
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    batch = _source_batch(goal_id)
    reached = CapabilityCandidate(
        capability="reach the screen headed Your order",
        goal_ids=(goal_id,),
        observation_ids=(batch.observation.observation_id,),
        confidence=1.0,
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        replace(batch, capability_candidates=(reached,)),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(runtime, FirstCandidateProposer())

    assert [item.capability for item in result.run.capability_candidates] == [
        "reach the screen headed Your order"
    ]


def test_a_reached_screen_naming_a_scenario_value_is_refused(header, budgets):
    """The map is shared knowledge, so a leaked value is a run-ending fault."""
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    batch = _source_batch(goal_id)
    leaking = CapabilityCandidate(
        capability="reach the screen headed 4242424242424242",
        goal_ids=(goal_id,),
        observation_ids=(batch.observation.observation_id,),
        confidence=1.0,
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        replace(batch, capability_candidates=(leaking,)),
        goal_id,
    )

    result = DiscoveryController(
        run, scenario_owned_values=("4242424242424242",)
    ).run_to_terminal(runtime, FirstCandidateProposer())

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.run.capability_candidates == ()


def test_a_runtime_that_cannot_prepare_is_an_infrastructure_fault(header, budgets):
    """A raise is a broken runtime. A refusal is a device reporting its state.

    One blocker for both left the first cold run saying only that setup
    blocked it, when the cause was a runtime reading an attribute no device
    has.
    """
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id

    class _Unpreparable(SuccessfulRuntime):
        def prepare(self, setup_contract_sha256):
            del setup_contract_sha256
            raise RuntimeError("the runtime reached for something no device has")

    runtime = _Unpreparable(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
    )

    result = DiscoveryController(run).run_to_terminal(runtime, FirstCandidateProposer())

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.run.observations == ()
