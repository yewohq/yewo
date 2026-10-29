from dataclasses import replace

import pytest
from pydantic import ValidationError

from yewo.agent.actuation.actions import Coordinates, PlannerTapAction
from yewo.agent.actuation.safety import ActionSafetyPolicy, ActionSource
from yewo.agent.memory.control_identity import control_identity_id
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.contracts import (
    CapabilityQuestion,
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTerminal,
    ObservedControl,
    SideEffectClass,
)
from yewo.discovery.controller import DiscoveryController
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryActionCandidate,
    DiscoveryDispatchReceipt,
    DiscoveryExecutionReceipt,
    DiscoveryFinding,
    DiscoveryObservationBatch,
    DiscoveryProposalReceipt,
    DiscoveryResetReceipt,
    DiscoverySetupReceipt,
)
from yewo.identity import derived_uuid


SOURCE_PHASH = "0" * 16
DESTINATION_PHASH = "f" * 16
SCREEN_SIZE = (1000, 2000)

def _run(header, budgets, question=None):
    return DiscoveryRun(
        header=header("discovery-run"),
        questions=(question or CapabilityQuestion(intent="reach the editor"),),
        budgets=budgets,
    )

def _observation(identifier, state, phash, controls=()):
    return DiscoveryObservation(
        observation_id=identifier,
        screen_state_id=state,
        phash=phash,
        controls=controls,
    )

def _candidate(question_id, observation_id="source-observation"):
    action = PlannerTapAction(
        coordinates=Coordinates(x=800, y=1600),
        target="observed control",
        reason="inspect the safe relevant frontier",
    )
    authorization = ActionSafetyPolicy().authorize(
        action,
        source=ActionSource.DISCOVERY,
        observation_id=observation_id,
        screen_phash=SOURCE_PHASH,
        screen_size=SCREEN_SIZE,
    )
    return DiscoveryActionCandidate(
        source_observation_id=observation_id,
        source_screen_state_id="source-state",
        control_identity=control_identity_id(authorization.control_identity),
        relevant_question_ids=(question_id,),
        action=action,
        side_effect=SideEffectClass.READ_ONLY,
        reversible=True,
        source_phash=SOURCE_PHASH,
        screen_size=SCREEN_SIZE,
    )


def _source_batch(question_id, *, candidate=None, depth=0):
    candidate = candidate or _candidate(question_id)
    control = ObservedControl(
        control_identity=candidate.control_identity,
        role="button",
        bounds=RelativeBounds(left=750, top=750, right=900, bottom=900),
        actionable=True,
    )
    observation = _observation(
        candidate.source_observation_id,
        candidate.source_screen_state_id,
        SOURCE_PHASH,
        (control,),
    )
    return DiscoveryObservationBatch(
        observation=observation,
        candidates=(candidate,),
        state_depth=depth,
    )


class FirstCandidateProposer:
    model_calls_per_proposal = 1
    maximum_cost_micro_usd = 100

    def select(self, run, candidates):
        del run
        return DiscoveryProposalReceipt(
            selected_frontier_id=candidates[0].frontier_id,
            model_calls=1,
            cost_micro_usd=50,
        )


class UnknownCandidateProposer(FirstCandidateProposer):
    def select(self, run, candidates):
        del run, candidates
        return DiscoveryProposalReceipt(
            selected_frontier_id="not-offered",
            model_calls=1,
            cost_micro_usd=50,
        )


class SuccessfulRuntime:
    def __init__(self, setup_sha, source, question_id):
        self.setup_sha = setup_sha
        self.source = source
        self.question_id = question_id
        self.execute_calls = 0

    def prepare(self, setup_contract_sha256):
        del setup_contract_sha256
        return DiscoverySetupReceipt(
            success=True,
            setup_contract_sha256=self.setup_sha,
        )

    def observe(self):
        return self.source

    def execute(self, authorization, candidate):
        self.execute_calls += 1
        destination = _observation(
            "destination-observation",
            "destination-state",
            DESTINATION_PHASH,
        )
        transition_identity = derived_uuid(
            "discovery-transition-v1",
            candidate.source_observation_id,
            candidate.control_identity,
            destination.observation_id,
        )
        return DiscoveryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id="dispatch-1",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id="settlement-1",
                control_identity=authorization.control_identity,
                source_phash=SOURCE_PHASH,
                destination_phash=DESTINATION_PHASH,
                status=SettlementStatus.DURABLE_PROGRESS,
                reason="stable destination",
                observation_ids=(
                    candidate.source_observation_id,
                    destination.observation_id,
                ),
            ),
            destination=DiscoveryObservationBatch(observation=destination, state_depth=1),
            capability_evidence=(
                CapabilityEvidenceReceipt(
                    question_id=self.question_id,
                    observation_id=destination.observation_id,
                    kind="route_verified",
                    predicate_id="editor-visible-v1",
                    transition_identity=transition_identity,
                ),
            ),
        )

    def reset(self):
        return DiscoveryResetReceipt(success=False, reason="not expected")


def test_controller_accepts_only_authorized_settled_local_evidence(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
    assert result.run.questions[0].state is CapabilityState.ROUTE_VERIFIED
    assert len(result.run.transitions) == 1
    assert result.run.transitions[0].authorization_receipt_id
    assert result.run.usage.actions == 1
    assert result.run.usage.side_effect_actions == 0
    assert runtime.execute_calls == 1
    assert result.findings == ()


def test_proposer_cannot_select_a_control_outside_authorized_frontier(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, UnknownCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_INCOMPLETE
    assert result.run.questions[0].state is CapabilityState.AMBIGUOUS
    assert result.run.usage.actions == 0
    assert runtime.execute_calls == 0


def test_policy_refuses_unsafe_frontier_before_model_or_dispatch(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    candidate = replace(
        _candidate(question_id),
        side_effect=SideEffectClass.EXTERNAL_COMMUNICATION,
        reversible=False,
    )
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id, candidate=candidate),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.POLICY
    assert result.run.questions[0].state is CapabilityState.BLOCKED_BY_POLICY
    assert result.run.usage.model_proposals == 0
    assert runtime.execute_calls == 0


def test_discovery_findings_remain_outside_scenario_and_run_verdicts(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    observation = _observation("source-observation", "source-state", SOURCE_PHASH)
    source = DiscoveryObservationBatch(
        observation=observation,
        capability_evidence=(
            CapabilityEvidenceReceipt(
                question_id=question_id,
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
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
    assert len(result.findings) == 1
    assert "findings" not in type(result.run).model_fields


def test_candidate_identity_cannot_be_reset_by_a_fresh_observation_id(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    first = _candidate(question_id, "observation-1")
    second = replace(first, source_observation_id="observation-2")

    assert first.frontier_id == second.frontier_id
    assert first.attempt_identity == second.attempt_identity


def test_malformed_runtime_candidate_becomes_typed_infrastructure_block(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    source = _source_batch(question_id)
    malformed = replace(source.candidates[0], source_screen_state_id="wrong-state")
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        replace(source, candidates=(malformed,)),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert runtime.execute_calls == 0


def test_runtime_depth_is_an_independent_typed_budget_terminal(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id, depth=budgets.state_depth + 1),
        question_id,
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
    question_id = run.questions[0].question_id
    candidate = replace(_candidate(question_id), **candidate_update)
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id, candidate=candidate),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.exhausted_budget is dimension
    assert result.run.usage.actions == 0
    assert runtime.execute_calls == 0


def test_duration_budget_is_checked_without_an_action(header, budgets):
    run = _run(header, budgets.model_copy(update={"duration_ms": 1}))
    question_id = run.questions[0].question_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id),
        question_id,
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
        _source_batch(run.questions[0].question_id),
        run.questions[0].question_id,
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
    question_id = run.questions[0].question_id

    class Runtime(SuccessfulRuntime):
        def execute(self, authorization, candidate):
            del authorization, candidate
            self.execute_calls += 1
            raise RuntimeError("lost Appium session")

    runtime = Runtime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(question_id),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.run.usage.actions == 1
    assert runtime.execute_calls == 1


def test_unsettled_edge_is_not_accepted_and_reset_failure_is_typed(header, budgets):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id

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
        _source_batch(question_id),
        question_id,
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.RESET
    assert result.run.transitions == ()
    assert result.run.questions[0].state is CapabilityState.NOT_EVALUATED


def test_reset_has_an_independent_zero_budget(header, budgets):
    limited = budgets.model_copy(update={"resets": 0})
    run = _run(header, limited)
    question_id = run.questions[0].question_id

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
        _source_batch(question_id),
        question_id,
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
        run.questions[0].question_id,
    )

    result = DiscoveryController(
        run,
        forbidden_scenario_values=("secret-scenario-value",),
    ).run_to_terminal(runtime, FirstCandidateProposer())

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
    assert result.findings == ()


def test_incompatible_setup_blocks_before_observation(header, budgets):
    run = _run(header, budgets)
    runtime = SuccessfulRuntime("f" * 64, _source_batch(run.questions[0].question_id), "unused")

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.SETUP
    assert result.run.usage.observations == 0


def test_capability_evidence_owner_cannot_be_changed_by_a_proposer():
    with pytest.raises(ValidationError):
        CapabilityEvidenceReceipt(
            evidence_owner="model",
            question_id="question-1",
            observation_id="observation-1",
            kind="observed",
            predicate_id="predicate-1",
        )
