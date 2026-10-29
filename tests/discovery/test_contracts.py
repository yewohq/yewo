"""Discovery state changes belong to deterministic typed contracts."""

import pytest
from pydantic import ValidationError

from yewo.discovery.contracts import (
    CapabilityQuestion,
    CapabilityCandidate,
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryBudgetUsage,
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTerminal,
    DiscoveryTerminalRecord,
    DiscoveryTerminationSignals,
    RelevantFrontierEntry,
    FrontierState,
    TransitionAttemptUsage,
    finalize_discovery,
)


def _run(header, budgets, question=None, usage=None, frontier=()):
    return DiscoveryRun(
        header=header("discovery-run"),
        questions=(question or CapabilityQuestion(intent="reach the editor"),),
        observations=(
            DiscoveryObservation(
                observation_id="observation-1",
                screen_state_id="screen-1",
                phash="0" * 64,
            ),
        ),
        frontier=frontier,
        budgets=budgets,
        usage=usage or DiscoveryBudgetUsage(),
    )


def test_capability_questions_have_one_legal_terminal():
    question = CapabilityQuestion(intent="reach the editor").activate().close(
        CapabilityState.ROUTE_VERIFIED, ("observation-1",)
    )

    assert question.is_terminal
    with pytest.raises(ValueError, match="immutable"):
        question.close(CapabilityState.UNAVAILABLE)


def test_capability_question_cannot_activate_twice_or_close_as_active():
    active = CapabilityQuestion(intent="reach the editor").activate()

    with pytest.raises(ValueError, match="pending"):
        active.activate()
    with pytest.raises(ValueError, match="terminal state"):
        active.close(CapabilityState.ACTIVE)


def test_success_requires_accepted_observation_evidence():
    with pytest.raises(ValueError, match="evidence"):
        CapabilityQuestion(intent="find the form").activate().close(
            CapabilityState.CAPABILITY_OBSERVED
        )


def test_success_cannot_skip_the_active_state():
    with pytest.raises(ValueError, match="active question"):
        CapabilityQuestion(intent="find the form").close(
            CapabilityState.CAPABILITY_OBSERVED, ("observation-1",)
        )


def test_frontier_is_owned_by_known_observations_and_questions(header, budgets):
    entry = RelevantFrontierEntry(
        source_observation_id="missing",
        control_identity="control-1",
        relevant_question_ids=("missing-question",),
    )

    with pytest.raises(ValueError, match="source observation"):
        _run(header, budgets, frontier=(entry,))


def test_frontier_closes_once_and_verified_transition_requires_identity():
    entry = RelevantFrontierEntry(
        source_observation_id="observation-1",
        control_identity="control-1",
        relevant_question_ids=("question-1",),
    )

    with pytest.raises(ValueError, match="transition identity"):
        entry.close(FrontierState.VERIFIED)
    closed = entry.close(FrontierState.VERIFIED, "transition-1")
    with pytest.raises(ValueError, match="immutable"):
        closed.close(FrontierState.ATTEMPTED)


def test_budget_dimensions_and_transition_attempts_are_independent(budgets):
    usage = DiscoveryBudgetUsage(
        actions=budgets.actions,
        transition_attempts=(
            TransitionAttemptUsage(transition_identity="transition-1", attempts=2),
        ),
    )

    assert usage.exhausted_dimension(budgets) is DiscoveryBudgetDimension.ACTIONS
    assert not usage.can_attempt_transition("transition-1", budgets)
    assert usage.can_attempt_transition("transition-2", budgets)
    assert DiscoveryBudgetUsage().exhausted_dimension(budgets) is None


def test_duplicate_transition_attempt_counters_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        DiscoveryBudgetUsage(
            transition_attempts=(
                TransitionAttemptUsage(transition_identity="transition-1", attempts=1),
                TransitionAttemptUsage(transition_identity="transition-1", attempts=2),
            )
        )


def test_interruption_has_terminal_precedence(header, budgets):
    usage = DiscoveryBudgetUsage(actions=budgets.actions)
    run = _run(header, budgets, usage=usage)

    ended = finalize_discovery(
        run,
        DiscoveryTerminationSignals(
            externally_interrupted=True,
            exhausted_budget=DiscoveryBudgetDimension.ACTIONS,
            blocker=DiscoveryBlocker.SETUP,
            relevant_frontier_remaining=False,
        ),
    )

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_INTERRUPTED
    assert ended.questions[0].state is CapabilityState.NOT_EVALUATED


def test_budget_precedes_blocking_and_completion(header, budgets):
    question = CapabilityQuestion(intent="reach editor").activate().close(
        CapabilityState.ROUTE_VERIFIED, ("observation-1",)
    )
    run = _run(
        header,
        budgets,
        question=question,
        usage=DiscoveryBudgetUsage(model_proposals=budgets.model_proposals),
    )

    ended = finalize_discovery(
        run,
        DiscoveryTerminationSignals(
            exhausted_budget=DiscoveryBudgetDimension.MODEL_PROPOSALS,
            blocker=DiscoveryBlocker.POLICY,
        ),
    )

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED
    assert ended.terminal.exhausted_budget is DiscoveryBudgetDimension.MODEL_PROPOSALS


def test_verified_required_questions_complete_discovery(header, budgets):
    question = CapabilityQuestion(intent="reach editor").activate().close(
        CapabilityState.ROUTE_VERIFIED, ("observation-1",)
    )

    ended = finalize_discovery(_run(header, budgets, question=question), DiscoveryTerminationSignals())

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
    with pytest.raises(ValueError, match="immutable"):
        finalize_discovery(ended, DiscoveryTerminationSignals())


def test_exhausted_frontier_closes_unresolved_questions(header, budgets):
    ended = finalize_discovery(
        _run(
            header,
            budgets,
            question=CapabilityQuestion(intent="reach editor").activate(),
        ),
        DiscoveryTerminationSignals(relevant_frontier_remaining=False),
    )

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_INCOMPLETE
    assert ended.questions[0].state is CapabilityState.UNAVAILABLE


def test_open_frontier_cannot_be_reported_as_a_terminal(header, budgets):
    with pytest.raises(ValueError, match="safe work remains"):
        finalize_discovery(_run(header, budgets), DiscoveryTerminationSignals())


def test_budget_signal_stops_before_the_next_operation_can_overspend(header, budgets):
    ended = finalize_discovery(
        _run(header, budgets),
        DiscoveryTerminationSignals(exhausted_budget=DiscoveryBudgetDimension.ACTIONS),
    )

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED


def test_terminal_reason_is_canonical():
    with pytest.raises(ValidationError, match="canonical wording"):
        DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_INTERRUPTED,
            reason="The model stopped.",
        )


@pytest.mark.parametrize(
    "terminal,reason,fields",
    [
        (
            DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED,
            "The actions discovery budget was exhausted.",
            {},
        ),
        (
            DiscoveryTerminal.DISCOVERY_BLOCKED,
            "Discovery was blocked by setup.",
            {},
        ),
        (
            DiscoveryTerminal.DISCOVERY_COMPLETE,
            "Every required discovery capability was verified.",
            {"blocker": DiscoveryBlocker.SETUP},
        ),
    ],
)
def test_terminal_metadata_must_match_its_kind(terminal, reason, fields):
    with pytest.raises(ValidationError):
        DiscoveryTerminalRecord(terminal=terminal, reason=reason, **fields)


@pytest.mark.parametrize(
    "blocker,expected_state",
    [
        (DiscoveryBlocker.POLICY, CapabilityState.BLOCKED_BY_POLICY),
        (DiscoveryBlocker.SETUP, CapabilityState.BLOCKED_BY_SETUP),
        (DiscoveryBlocker.RESET, CapabilityState.NOT_EVALUATED),
        (DiscoveryBlocker.INFRASTRUCTURE, CapabilityState.NOT_EVALUATED),
    ],
)
def test_blockers_close_questions_with_owned_state(header, budgets, blocker, expected_state):
    ended = finalize_discovery(
        _run(header, budgets), DiscoveryTerminationSignals(blocker=blocker)
    )

    assert ended.terminal.terminal is DiscoveryTerminal.DISCOVERY_BLOCKED
    assert ended.questions[0].state is expected_state


def _mutated_run(run, mutate):
    payload = run.model_dump(mode="json")
    mutate(payload)
    return DiscoveryRun.model_validate(payload)


def test_discovery_run_rejects_wrong_schema_owner(header, budgets):
    run = _run(header, budgets)

    with pytest.raises(ValidationError, match="discovery-run schema"):
        _mutated_run(run, lambda payload: payload["header"].update(schema_name="app-map"))


@pytest.mark.parametrize("collection", ["questions", "observations"])
def test_discovery_run_rejects_duplicate_primary_ids(header, budgets, collection):
    run = _run(header, budgets)

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        _mutated_run(
            run,
            lambda payload: payload[collection].append(payload[collection][0]),
        )


def test_discovery_run_rejects_unknown_candidate_evidence(header, budgets):
    run = _run(header, budgets)
    candidate = CapabilityCandidate(
        capability="editor",
        question_ids=(run.questions[0].question_id,),
        observation_ids=("missing",),
        confidence=0.8,
    )

    with pytest.raises(ValidationError, match="accepted observations"):
        DiscoveryRun(
            **run.model_dump(exclude={"capability_candidates"}),
            capability_candidates=(candidate,),
        )


def test_discovery_run_rejects_unknown_question_evidence(header, budgets):
    question = CapabilityQuestion(intent="editor").activate().close(
        CapabilityState.CAPABILITY_OBSERVED, ("missing",)
    )

    with pytest.raises(ValidationError, match="accepted observations"):
        _run(header, budgets, question=question)


def test_discovery_run_rejects_overspent_budgets(header, budgets):
    with pytest.raises(ValidationError, match="cannot exceed"):
        _run(
            header,
            budgets,
            usage=DiscoveryBudgetUsage(actions=budgets.actions + 1),
        )


def test_discovery_run_rejects_excess_transition_attempts(header, budgets):
    with pytest.raises(ValidationError, match="independent limit"):
        _run(
            header,
            budgets,
            usage=DiscoveryBudgetUsage(
                transition_attempts=(
                    TransitionAttemptUsage(
                        transition_identity="transition-1",
                        attempts=budgets.attempts_per_transition + 1,
                    ),
                )
            ),
        )


@pytest.mark.parametrize("kind", ["frontier", "candidate"])
def test_discovery_run_rejects_duplicate_secondary_ids(header, budgets, kind):
    run = _run(header, budgets)
    question_id = run.questions[0].question_id
    if kind == "frontier":
        item = RelevantFrontierEntry(
            source_observation_id="observation-1",
            control_identity="control-1",
            relevant_question_ids=(question_id,),
        )
        run = run.model_copy(update={"frontier": (item,)})
        field = "frontier"
    else:
        item = CapabilityCandidate(
            capability="editor",
            question_ids=(question_id,),
            observation_ids=("observation-1",),
            confidence=0.8,
        )
        run = run.model_copy(update={"capability_candidates": (item,)})
        field = "capability_candidates"

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        _mutated_run(run, lambda payload: payload[field].append(payload[field][0]))


def test_frontier_rejects_unknown_or_already_settled_question(header, budgets):
    run = _run(header, budgets)
    unknown = RelevantFrontierEntry(
        source_observation_id="observation-1",
        control_identity="control-1",
        relevant_question_ids=("missing",),
    )
    settled = run.questions[0].activate().close(
        CapabilityState.CAPABILITY_OBSERVED, ("observation-1",)
    )
    available = RelevantFrontierEntry(
        source_observation_id="observation-1",
        control_identity="control-1",
        relevant_question_ids=(settled.question_id,),
    )

    with pytest.raises(ValidationError, match="unknown capability question"):
        DiscoveryRun.model_validate(
            run.model_copy(update={"frontier": (unknown,)}).model_dump(mode="json")
        )
    with pytest.raises(ValidationError, match="unresolved question"):
        DiscoveryRun(
            header=run.header,
            questions=(settled,),
            observations=run.observations,
            frontier=(available,),
            budgets=budgets,
        )


def test_capability_candidate_rejects_unknown_question(header, budgets):
    run = _run(header, budgets)
    candidate = CapabilityCandidate(
        capability="editor",
        question_ids=("missing",),
        observation_ids=("observation-1",),
        confidence=0.8,
    )

    with pytest.raises(ValidationError, match="unknown question"):
        DiscoveryRun(
            header=run.header,
            questions=run.questions,
            observations=run.observations,
            capability_candidates=(candidate,),
            budgets=budgets,
        )


def test_terminal_run_rejects_open_or_unsuccessful_required_question(header, budgets):
    terminal = DiscoveryTerminalRecord(
        terminal=DiscoveryTerminal.DISCOVERY_COMPLETE,
        reason="Every required discovery capability was verified.",
    )
    open_run = _run(header, budgets)
    unavailable = open_run.questions[0].activate().close(CapabilityState.UNAVAILABLE)

    with pytest.raises(ValidationError, match="open question"):
        DiscoveryRun(
            header=open_run.header,
            questions=open_run.questions,
            observations=open_run.observations,
            budgets=budgets,
            terminal=terminal,
        )
    with pytest.raises(ValidationError, match="every required capability"):
        DiscoveryRun(
            header=open_run.header,
            questions=(unavailable,),
            observations=open_run.observations,
            budgets=budgets,
            terminal=terminal,
        )
