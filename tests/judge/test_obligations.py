from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from yewo.identity import derived_uuid
from yewo.testing.models import AssertionScope, ExpectedOutcome
from yewo.testing.obligations import (
    ObligationActivation,
    ObligationController,
    ObligationEvent,
    ObligationSettlement,
    ObligationState,
    RouteDecision,
    select_outcomes,
)


@dataclass
class _Scenario:
    expected_outcomes: list[ExpectedOutcome]
    scenario_uuid: str = "scenario-1"

    def outcome_uuid(self, index: int) -> str:
        return derived_uuid(self.scenario_uuid, index)


def _controller(*outcomes: ExpectedOutcome) -> ObligationController:
    return ObligationController.from_scenario(_Scenario(list(outcomes)))


def test_default_screen_outcome_is_routed_only_at_run_end() -> None:
    controller = _controller(ExpectedOutcome(statement="The saved schedule is visible."))

    assert controller.route(ObligationEvent.OBSERVATION) == ()
    assert controller.route(ObligationEvent.RUN_END) == (0,)
    assert [receipt.decision for receipt in controller.route_receipts] == [
        RouteDecision.INACTIVE,
        RouteDecision.ACTIVE,
    ]


def test_transition_scope_owns_its_activation_and_original_index() -> None:
    current = ExpectedOutcome(statement="The final screen is visible.")
    transition = ExpectedOutcome(
        statement="Saving reaches the schedule list.",
        scope=AssertionScope.TRANSITION,
    )
    controller = _controller(current, transition)

    indexes = controller.route(ObligationEvent.TRANSITION)

    assert transition.obligation.activation is ObligationActivation.ON_TRANSITION
    assert indexes == (1,)
    assert select_outcomes([current, transition], indexes) == [None, transition]


def test_unknown_instruction_state_routes_to_evaluation_without_passing() -> None:
    controller = _controller(
        ExpectedOutcome(
            statement="The draft name is visible.",
            obligation={
                "activation": ObligationActivation.DURING_INSTRUCTION,
                "instruction_index": 2,
            },
        )
    )

    assert controller.route(ObligationEvent.OBSERVATION) == (0,)
    assert controller.route_receipts[-1].decision is RouteDecision.UNKNOWN
    assert not controller.readiness().strict_ready


def test_absent_conditional_subject_is_not_passed_or_failed() -> None:
    controller = _controller(
        ExpectedOutcome(
            statement="The exercise row shows its settings.",
            obligation={
                "activation": ObligationActivation.WHEN_SUBJECT_OBSERVED,
                "subject": "exercise row",
                "settlement": ObligationSettlement.CONDITIONAL,
            },
        )
    )

    assert controller.route(
        ObligationEvent.OBSERVATION, subject_observed=False
    ) == ()
    assert controller.route_receipts[-1].decision is RouteDecision.CONDITION_ABSENT
    assert controller.close_run()[0].state is ObligationState.INACTIVE


def test_only_evidence_from_an_active_window_can_settle_obligation() -> None:
    controller = _controller(ExpectedOutcome(statement="The result is visible."))
    controller.record_verification(0, "passed", True, "too-early-screen")
    controller.route(ObligationEvent.RUN_END, evidence_reference="final-screen")
    controller.record_verification(0, "passed", True, "final-screen")

    receipt = controller.close_run()[0]

    assert receipt.state is ObligationState.SATISFIED
    assert receipt.evidence_references == ("final-screen",)


def test_failed_final_evidence_fails_exists_obligation() -> None:
    controller = _controller(ExpectedOutcome(statement="The result is visible."))
    controller.route(ObligationEvent.RUN_END)
    controller.record_verification(0, "failed", True, "final-screen")

    assert controller.close_run()[0].state is ObligationState.FAILED


def test_missing_final_evidence_remains_unverified() -> None:
    controller = _controller(ExpectedOutcome(statement="The result is visible."))
    controller.route(ObligationEvent.RUN_END)

    assert controller.close_run()[0].state is ObligationState.UNVERIFIED


def test_instruction_activation_requires_an_instruction_index() -> None:
    with pytest.raises(ValidationError, match="instruction_index"):
        ExpectedOutcome(
            statement="The field is visible.",
            obligation={"activation": ObligationActivation.AFTER_INSTRUCTION},
        )


def test_subject_activation_requires_a_subject() -> None:
    with pytest.raises(ValidationError, match="must name its subject"):
        ExpectedOutcome(
            statement="The subject is visible.",
            obligation={
                "activation": ObligationActivation.WHEN_SUBJECT_OBSERVED
            },
        )
