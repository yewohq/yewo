"""Deterministic confirmation of general auditor findings."""

from yewo.project_config import CoverageCategory
from yewo.reporting import AuditObligationStatus
from yewo.scenarios.auditor_findings import (
    AuditCandidateFindingOutput,
    confirm_audit_finding,
    resolve_candidate_findings,
)


def _candidate(verification: dict[str, object]) -> AuditCandidateFindingOutput:
    return AuditCandidateFindingOutput.model_validate(
        {
            "outcome_id": "outcome_1",
            "claim": "The visible state is inconsistent.",
            "regions": [
                {
                    "observation_id": "observation_1",
                    "label": "visible evidence",
                    "left": 100,
                    "top": 100,
                    "right": 900,
                    "bottom": 900,
                }
            ],
            "verification": verification,
        }
    )


def test_visible_sum_mismatch_is_confirmed() -> None:
    result = confirm_audit_finding(
        _candidate(
            {
                "kind": "arithmetic",
                "relation": "sum",
                "operands": ["9.99", "-1.00"],
                "displayed_result": "9.00",
                "precision_places": 2,
                "unit": "USD",
            }
        )
    )

    assert result.is_violation
    assert "8.99 USD" in result.reason
    assert "9.00 USD" in result.reason


def test_visible_sum_that_reconciles_is_not_a_violation() -> None:
    result = confirm_audit_finding(
        _candidate(
            {
                "kind": "arithmetic",
                "relation": "sum",
                "operands": ["9.99", "-1.00", "0.01"],
                "displayed_result": "9.00",
                "precision_places": 2,
                "unit": "USD",
            }
        )
    )

    assert not result.is_violation


def test_changed_selection_is_confirmed_without_numeric_values() -> None:
    result = confirm_audit_finding(
        _candidate(
            {
                "kind": "equality",
                "expected_text": "Collect in store",
                "observed_text": "Home delivery",
            }
        )
    )

    assert result.is_violation
    assert "Collect in store" in result.reason
    assert "Home delivery" in result.reason


def test_arithmetic_candidate_cannot_fail_ui_visual_quality() -> None:
    status, reason, finding = resolve_candidate_findings(
        CoverageCategory.UI_VISUAL,
        AuditObligationStatus.UNMET,
        "The total is inconsistent.",
        (
            _candidate(
                {
                    "kind": "arithmetic",
                    "relation": "difference",
                    "operands": ["9.99", "1.00"],
                    "displayed_result": "9.00",
                }
            ),
        ),
    )

    assert status is AuditObligationStatus.UNVERIFIED
    assert reason == "The auditor reported a violation without a confirmed candidate."
    assert finding is None


def test_visual_candidate_can_fail_ui_visual_quality() -> None:
    status, reason, finding = resolve_candidate_findings(
        CoverageCategory.UI_VISUAL,
        AuditObligationStatus.UNMET,
        "A visible defect is present.",
        (_candidate({"kind": "visual"}),),
    )

    assert status is AuditObligationStatus.UNMET
    assert "directly visible defect" in reason
    assert finding is not None
    assert finding.regions[0].observation_id == "observation_1"
