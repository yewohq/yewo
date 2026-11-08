"""Independent scenario-auditor contracts."""

from io import BytesIO
from threading import Barrier

from PIL import Image

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import ModelImageConstraints, ModelRole
from yewo.observations import (
    ObservationStore,
    RedactionProvenance,
    ScreenshotStore,
)
from yewo.project_config import CoverageCategory
from yewo.reporting import AuditObligationStatus, ScenarioAuditVerdict
from yewo.scenarios import ScenarioAuditorService, ScenarioExecutionChange
from yewo.scenarios import auditor as auditor_module
from yewo.scenarios import auditor_evidence

from agent.planning.support import StubPlannerModel
from agent.replay.support import observations
from scenarios.auditor_support import visual_finding
from scenarios.test_coordinator import _coordinator, _generated_suite


def _selected_scenario(category: CoverageCategory):
    suite = _generated_suite()
    baseline = next(
        scenario
        for scenario in suite.scenarios
        if scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    obligation = next(item for item in baseline.obligations if item.category is category)
    scenario = baseline.model_copy(update={"obligations": (obligation,)})
    return suite.model_copy(update={"scenarios": (scenario,)}), scenario


def _selected_consistency_scenario():
    suite = _generated_suite()
    baseline = next(
        scenario
        for scenario in suite.scenarios
        if scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    template = next(
        item for item in baseline.obligations if item.category is CoverageCategory.STATE
    )
    template_outcome = template.expected_outcomes[0]
    outcome = template_outcome.model_copy(
        update={
            "outcome_id": "consistency_outcome",
            "statement": "Derived values must match their observed source values.",
        }
    )
    obligation = template.model_copy(
        update={
            "obligation_id": "consistency_obligation",
            "category_assessment_id": "consistency_assessment",
            "category": CoverageCategory.DATA_CONSISTENCY,
            "coverage_cell_id": "data_consistency.happy_path.0",
            "expected_outcomes": (outcome,),
        }
    )
    scenario = baseline.model_copy(update={"obligations": (obligation,)})
    return suite.model_copy(update={"scenarios": (scenario,)}), scenario


def _persist_final_observation(root) -> None:
    output = BytesIO()
    Image.new("RGB", (200, 400), "white").save(output, format="PNG")
    reference = ScreenshotStore(root).persist_redacted(
        output.getvalue(),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
        redaction=RedactionProvenance(
            policy_id="test-redaction",
            region_count=0,
            region_set_sha256="a" * 64,
            input_sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        ),
    )
    final = observations()[-1]
    final = final.model_copy(
        update={
            "redacted_screenshot": reference,
            "screen_fingerprint": final.screen_fingerprint.model_copy(
                update={"source_artifact_id": reference.artifact_id}
            ),
        }
    )
    ObservationStore(root).persist(final)


def _constraints() -> ModelImageConstraints:
    return ModelImageConstraints(
        max_images=4,
        max_image_bytes=2_000_000,
        max_width_px=1_000,
        max_height_px=1_000,
    )


def test_completed_scenario_receives_an_independent_model_verdict(tmp_path) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.FUNCTIONAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    outcomes = tuple(
        outcome for obligation in scenario.obligations for outcome in obligation.expected_outcomes
    )
    model = StubPlannerModel(
        {
            "summary": "Current-run evidence satisfies every supplied outcome.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "satisfied",
                    "reason": "The supplied current-run evidence shows the expected result.",
                }
                for outcome in outcomes
            ],
        }
    )

    auditor = ScenarioAuditorService(model, "auditor-primary", _constraints())
    audit = auditor.audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.PASSED
    assert {item.status for item in audit.obligations} == {AuditObligationStatus.SATISFIED}
    assert model.requests[0].role is ModelRole.AUDITOR
    assert model.requests[0].response_contract is not None
    assert model.requests[0].response_contract.name == "scenario-audit-v1"
    assert auditor.model_call_count == 1


def test_unmet_outcome_fails_the_independent_audit(tmp_path) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.FUNCTIONAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    outcomes = tuple(
        outcome for obligation in scenario.obligations for outcome in obligation.expected_outcomes
    )
    model = StubPlannerModel(
        {
            "summary": "Current-run evidence contradicts an expected outcome.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "unmet",
                    "reason": "The supplied current-run evidence shows the wrong result.",
                }
                for outcome in outcomes
            ],
        }
    )

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.FAILED
    assert {item.status for item in audit.obligations} == {AuditObligationStatus.UNMET}


def test_missing_accessibility_inspection_remains_unverified(tmp_path) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.ACCESSIBILITY)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    model = StubPlannerModel(None)

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.UNVERIFIED
    assert model.requests == []
    assert "accessibility_inspection" in audit.obligations[0].reason


def test_consistency_auditor_inspects_screens_before_measurement_exists(tmp_path) -> None:
    suite, scenario = _selected_consistency_scenario()
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    outcome = scenario.obligations[0].expected_outcomes[0]
    model = StubPlannerModel(
        {
            "summary": "The visible values are internally consistent.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "satisfied",
                    "reason": "The supplied screenshots contain no visible contradiction.",
                }
            ],
        }
    )

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.PASSED
    assert len(model.requests) == 1
    assert "measurement" not in audit.obligations[0].reason


def test_arithmetic_candidate_confirms_visible_consistency_failure(tmp_path) -> None:
    suite, scenario = _selected_consistency_scenario()
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    outcome = scenario.obligations[0].expected_outcomes[0]
    observation_id = observations()[-1].observation_id
    model = StubPlannerModel(
        {
            "summary": "A visible calculation needs deterministic confirmation.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "unverified",
                    "reason": "The displayed total appears inconsistent.",
                }
            ],
            "findings": [
                {
                    "outcome_id": outcome.outcome_id,
                    "claim": "The subtotal and discount do not reconcile with the total.",
                    "regions": [
                        {
                            "observation_id": observation_id,
                            "label": "checkout values",
                            "left": 100,
                            "top": 400,
                            "right": 900,
                            "bottom": 900,
                        }
                    ],
                    "verification": {
                        "kind": "arithmetic",
                        "relation": "sum",
                        "operands": ["9.99", "-1.00"],
                        "displayed_result": "9.00",
                        "precision_places": 2,
                        "unit": "USD",
                    },
                }
            ],
        }
    )

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.FAILED
    assert audit.obligations[0].status is AuditObligationStatus.UNMET
    assert "8.99 USD" in audit.obligations[0].reason
    assert "9.00 USD" in audit.obligations[0].reason


def test_unconfirmed_consistency_claim_remains_unverified(tmp_path) -> None:
    suite, scenario = _selected_consistency_scenario()
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    outcome = scenario.obligations[0].expected_outcomes[0]
    model = StubPlannerModel(
        {
            "summary": "The auditor claims a mismatch without structured evidence.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "unmet",
                    "reason": "The values look wrong.",
                }
            ],
        }
    )

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.UNVERIFIED
    assert audit.obligations[0].status is AuditObligationStatus.UNVERIFIED
    assert "without a confirmed candidate" in audit.obligations[0].reason


def test_missing_obligation_does_not_block_an_auditable_obligation(tmp_path) -> None:
    suite = _generated_suite()
    baseline = next(
        scenario
        for scenario in suite.scenarios
        if scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    accessibility = next(
        item for item in baseline.obligations if item.category is CoverageCategory.ACCESSIBILITY
    )
    functional = next(
        item for item in baseline.obligations if item.category is CoverageCategory.FUNCTIONAL
    )
    scenario = baseline.model_copy(update={"obligations": (accessibility, functional)})
    suite = suite.model_copy(
        update={
            "scenarios": tuple(
                scenario if item.scenario_id == scenario.scenario_id else item
                for item in suite.scenarios
            )
        }
    )
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    model = StubPlannerModel(
        {
            "summary": "The functional evidence is sufficient.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "satisfied",
                    "reason": "The current-run evidence shows the expected result.",
                }
                for outcome in functional.expected_outcomes
            ],
        }
    )

    audit = ScenarioAuditorService(model, "auditor-primary", _constraints()).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.UNVERIFIED
    assert len(model.requests) == 1
    assert {item.status for item in audit.obligations} == {
        AuditObligationStatus.SATISFIED,
        AuditObligationStatus.UNVERIFIED,
    }


def test_oversized_obligation_is_unverified_instead_of_auditor_error(
    tmp_path,
    monkeypatch,
) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.FUNCTIONAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    observation = ObservationStore(tmp_path).load_selected({observations()[-1].observation_id})[0]
    evidence = auditor_evidence.ScenarioEvidence(
        images=tuple(
            auditor_evidence.GoalEvidenceSource(
                observation_id=f"observation_{index}",
                labels=("Current-run evidence.",),
                screenshot=observation.redacted_screenshot,
            )
            for index in range(11)
        ),
        owner_counts={
            auditor_module.RequiredEvidenceOwner.APPLICATION_OBSERVATION: 11,
            auditor_module.RequiredEvidenceOwner.ACTION_OUTCOME: 1,
        },
        action_facts=("The action completed.",),
        capability_facts=(),
    )
    monkeypatch.setattr(
        auditor_module,
        "collect_outcome_evidence",
        lambda unused_root, unused_suite, unused_outcome, unused_result: evidence,
    )
    model = StubPlannerModel(None)
    constraints = _constraints().model_copy(update={"max_images": 1})

    audit = ScenarioAuditorService(model, "auditor-primary", constraints).audit(
        tmp_path,
        suite,
        scenario,
        execution,
    )

    assert audit.verdict is ScenarioAuditVerdict.UNVERIFIED
    assert model.requests == []
    assert all("auditor_image_capacity" in item.reason for item in audit.obligations)


def test_compatible_outcomes_share_one_bounded_audit_request(tmp_path) -> None:
    suite = _generated_suite()
    baseline = next(
        scenario
        for scenario in suite.scenarios
        if scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    obligations = tuple(
        item
        for item in baseline.obligations
        if item.category in {CoverageCategory.FUNCTIONAL, CoverageCategory.STATE}
    )
    scenario = baseline.model_copy(update={"obligations": obligations})
    suite = suite.model_copy(update={"scenarios": (scenario,)})
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)

    outcomes = tuple(
        outcome for obligation in obligations for outcome in obligation.expected_outcomes
    )
    model = StubPlannerModel(
        {
            "summary": "The current-run evidence proves the supplied outcomes.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "satisfied",
                    "reason": "The current-run evidence proves the expected outcome.",
                }
                for outcome in outcomes
            ],
        }
    )
    audit = ScenarioAuditorService(
        model,
        "auditor-primary",
        _constraints(),
    ).audit(tmp_path, suite, scenario, execution)

    assert audit.verdict is ScenarioAuditVerdict.PASSED
    assert audit.summary == (
        "Independent scenario audit completed across 1 bounded evidence request(s)."
    )
    assert len(model.requests) == 1


def test_visual_batches_run_in_parallel_and_one_render_error_fails(
    tmp_path,
    monkeypatch,
) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.UI_VISUAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    observation = ObservationStore(tmp_path).load_selected({observations()[-1].observation_id})[0]
    evidence = auditor_evidence.ScenarioEvidence(
        images=tuple(
            auditor_evidence.GoalEvidenceSource(
                observation_id=f"visual_observation_{index}",
                labels=(f"Visual surface {index}.",),
                screenshot=observation.redacted_screenshot,
            )
            for index in range(11)
        ),
        owner_counts={
            auditor_module.RequiredEvidenceOwner.APPLICATION_OBSERVATION: 11,
            auditor_module.RequiredEvidenceOwner.ACTION_OUTCOME: 1,
        },
        action_facts=("The action completed.",),
        capability_facts=(),
    )
    monkeypatch.setattr(
        auditor_module,
        "collect_visual_outcome_evidence",
        lambda unused_root, unused_suite, unused_outcome, unused_result: evidence,
    )
    barrier = Barrier(2, timeout=2)
    observed_batches: list[int] = []

    def audit_batch(unused_self, unused_scenario, batch):
        observed_batches.append(batch.batch_index)
        barrier.wait()
        outcome = batch.targets[0].outcome
        failed = batch.batch_index == 1
        finding = visual_finding(
            outcome.outcome_id,
            batch.evidence.images[0].observation_id,
        )
        return auditor_module.AuditBatchAttempt(
            output=auditor_module.ScenarioAuditorOutput(
                summary="The visual batch was inspected.",
                outcomes=(
                    auditor_module.ScenarioAuditOutcomeOutput(
                        outcome_id=outcome.outcome_id,
                        status=(
                            AuditObligationStatus.UNMET
                            if failed
                            else AuditObligationStatus.SATISFIED
                        ),
                        reason=(
                            "A visible overflow diagnostic is present."
                            if failed
                            else "No visible UI defect appears in this batch."
                        ),
                    ),
                ),
                findings=(finding,) if failed else (),
            )
        )

    monkeypatch.setattr(ScenarioAuditorService, "_audit_batch", audit_batch)
    constraints = _constraints().model_copy(update={"max_images": 1})
    audit = ScenarioAuditorService(
        StubPlannerModel(None),
        "auditor-primary",
        constraints,
    ).audit(tmp_path, suite, scenario, execution)

    assert audit.verdict is ScenarioAuditVerdict.FAILED
    assert set(observed_batches) == {0, 1}
    assert "overflow diagnostic" in audit.obligations[0].reason
