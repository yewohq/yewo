"""Bounded recovery and aggregation tests for scenario auditing."""

from yewo.models import ModelCallDisposition
from yewo.observations import ObservationStore
from yewo.project_config import CoverageCategory
from yewo.reporting import AuditObligationStatus, ScenarioAuditVerdict
from yewo.scenarios import ScenarioAuditorService
from yewo.scenarios import auditor as auditor_module
from yewo.scenarios import auditor_evidence

from agent.planning.support import StubPlannerModel
from agent.replay.support import observations
from scenarios.auditor_support import visual_finding
from scenarios.test_auditor import (
    _constraints,
    _persist_final_observation,
    _selected_scenario,
)
from scenarios.test_coordinator import _coordinator


def test_reasoning_exhaustion_splits_evidence_and_preserves_failure(
    tmp_path,
    monkeypatch,
) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.UI_VISUAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    observation = ObservationStore(tmp_path).load_selected({observations()[-1].observation_id})[0]
    images = tuple(
        auditor_evidence.GoalEvidenceSource(
            observation_id=f"visual_observation_{index}",
            labels=(f"Visual surface {index}.",),
            screenshot=observation.screenshot,
        )
        for index in range(11)
    )
    evidence = auditor_evidence.ScenarioEvidence(
        images=images,
        owner_counts={
            auditor_module.RequiredEvidenceOwner.APPLICATION_OBSERVATION: 11,
            auditor_module.RequiredEvidenceOwner.ACTION_OUTCOME: 1,
        },
        action_facts=("The action completed.",),
        capability_facts=(),
        visual_episodes=(
            auditor_evidence.VisualEvidenceEpisode(
                episode_id="visual_episode_001",
                label="Sign-in screen states",
                images=images,
            ),
        ),
    )
    monkeypatch.setattr(
        auditor_module,
        "collect_visual_outcome_evidence",
        lambda unused_root, unused_suite, unused_outcome, unused_result: evidence,
    )
    observed_source_counts: list[int] = []

    def audit_batch(unused_self, unused_scenario, batch):
        source_count = len(batch.evidence.images)
        observed_source_counts.append(source_count)
        if source_count > 2:
            return auditor_module.AuditBatchAttempt(
                output=None,
                disposition=ModelCallDisposition.REASONING_EXHAUSTED,
            )
        outcome = batch.targets[0].outcome
        is_failure = source_count == 2
        finding = visual_finding(
            outcome.outcome_id,
            batch.evidence.images[0].observation_id,
        )
        return auditor_module.AuditBatchAttempt(
            output=auditor_module.ScenarioAuditorOutput(
                outcomes=(
                    auditor_module.ScenarioAuditOutcomeOutput(
                        outcome_id=outcome.outcome_id,
                        status=(
                            AuditObligationStatus.UNMET
                            if is_failure
                            else AuditObligationStatus.SATISFIED
                        ),
                        reason=(
                            "A visible overflow diagnostic is present."
                            if is_failure
                            else "No visible defect appears in this evidence."
                        ),
                    ),
                ),
                findings=(finding,) if is_failure else (),
            ),
            disposition=ModelCallDisposition.COMPLETED,
        )

    monkeypatch.setattr(ScenarioAuditorService, "_audit_batch", audit_batch)
    constraints = _constraints().model_copy(update={"max_images": 1})

    audit = ScenarioAuditorService(
        StubPlannerModel(None),
        "auditor-primary",
        constraints,
    ).audit(tmp_path, suite, scenario, execution)

    assert audit.verdict is ScenarioAuditVerdict.FAILED
    assert audit.obligations[0].status is AuditObligationStatus.UNMET
    assert "overflow diagnostic" in audit.obligations[0].reason
    assert 4 in observed_source_counts
    assert max(observed_source_counts) == 4
    assert any(count < 4 for count in observed_source_counts)
