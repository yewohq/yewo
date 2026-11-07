"""Coverage semantics for independent visual scenario audit."""

from yewo.observations import ObservationStore
from yewo.project_config import CoverageCategory
from yewo.reporting import ScenarioAuditVerdict
from yewo.scenarios import ScenarioAuditorService
from yewo.scenarios import auditor as auditor_module
from yewo.scenarios import auditor_evidence

from agent.planning.support import StubPlannerModel
from agent.replay.support import observations
from scenarios.test_auditor import (
    _constraints,
    _persist_final_observation,
    _selected_scenario,
)
from scenarios.test_coordinator import _coordinator


def test_missing_visual_surface_coverage_cannot_pass(tmp_path, monkeypatch) -> None:
    suite, scenario = _selected_scenario(CoverageCategory.UI_VISUAL)
    coordinator, unused_task_executor = _coordinator(tmp_path, [])
    execution = coordinator.execute(suite, scenario)
    _persist_final_observation(tmp_path)
    observation = ObservationStore(tmp_path).load_selected({observations()[-1].observation_id})[0]
    evidence = auditor_evidence.ScenarioEvidence(
        images=(
            auditor_evidence.GoalEvidenceSource(
                observation_id=observation.observation_id,
                labels=("One covered visual surface.",),
                screenshot=observation.redacted_screenshot,
            ),
        ),
        owner_counts={
            auditor_module.RequiredEvidenceOwner.APPLICATION_OBSERVATION: 1,
            auditor_module.RequiredEvidenceOwner.ACTION_OUTCOME: 1,
        },
        action_facts=("The action completed.",),
        capability_facts=(),
        required_surface_ids=("covered_surface", "missing_surface"),
        covered_surface_ids=("covered_surface",),
        missing_surface_ids=("missing_surface",),
    )
    monkeypatch.setattr(
        auditor_module,
        "collect_visual_outcome_evidence",
        lambda unused_root, unused_suite, unused_outcome, unused_result: evidence,
    )
    outcome = scenario.obligations[0].expected_outcomes[0]
    model = StubPlannerModel(
        {
            "summary": "The supplied frame has no visible defect.",
            "outcomes": [
                {
                    "outcome_id": outcome.outcome_id,
                    "status": "satisfied",
                    "reason": "The supplied frame has no visible defect.",
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
    assert "missing_surface" in audit.obligations[0].reason
