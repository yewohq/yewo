"""Regression coverage for reconciliation evidence assembly."""

import json
from pathlib import Path

import pytest

from yewo.agent import AgentTask, AgentTerminalReason, ScreenComparisonPolicy
from yewo.agent.actuation import ActionRefusal, ActionRefusalCode
from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.replay import (
    ProcedureCompilationStatus,
    build_evidence_document,
)
from yewo.agent.replay.procedure.records import (
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.reconciliation.compilation import compile_completed_discovery_run
from yewo.agent.replay.reconciliation.records import EvidenceResult
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenAnalysisStore,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)
from yewo.models import TextModelContent

from agent.discovery.support import agent_result
from agent.planning.support import StubPlannerModel
from agent.replay.test_procedure_compilation import (
    GENERATED_CONDITION,
    _identifiers,
    _result,
)


def test_builder_keeps_refusal_code_when_crash_is_primary() -> None:
    result, before, after = _result()
    source = result.agent_result.steps[0]
    combined = source.model_copy(
        update={
            "refusal": ActionRefusal(
                proposal_id=source.proposal.proposal_id,
                code=ActionRefusalCode.DISCOVERY_INCOMPLETE,
                reason="Discovery has not established this control.",
            ),
            "application_failure": ActionApplicationFailure(
                classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
                failure_kinds=(ApplicationFailureKind.CRASH,),
                summary="The application crashed after the attempted action.",
            ),
        }
    )

    evidence = build_evidence_document(
        run_id="run_1",
        objective_id="objective_1",
        task=AgentTask(objective="Save one reusable record.", discovery_mode=True),
        steps=(combined,),
        observations=(before, after),
        identifier_factory=lambda: "evidence_1",
    )

    assert evidence.steps[0].result is EvidenceResult.CRASHED
    assert evidence.steps[0].refusal_code is ActionRefusalCode.DISCOVERY_INCOMPLETE


def test_compilation_uses_replay_screen_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    replay_agent_result = agent_result(
        before,
        after,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    )
    replay_source = replay_agent_result.steps[0]
    replay_step = replay_source.model_copy(
        update={
            "proposal": replay_source.proposal.model_copy(
                update={
                    "target_element_ref": "replay-control",
                    "target_expression": "the visible navigation control",
                }
            ),
        }
    )
    replay_agent_result = replay_agent_result.model_copy(
        update={
            "steps": (replay_step,),
            "screen_analysis_ids": ("replay-analysis",),
        }
    )
    replay = ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="source_evidence_1",
        source_run_id="source_run_1",
        setup_id="setup_1",
        application=before.application,
        terminal=ProcedureReplayTerminal.BUDGET_EXHAUSTED,
        procedure_step_count=1,
        completed_step_count=0,
        reason="The replay budget ended before completion.",
        agent_result=replay_agent_result,
    )
    result = result.model_copy(update={"procedure_replay": replay})
    ScreenAnalysisStore(tmp_path).persist(
        CurrentScreenAnalysis(
            analysis_id="replay-analysis",
            observation_id=before.observation_id,
            screenshot_sha256=before.redacted_screenshot.sha256,
            presentation=before.presentation_key,
            screen_summary="Visible navigation control.",
            elements=(
                ScreenElement(
                    element_ref="replay-control",
                    name="Records",
                    kind=ScreenElementKind(
                        visual=ScreenVisualKind.TEXT_AND_ICON,
                        semantic=ScreenSemanticRole.NAVIGATION_ITEM,
                    ),
                    confidence=ScreenElementConfidence.HIGH,
                    evidence="A labelled navigation control is visible.",
                ),
            ),
        )
    )
    model = StubPlannerModel(
        {
            "completion_conditions": [GENERATED_CONDITION],
            "procedure": [1],
            "milestones": {"0": 1},
            "decisions": [],
            "findings": [],
        }
    )
    monkeypatch.setattr(
        "yewo.agent.replay.reconciliation.compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    report = compile_completed_discovery_run(
        root=tmp_path,
        run_id="run_1",
        discovery_result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        identifier_factory=_identifiers("evidence_1", "model_request_1", "procedure_1"),
    )

    assert report.status is ProcedureCompilationStatus.COMPILED
    request_content = model.requests[0].messages[1].content[0]
    assert isinstance(request_content, TextModelContent)
    payload = json.loads(request_content.text.split("Evidence JSON follows:\n", 1)[1])
    replay_control = payload["evidence"]["steps"][0]["control"]
    assert replay_control["kind"]["semantic"] == "navigation_item"
    assert replay_control["kind_source"] == "screen_analysis"
