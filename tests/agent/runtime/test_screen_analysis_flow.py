"""Offline integration of analysis, planning identity, grounding, and dispatch."""

from pathlib import Path

from yewo.agent.actuation import ActionKind
from yewo.agent.runtime.records import AgentTerminalReason, build_runtime_grounding_request
from yewo.agent.screen_analysis import CurrentScreenAnalysisManager, ModelScreenAnalyzer
from yewo.grounding import (
    GroundingOutputShape,
    GroundingRequest,
    GroundingResult,
    GroundingTargetOperation,
    NormalizedRegion,
    ScreenAnalysisGroundingCandidate,
)

from agent.actuation.support import proposal
from agent.planning.support import StubPlannerModel
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)
from grounding.support import DeterministicGrounder


class _RecordingGrounder:
    def __init__(self) -> None:
        self.requests: list[GroundingRequest] = []
        self._delegate = DeterministicGrounder(100, 200)

    def required_model_requests(self, request: GroundingRequest) -> int:
        return self._delegate.required_model_requests(request)

    def ground(self, request: GroundingRequest) -> GroundingResult:
        self.requests.append(request)
        return self._delegate.ground(request)


def test_scroll_request_ignores_a_point_only_analysis_candidate() -> None:
    observation = _observation("obs_1", "0000000000000000")
    request = build_runtime_grounding_request(
        "grounding-1",
        "the visible scroll region",
        GroundingTargetOperation.SCROLL_REGION,
        observation,
        None,
        ScreenAnalysisGroundingCandidate(
            analysis_id="analysis-1",
            element_ref="element-1",
            screenshot_sha256=observation.screenshot.sha256,
            region=NormalizedRegion(left=0.1, top=0.1, right=0.9, bottom=0.9),
        ),
    )

    assert request is not None
    assert request.output_shape is GroundingOutputShape.REGION
    assert request.screen_analysis_candidate is None


def test_offline_screen_analysis_flow_reaches_grounded_dispatch(tmp_path: Path) -> None:
    analyzer_model = StubPlannerModel(
        {
            "screen_summary": "Visible application form.",
            "elements": [
                {
                    "analysis_key": "primary-field",
                    "name": "Primary field",
                    "kind": {"visual": "input_surface", "semantic": "text_field"},
                    "state": "empty",
                    "confidence": "high",
                    "role_alternatives": [],
                    "input_surface_key": "primary-input",
                    "parent_key": None,
                    "position_in_parent": None,
                    "visual_cues": ["below Section heading", "above Secondary field"],
                    "region_hint": {
                        "left": 0.4,
                        "top": 0.4,
                        "right": 0.6,
                        "bottom": 0.6,
                    },
                }
            ],
        }
    )
    analyzer_ids = iter(("analysis-request-1", "analysis-1", "analysis-request-2", "analysis-2"))
    analyzer = ModelScreenAnalyzer(
        analyzer_model,
        "screen-analyzer-primary",
        identifier_factory=lambda: next(analyzer_ids),
    )
    planner = SequencePlanner(
        [
            proposal(
                ActionKind.FOCUS,
                target_expression=None,
                target_element_ref="element-1",
            ),
            proposal(ActionKind.CANNOT_PROGRESS),
        ]
    )
    first = _observation("obs_1", "0000000000000000")
    focused = _observation(
        "obs_2",
        "1111111111111111",
        keyboard_visible=True,
        offset_seconds=1,
    )
    focused_settled = _observation(
        "obs_3",
        "1111111111111111",
        keyboard_visible=True,
        offset_seconds=2,
    )
    grounder = _RecordingGrounder()
    device = RuntimeDevice()

    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence([first, focused, focused_settled]),
        device,
        grounder=grounder,
        screen_analysis=CurrentScreenAnalysisManager(analyzer),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    first_proposal = result.steps[0].proposal
    assert first_proposal.target_element_ref == "element-1"
    assert "visible input surface for 'Primary field'" in (first_proposal.target_expression or "")
    assert "below Section heading" in grounder.requests[0].expression
    candidate = grounder.requests[0].screen_analysis_candidate
    assert candidate is not None
    assert candidate.analysis_id == "analysis-1"
    assert candidate.element_ref == "element-1"
    assert device.calls[0][0] == "focus"
    assert planner.requests[0].current_screen_analysis is not None
    assert len(analyzer_model.requests) == 2
    assert result.usage.planning_requests == 4
    assert result.usage.grounding_requests == 1


def test_screen_analysis_budget_stops_before_a_zero_cost_planner(tmp_path: Path) -> None:
    analyzer_model = StubPlannerModel(
        {"screen_summary": "Visible application screen.", "elements": []}
    )
    analyzer = ModelScreenAnalyzer(
        analyzer_model,
        "screen-analyzer-primary",
        identifier_factory=lambda: "unused-analysis-id",
    )
    planner = SequencePlanner(
        [proposal(ActionKind.WAIT)],
        model_requests_per_plan=0,
    )
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        screen_analysis=CurrentScreenAnalysisManager(analyzer),
        initial_planning_requests=1,
    ).run(
        _task(),
        _mode_policy(maximum_model_requests=1),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.planning_requests == 1
    assert analyzer_model.requests == []
    assert planner.requests == []
