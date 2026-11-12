"""When the analysis manager reanalyzes a screen and when it reuses a stored variant."""

from observations.factories import build_observation
from agent.planning.support import StubPlannerModel
from agent.screen_analysis.test_service import _output
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    CurrentScreenAnalysisManager,
    ModelScreenAnalyzer,
    ScreenAnalysisError,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    LiteralText,
)
from yewo.artifacts import ArtifactRetention
from yewo.models import ModelInputError
from yewo.observations import Observation


class _CountingAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        self.calls += 1
        return CurrentScreenAnalysis(
            analysis_id=f"analysis-{self.calls}",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
            elements=(),
        )


def test_manager_reuses_same_variant_and_reanalyzes_keyboard_variant() -> None:
    source = _CountingAnalyzer()
    manager = CurrentScreenAnalysisManager(
        source,
        maximum_variants=2,
        identifier_factory=lambda: "analysis-rebound",
    )
    first = build_observation()
    same = first.model_copy(
        update={
            "observation_id": "obs-2",
            "screenshot": first.screenshot.model_copy(update={"sha256": "b" * 64}),
        }
    )
    keyboard = first.model_copy(update={"observation_id": "obs-3", "keyboard_visible": True})

    assert manager.required_model_requests(first) == 1
    first_analysis = manager.analyze(first)
    assert first_analysis is not None
    assert manager.required_model_requests(same) == 0
    same_analysis = manager.analyze(same)
    assert same_analysis is not None and same_analysis.observation_id == "obs-2"
    assert same_analysis.analysis_id == "analysis-rebound"
    assert manager.required_model_requests(keyboard) == 1
    manager.analyze(keyboard)
    assert source.calls == 2


class _FailingAnalyzer(_CountingAnalyzer):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self._failures = failures

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        self.calls += 1
        if self.calls <= self._failures:
            raise ScreenAnalysisError("Temporary analyzer failure.")
        return CurrentScreenAnalysis(
            analysis_id=f"analysis-{self.calls}",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
            elements=(),
        )


class _InvalidInputAnalyzer(_CountingAnalyzer):
    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        del observation
        self.calls += 1
        raise ModelInputError("The analyzer request exceeds its input contract.")


def test_manager_can_be_configured_for_a_second_source_attempt() -> None:
    source = _FailingAnalyzer(failures=1)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 1
    assert manager.analyze(observation) is not None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 2


def test_manager_contains_model_input_failure_from_optional_analysis() -> None:
    source = _InvalidInputAnalyzer()
    manager = CurrentScreenAnalysisManager(source)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 1


def test_manager_stops_after_bounded_failures_for_one_variant() -> None:
    source = _FailingAnalyzer(failures=3)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert source.calls == 2


def test_manager_default_caches_failure_after_sources_bounded_repair() -> None:
    source = _FailingAnalyzer(failures=2)
    manager = CurrentScreenAnalysisManager(source)
    observation = build_observation()

    assert manager.analyze(observation) is None
    assert manager.required_model_requests(observation) == 0
    assert manager.analyze(observation) is None
    assert source.calls == 1


def test_manager_reanalyzes_after_verified_text_entry() -> None:
    model = StubPlannerModel(_output())
    identifiers = iter(("request-1", "analysis-1", "element-1", "analysis-rebound"))
    manager = CurrentScreenAnalysisManager(
        ModelScreenAnalyzer(
            model,
            "screen-analyzer-primary",
            identifier_factory=identifiers.__next__,
        ),
        identifier_factory=identifiers.__next__,
    )
    before = build_observation()
    analysis = manager.analyze(before)
    assert analysis is not None
    after = before.model_copy(
        update={
            "observation_id": "observation-after-type",
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
            "screenshot": before.screenshot.model_copy(update={"sha256": "b" * 64}),
        }
    )
    candidate = ActionProposal(
        proposal_id="proposal-type",
        source_observation_id=before.observation_id,
        action=ActionKind.TYPE,
        target_element_ref=analysis.elements[0].element_ref,
        text=LiteralText(value="new value"),
        expected_outcome="Primary field contains the entered text.",
        reason="Enter the required value.",
    )
    manager.observe_outcome(candidate, before, after, _observed_outcome(candidate, after))

    assert manager.required_model_requests(after) == 1
    assert manager.request_reason(after) == "structural_change_after_type"
    assert len(model.requests) == 1


def test_manager_reanalyzes_after_keyboard_layout_or_scroll_changes() -> None:
    source = _CountingAnalyzer()
    manager = CurrentScreenAnalysisManager(source)
    before = build_observation()
    assert manager.analyze(before) is not None
    keyboard = before.model_copy(
        update={
            "observation_id": "observation-keyboard",
            "keyboard_visible": not before.keyboard_visible,
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )
    focus = ActionProposal(
        proposal_id="proposal-focus",
        source_observation_id=before.observation_id,
        action=ActionKind.FOCUS,
        target_expression="the visible field",
        expected_outcome="The field gains focus.",
        reason="Prepare the field for entry.",
    )
    manager.observe_outcome(focus, before, keyboard, _observed_outcome(focus, keyboard))

    assert manager.required_model_requests(keyboard) == 1
    assert manager.request_reason(keyboard) == "keyboard_layout_changed"


def test_failed_analysis_budget_does_not_poison_a_new_dynamic_state_variant() -> None:
    source = _FailingAnalyzer(failures=1)
    manager = CurrentScreenAnalysisManager(source, maximum_attempts_per_variant=2)
    before = build_observation()
    assert manager.analyze(before) is None
    after = before.model_copy(
        update={
            "observation_id": "observation-after-type",
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "1111111111111111"}
            ),
        }
    )
    candidate = ActionProposal(
        proposal_id="proposal-type",
        source_observation_id=before.observation_id,
        action=ActionKind.TYPE,
        target_expression="the focused field",
        text=LiteralText(value="value"),
        expected_outcome="The field contains the entered text.",
        reason="Enter the value.",
    )
    manager.observe_outcome(candidate, before, after, _observed_outcome(candidate, after))

    assert manager.required_model_requests(after) == 1
    assert manager.analyze(after) is not None
    assert manager.required_model_requests(after) == 0
    assert source.calls == 2


def _observed_outcome(
    proposal: ActionProposal,
    after: Observation,
) -> ActionOutcome:
    return ActionOutcome(
        outcome_id=f"outcome-{proposal.proposal_id}",
        authorization_id=f"authorization-{proposal.proposal_id}",
        receipt_id=f"receipt-{proposal.proposal_id}",
        action=proposal.action,
        source_observation_id=proposal.source_observation_id,
        settled_observation_id=after.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The settled observation contains action-outcome evidence.",
        retention=ArtifactRetention.RUN,
    )
