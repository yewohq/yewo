"""Bounded reuse of observation-scoped semantic screen analyses."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.actuation import (
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
    ActionKind,
)
from yewo.artifacts import new_record_identifier
from yewo.agent.screen_analysis.catalog import ScreenSemanticRole, ScreenWorkflowRole
from yewo.agent.screen_analysis.records import CurrentScreenAnalysis, ScreenElement
from yewo.agent.screen_analysis.service import ScreenAnalysisError, ScreenAnalysisSource
from yewo.models import ModelInputError
from yewo.observations import Observation

ScreenSignature = tuple[str, str | None, bool | None, str | None]


@dataclass(frozen=True)
class _AnalysisCacheEntry:
    analysis: CurrentScreenAnalysis | None = None
    failed_attempts: int = 0


def _signature(observation: Observation) -> ScreenSignature:
    orientation = observation.orientation.value if observation.orientation is not None else None
    return (
        observation.screen_fingerprint.value,
        observation.platform_context,
        observation.keyboard_visible,
        orientation,
    )


class CurrentScreenAnalysisManager:
    """Analyze each distinct screen variant once and retain only a bounded set."""

    def __init__(
        self,
        source: ScreenAnalysisSource,
        *,
        maximum_variants: int = 8,
        maximum_attempts_per_variant: int = 1,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        if maximum_variants < 1 or maximum_attempts_per_variant < 1:
            raise ValueError("Screen analysis bounds must be positive.")
        self._source = source
        self._maximum_variants = maximum_variants
        self._maximum_attempts_per_variant = maximum_attempts_per_variant
        self._identifier_factory = identifier_factory
        self._entries: OrderedDict[ScreenSignature, _AnalysisCacheEntry] = OrderedDict()
        self._request_reasons: OrderedDict[ScreenSignature, str] = OrderedDict()

    def required_model_requests(self, observation: Observation) -> int:
        entry = self._entries.get(_signature(observation))
        if entry is not None and (
            entry.analysis is not None
            or entry.failed_attempts >= self._maximum_attempts_per_variant
        ):
            return 0
        return self._source.required_model_requests(observation)

    def request_reason(self, observation: Observation) -> str:
        """Describe why this observation needs another analyzer request."""
        signature = _signature(observation)
        entry = self._entries.get(signature)
        if entry is None:
            return self._request_reasons.get(signature, "structural_variant_cache_miss")
        if entry.analysis is None:
            return "repair_after_malformed_analysis"
        return "cached_structural_variant"

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis | None:
        signature = _signature(observation)
        entry = self._entries.get(signature, _AnalysisCacheEntry())
        if entry.analysis is None and entry.failed_attempts < self._maximum_attempts_per_variant:
            try:
                analysis = self._source.analyze(observation)
            except (ModelInputError, ScreenAnalysisError):
                entry = _AnalysisCacheEntry(failed_attempts=entry.failed_attempts + 1)
            else:
                entry = _AnalysisCacheEntry(analysis=analysis)
            self._entries[signature] = entry
            self._request_reasons.pop(signature, None)
            while len(self._entries) > self._maximum_variants:
                removed_signature, unused = self._entries.popitem(last=False)
                self._request_reasons.pop(removed_signature, None)
        else:
            self._entries.move_to_end(signature)
        cached = entry.analysis
        if cached is None:
            return None
        if cached.observation_id == observation.observation_id:
            return cached
        return self._bind(cached, observation)

    def observe_outcome(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
    ) -> None:
        """Reuse one structural map across a verified dynamic-state-only change."""
        before_signature = _signature(before)
        after_signature = _signature(after)
        entry = self._entries.get(before_signature)
        if entry is None:
            self._record_request_reason(after_signature, "structural_variant_cache_miss")
            return
        if (
            outcome.disposition is not ActionOutcomeDisposition.OBSERVED
            and entry.analysis is not None
            and _is_confirmation_target(entry.analysis, proposal)
        ):
            self.invalidate(before, "submission_refused")
            self.invalidate(after, "submission_refused")
            return
        if not self._can_reuse(proposal, before, after, outcome, entry.analysis):
            self._entries.pop(after_signature, None)
            self._record_request_reason(
                after_signature,
                _reanalyze_reason(proposal, before, after),
            )
            return
        analysis = entry.analysis
        if analysis is not None:
            analysis = self._clear_dynamic_state(analysis, proposal)
            analysis = self._bind(analysis, after)
        if after_signature != before_signature:
            self._entries.pop(before_signature, None)
        self._entries[after_signature] = _AnalysisCacheEntry(
            analysis=analysis,
            failed_attempts=entry.failed_attempts,
        )
        self._entries.move_to_end(after_signature)
        while len(self._entries) > self._maximum_variants:
            removed_signature, unused = self._entries.popitem(last=False)
            self._request_reasons.pop(removed_signature, None)

    def invalidate(self, observation: Observation, reason: str) -> None:
        """Require fresh analysis after evidence contradicts the cached hypothesis."""
        signature = _signature(observation)
        self._entries.pop(signature, None)
        self._record_request_reason(signature, reason)

    def _can_reuse(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
        analysis: CurrentScreenAnalysis | None,
    ) -> bool:
        if outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return False
        if analysis is None:
            return False
        if (
            before.platform_context != after.platform_context
            or before.keyboard_visible != after.keyboard_visible
            or before.orientation != after.orientation
            or before.redacted_screenshot.width_px != after.redacted_screenshot.width_px
            or before.redacted_screenshot.height_px != after.redacted_screenshot.height_px
        ):
            return False
        if proposal.action is not ActionKind.TAP:
            return False
        target = _target_element(analysis, proposal)
        return target is not None and target.kind.semantic in {
            ScreenSemanticRole.CHECKBOX,
            ScreenSemanticRole.RADIO_BUTTON,
            ScreenSemanticRole.SWITCH,
        }

    def _clear_dynamic_state(
        self,
        analysis: CurrentScreenAnalysis,
        proposal: ActionProposal,
    ) -> CurrentScreenAnalysis:
        """Keep the structural map without claiming an unobserved target state."""
        target = _target_element(analysis, proposal)
        if target is None or target.state is None:
            return analysis
        cleared = ScreenElement.model_validate(
            {**target.model_dump(mode="python"), "state": None}
        )
        return CurrentScreenAnalysis.model_validate(
            {
                **analysis.model_dump(mode="python"),
                "elements": tuple(
                    cleared if item.element_ref == target.element_ref else item
                    for item in analysis.elements
                ),
            }
        )

    def _record_request_reason(self, signature: ScreenSignature, reason: str) -> None:
        self._request_reasons[signature] = reason
        self._request_reasons.move_to_end(signature)
        while len(self._request_reasons) > self._maximum_variants:
            self._request_reasons.popitem(last=False)

    def _bind(
        self,
        analysis: CurrentScreenAnalysis,
        observation: Observation,
    ) -> CurrentScreenAnalysis:
        return analysis.model_copy(
            update={
                "analysis_id": self._identifier_factory(),
                "observation_id": observation.observation_id,
                "screen_fingerprint": observation.screen_fingerprint.value,
                "screenshot_sha256": observation.redacted_screenshot.sha256,
                "platform_context": observation.platform_context,
                "keyboard_visible": observation.keyboard_visible,
                "orientation": observation.orientation,
            }
        )


def _target_element(
    analysis: CurrentScreenAnalysis,
    proposal: ActionProposal,
) -> ScreenElement | None:
    if proposal.target_element_ref is None:
        return None
    return next(
        (item for item in analysis.elements if item.element_ref == proposal.target_element_ref),
        None,
    )


def _is_confirmation_target(
    analysis: CurrentScreenAnalysis,
    proposal: ActionProposal,
) -> bool:
    target = _target_element(analysis, proposal)
    if target is None or target.kind.semantic is not ScreenSemanticRole.BUTTON:
        return False
    return target.workflow_role is ScreenWorkflowRole.ADVANCE_STAGE


def _reanalyze_reason(
    proposal: ActionProposal,
    before: Observation,
    after: Observation,
) -> str:
    if before.keyboard_visible != after.keyboard_visible:
        return "keyboard_layout_changed"
    if proposal.action is ActionKind.SCROLL:
        return "scroll_revealed_new_region"
    if before.platform_context != after.platform_context:
        return "logical_screen_changed"
    return "structural_change_after_" + proposal.action.value
