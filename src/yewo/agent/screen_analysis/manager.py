"""Bounded reuse of observation-scoped semantic screen analyses."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from yewo.agent.actuation import (
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
    ActionKind,
)
from yewo.artifacts import new_record_identifier
from yewo.agent.screen_analysis.catalog import ScreenSemanticRole, ScreenWorkflowRole
from yewo.agent.screen_analysis.records import CurrentScreenAnalysis, ScreenElement
from yewo.agent.screen_analysis.service import (
    ScreenAnalysisCapacityError,
    ScreenAnalysisError,
    ScreenAnalysisSource,
)
from yewo.agent.screen_analysis.store import ScreenAnalysisStore
from yewo.models import ModelInputError, ModelRole
from yewo.observations import Observation, ScreenMatcher

ScreenSignature = tuple[object, ...]
CapacityFailureKey = tuple[ModelRole, str, str, str]


@runtime_checkable
class _CapacityIdentifiedSource(Protocol):
    def capacity_identity(self) -> tuple[str, str]: ...


class ScreenAnalysisFailureLedger:
    """Share capacity failures across agent phases in one workflow run."""

    def __init__(self) -> None:
        self._failures: set[CapacityFailureKey] = set()

    def record(self, key: CapacityFailureKey) -> None:
        """Record one exhausted profile-contract-screen combination."""
        self._failures.add(key)

    def contains(self, key: CapacityFailureKey | None) -> bool:
        """Return whether an identical analyzer request already exhausted."""
        return key is not None and key in self._failures


@dataclass(frozen=True)
class _AnalysisCacheEntry:
    analysis: CurrentScreenAnalysis | None = None
    failed_attempts: int = 0


def _signature(observation: Observation) -> ScreenSignature:
    return ScreenMatcher.analysis_cache_key(observation.presentation_key)


def _fingerprint_identity(observation: Observation) -> str:
    fingerprint = observation.screen_fingerprint
    return ":".join(
        (
            fingerprint.value,
            fingerprint.masked_pixels_sha256,
            fingerprint.system_region_mask_sha256 or "none",
        )
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
        store: ScreenAnalysisStore | None = None,
        failure_ledger: ScreenAnalysisFailureLedger | None = None,
    ) -> None:
        if maximum_variants < 1 or maximum_attempts_per_variant < 1:
            raise ValueError("Screen analysis bounds must be positive.")
        self._source = source
        self._maximum_variants = maximum_variants
        self._maximum_attempts_per_variant = maximum_attempts_per_variant
        self._identifier_factory = identifier_factory
        self._store = store
        self._failure_ledger = failure_ledger or ScreenAnalysisFailureLedger()
        self._entries: OrderedDict[ScreenSignature, _AnalysisCacheEntry] = OrderedDict()
        self._request_reasons: OrderedDict[ScreenSignature, str] = OrderedDict()

    def required_model_requests(self, observation: Observation) -> int:
        if self._failure_ledger.contains(self._capacity_key(observation)):
            return 0
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

    def capacity_failure_reason(self, observation: Observation) -> str | None:
        """Return a typed terminal reason for an already exhausted request."""
        key = self._capacity_key(observation)
        if not self._failure_ledger.contains(key) or key is None:
            return None
        role, profile_id, contract_id, fingerprint = key
        return (
            "screen_analysis_capacity_exhausted: "
            f"role={role.value} profile={profile_id} contract={contract_id} "
            f"fingerprint={fingerprint}"
        )

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis | None:
        if self._failure_ledger.contains(self._capacity_key(observation)):
            return None
        signature = _signature(observation)
        entry = self._entries.get(signature, _AnalysisCacheEntry())
        if entry.analysis is None and entry.failed_attempts < self._maximum_attempts_per_variant:
            try:
                analysis = self._source.analyze(observation)
            except ScreenAnalysisCapacityError as error:
                self._failure_ledger.record(
                    (
                        ModelRole.SCREEN_ANALYZER,
                        error.profile_id,
                        error.response_contract_id,
                        _fingerprint_identity(observation),
                    )
                )
                entry = _AnalysisCacheEntry(failed_attempts=entry.failed_attempts + 1)
            except (ModelInputError, ScreenAnalysisError):
                entry = _AnalysisCacheEntry(failed_attempts=entry.failed_attempts + 1)
            else:
                entry = _AnalysisCacheEntry(analysis=analysis)
                self._persist(analysis)
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
        rebound = self._bind(cached, observation)
        self._persist(rebound)
        return rebound

    def _capacity_key(self, observation: Observation) -> CapacityFailureKey | None:
        if not isinstance(self._source, _CapacityIdentifiedSource):
            return None
        profile_id, contract_id = self._source.capacity_identity()
        return (
            ModelRole.SCREEN_ANALYZER,
            profile_id,
            contract_id,
            _fingerprint_identity(observation),
        )

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
            self._persist(analysis)
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
        if not ScreenMatcher.metadata_matches(
            before.presentation_key,
            after.presentation_key,
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
        cleared = ScreenElement.model_validate({**target.model_dump(mode="python"), "state": None})
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
                "screenshot_sha256": observation.screenshot.sha256,
                "presentation": observation.presentation_key,
            }
        )

    def _persist(self, analysis: CurrentScreenAnalysis) -> None:
        if self._store is not None:
            self._store.persist(analysis)


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
