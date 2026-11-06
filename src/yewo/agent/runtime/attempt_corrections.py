"""Admit one bounded retry of an attempt that produced no observable change."""

from __future__ import annotations

from yewo.agent.actuation import ActionProposal, ActionRefusal, ActionRefusalCode
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.runtime.interaction_intent import same_interaction_family
from yewo.agent.memory.screens import observations_match
from yewo.observations import Observation


class FailedAttemptCorrections:
    """Own the no-effect attempt log and answer which retries it still allows."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self._attempts: list[tuple[ActionProposal, Observation]] = []
        self._latest_feedback_proposal_id: str | None = None

    def start(self) -> None:
        """Drop evidence from an earlier application lifetime."""
        self._attempts = []
        self._latest_feedback_proposal_id = None

    def record_no_change(self, proposal: ActionProposal, observation: Observation) -> None:
        """Retain one dispatched action that left the screen unchanged."""
        self._attempts.append((proposal, observation))

    def record_feedback(self, proposal_id: str | None) -> None:
        """Track the proposal that produced the most recent trusted feedback."""
        self._latest_feedback_proposal_id = proposal_id

    def correctable_proposal_id(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> str | None:
        """Name the no-effect attempt whose recorded point this action may re-ground."""
        corrected = self._corrected_proposal_ids()
        return next(
            (
                candidate.proposal_id
                for candidate, screen in reversed(self._attempts)
                if same_interaction_family(proposal, candidate)
                and observations_match(observation, screen, self._comparison)
                and candidate.grounding_correction_for_proposal_id is None
                and candidate.proposal_id not in corrected
            ),
            None,
        )

    def _corrected_proposal_ids(self) -> set[str]:
        """Name every attempt that already spent its one re-grounding."""
        return {
            candidate.grounding_correction_for_proposal_id
            for candidate, _ in self._attempts
            if candidate.grounding_correction_for_proposal_id is not None
        }

    def corrects_one_failed_grounding(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> bool:
        """Report whether this proposal re-grounds one earlier no-effect attempt."""
        original_id = proposal.grounding_correction_for_proposal_id
        if original_id is None:
            return False
        return any(
            candidate.proposal_id == original_id
            and observations_match(observation, screen, self._comparison)
            for candidate, screen in self._attempts
        )

    def refinement_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        """Refuse an interaction refinement that does not refine the preceding attempt."""
        original_id = proposal.interaction_refinement_for_proposal_id
        if original_id is None:
            return None
        original = next(
            (
                candidate
                for candidate, screen in reversed(self._attempts)
                if candidate.proposal_id == original_id
                and observations_match(observation, screen, self._comparison)
            ),
            None,
        )
        reason = _refinement_error(proposal, original, original_id, self._latest_feedback_proposal_id)
        if reason is None:
            return None
        return ActionRefusal(
            proposal_id=proposal.proposal_id,
            code=ActionRefusalCode.INVALID_CONSTRAINT,
            reason=reason,
        )


def _refinement_error(
    proposal: ActionProposal,
    original: ActionProposal | None,
    original_id: str,
    latest_feedback_proposal_id: str | None,
) -> str | None:
    if original is None or original_id != latest_feedback_proposal_id:
        return "An interaction refinement must reference the immediately preceding no-effect proposal."
    if original.interaction_refinement_for_proposal_id is not None:
        return "An interaction refinement cannot refine another interaction refinement."
    if original.grounding_correction_for_proposal_id is not None:
        return "An interaction refinement cannot refine a grounding correction."
    if proposal.action is not original.action:
        return "An interaction refinement must use the original action."
    if proposal.target_expression == original.target_expression:
        return "An interaction refinement must identify a distinct visible surface."
    return None


__all__ = ["FailedAttemptCorrections"]
