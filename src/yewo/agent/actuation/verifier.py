"""Pre-dispatch discovery decisions for actions that can advance a workflow."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json
import unicodedata
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

from yewo.agent.actuation import (
    ActionKind,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    RequiredControlBasis,
    UnresolvedControl,
)
from yewo.agent.actuation.verifier_prompt import (
    ACTION_VERIFIER_SYSTEM_PROMPT,
    ACTION_VERIFIER_USER_PROMPT,
)
from yewo.agent.planning.records import (
    PlanningRequest,
    planning_focus_context,
    planning_progress_context,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    element_affordance,
    element_may_be_targeted,
)
from yewo.agent.planning.history import (
    trusted_history_content,
    planning_screen_fact_context,
)
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)


ACTION_VERIFIER_MAX_COMPLETION_TOKENS = 8_192
_VERIFIED_ACTIONS = frozenset({ActionKind.TAP, ActionKind.BACK})


class ActionReadinessDisposition(StrEnum):
    """Whether the proposed discovery action may reach device dispatch."""

    ALLOW = "allow"
    REFUSE = "refuse"
    UNAVAILABLE = "unavailable"


class ActionReadinessRefusalBasis(StrEnum):
    """Which verification rule a refusal rests on."""

    INCOHERENT_PROPOSAL = "incoherent_proposal"
    UNRESOLVED_CONTROL = "unresolved_control"
    COMPLETED_WORK_REGRESSION = "completed_work_regression"
    REOPENED_STAGE = "reopened_stage"


class ActionReadinessDecision(BaseModel):
    """Model-authored readiness judgment without runtime-owned identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    disposition: ActionReadinessDisposition
    reason: str = Field(min_length=1, max_length=2_000)
    unresolved_controls: tuple[UnresolvedControl, ...] = Field(default=(), max_length=30)
    completed_work_items: tuple[str, ...] = Field(max_length=100)
    proposal_work_item: str | None = Field(min_length=1, max_length=4_000)
    restarts_completed_work_item: str | None = Field(
        min_length=1,
        max_length=4_000,
    )
    refusal_basis: ActionReadinessRefusalBasis | None = Field(
        default=None,
        description=(
            "The verification rule behind a refusal. Required for a refusal that carries no "
            "unresolved control and no restarted work item."
        ),
    )

    @model_validator(mode="after")
    def _validate_control_evidence(self, info: ValidationInfo) -> Self:
        unresolved_names = [control.control_name for control in self.unresolved_controls]
        if len(set(unresolved_names)) != len(unresolved_names):
            raise ValueError("Each unresolved control must be reported once.")
        context = info.context
        if isinstance(context, dict):
            interactive_names = context.get("interactive_control_names")
            if isinstance(interactive_names, (tuple, list, frozenset, set)) and any(
                unresolved.control_name not in interactive_names
                for unresolved in self.unresolved_controls
            ):
                raise ValueError(
                    "Each unresolved control must reference an analyzed interactive control."
                )
            trusted_evidence = context.get("required_control_evidence")
            if isinstance(trusted_evidence, dict) and any(
                control.requirement_evidence
                not in trusted_evidence.get(control.control_name, {}).get(
                    control.required_by.value,
                    (),
                )
                for control in self.unresolved_controls
            ):
                raise ValueError(
                    "Each unresolved control must cite trusted evidence that names that control."
                )
        if self.disposition is ActionReadinessDisposition.REFUSE:
            self._validate_refusal_evidence()
        elif (
            self.unresolved_controls
            or self.restarts_completed_work_item is not None
            or self.refusal_basis is not None
        ):
            raise ValueError("Only a refusal can identify unresolved evidence.")
        return self

    def _validate_refusal_evidence(self) -> None:
        """Require the evidence that the stated basis depends on, and no other."""
        basis = self.refusal_basis
        if basis is None:
            # An older prompt names no basis, so fall back to the evidence-bearing rule.
            if not self.unresolved_controls and self.restarts_completed_work_item is None:
                raise ValueError("A refusal must identify unresolved evidence.")
            return
        if basis is ActionReadinessRefusalBasis.UNRESOLVED_CONTROL:
            if not self.unresolved_controls:
                raise ValueError("An unresolved-control refusal must name one unresolved control.")
            return
        if basis is ActionReadinessRefusalBasis.COMPLETED_WORK_REGRESSION:
            if self.restarts_completed_work_item is None:
                raise ValueError("A completed-work refusal must name the restarted work item.")
            return
        if self.unresolved_controls or self.restarts_completed_work_item is not None:
            raise ValueError(f"A {basis.value} refusal cannot carry unresolved evidence.")

    @model_validator(mode="after")
    def _validate_work_items(self, info: ValidationInfo) -> Self:
        context = info.context
        if not isinstance(context, dict):
            return self
        allowed_work_items = context.get("allowed_work_items")
        completed_work_items = context.get("completed_work_items")
        allowed_items = set(allowed_work_items) if isinstance(allowed_work_items, tuple) else set()
        completed_items = (
            set(completed_work_items) if isinstance(completed_work_items, tuple) else set()
        )
        reported_items = set(self.completed_work_items)
        if self.proposal_work_item is not None:
            reported_items.add(self.proposal_work_item)
        if self.restarts_completed_work_item is not None:
            reported_items.add(self.restarts_completed_work_item)
        if not reported_items.issubset(allowed_items):
            raise ValueError("Copy each work item exactly from the progress context.")
        if allowed_items and self.proposal_work_item is None:
            raise ValueError("Set proposal_work_item to one exact work item.")
        if self.restarts_completed_work_item is not None and (
            self.disposition is not ActionReadinessDisposition.REFUSE
            or self.restarts_completed_work_item not in completed_items
        ):
            raise ValueError(
                "A restarted work item must already be completed and the disposition must refuse."
            )
        return self


class ActionReadinessAssessment(ActionReadinessDecision):
    """One independent pre-dispatch assessment bound to a proposal."""

    verification_id: RecordIdentifier
    proposal_id: RecordIdentifier

    def refusal(self) -> ActionRefusal | None:
        """Convert a refused assessment to the runtime's existing refusal contract."""
        if self.disposition is ActionReadinessDisposition.ALLOW:
            return None
        details = [
            f"{control.control_name}: {control.reason}" for control in self.unresolved_controls
        ]
        if self.restarts_completed_work_item is not None:
            details.append(
                f"The proposal restarts completed work: {self.restarts_completed_work_item}"
            )
        detail = "; ".join(details)
        reason = self.reason if not detail else f"{self.reason} Unresolved: {detail}"
        return ActionRefusal(
            proposal_id=self.proposal_id,
            code=(
                ActionRefusalCode.COMPLETED_WORK_REGRESSION
                if self.restarts_completed_work_item is not None
                or self.refusal_basis is ActionReadinessRefusalBasis.REOPENED_STAGE
                else (
                    ActionRefusalCode.DISCOVERY_INCOMPLETE
                    if self.disposition is ActionReadinessDisposition.REFUSE
                    else ActionRefusalCode.VERIFICATION_UNAVAILABLE
                )
            ),
            reason=reason,
        )


class ActionVerifier(Protocol):
    """Independently assess model-selected actions before discovery dispatch."""

    def required_model_requests(self, proposal: ActionProposal) -> int: ...

    def verify(
        self,
        planning_request: PlanningRequest,
        proposal: ActionProposal,
    ) -> ActionReadinessAssessment: ...


class ModelActionVerifier:
    """Use a dedicated model profile for discovery action readiness."""

    def __init__(
        self,
        model_client: ModelClient,
        model_profile_id: str,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._model_client = model_client
        self._model_profile_id = model_profile_id
        self._identifier_factory = identifier_factory

    def required_model_requests(self, proposal: ActionProposal) -> int:
        """Assess only pointer and back actions that can advance or leave a workflow."""
        return int(proposal.action in _VERIFIED_ACTIONS)

    def verify(
        self,
        planning_request: PlanningRequest,
        proposal: ActionProposal,
    ) -> ActionReadinessAssessment:
        """Return one fail-closed assessment tied to the proposal and screenshot."""
        verification_id = self._identifier_factory()
        task_context = {
            "objective": planning_request.objective,
            "completion_conditions": planning_request.completion_conditions,
            "known_facts": planning_request.known_facts,
            "declared_value_names": planning_request.declared_value_names,
        }
        context = {
            "open_work": planning_request.open_work,
            "screen_facts": [
                planning_screen_fact_context(fact) for fact in planning_request.screen_facts
            ],
            "progress_context": (
                planning_progress_context(planning_request.progress_context)
                if planning_request.progress_context is not None
                else None
            ),
            "planning_focus": (
                planning_focus_context(planning_request.planning_focus)
                if planning_request.planning_focus is not None
                else None
            ),
            "proposal": _action_verifier_proposal_context(proposal),
            "current_screen_analysis": (
                _action_verifier_screen_context(planning_request.current_screen_analysis)
                if planning_request.current_screen_analysis is not None
                else None
            ),
        }
        progress_context = planning_request.progress_context
        known_work_items = tuple(
            dict.fromkeys(
                (
                    *planning_request.open_work,
                    *(progress_context.completed_work_items if progress_context else ()),
                )
            )
        )
        completed_work_items = (
            progress_context.completed_work_items if progress_context is not None else ()
        )
        request = ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=self._model_profile_id,
            role=ModelRole.ACTION_VERIFIER,
            max_completion_tokens=ACTION_VERIFIER_MAX_COMPLETION_TOKENS,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text=ACTION_VERIFIER_SYSTEM_PROMPT),),
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(TextModelContent(text=ACTION_VERIFIER_USER_PROMPT),),
                    cache_prefix_end=True,
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(
                        *trusted_history_content(planning_request.history, include_text=False),
                        TextModelContent(
                            text="Task context for this run:\n"
                            + json.dumps(task_context, sort_keys=True, separators=(",", ":"))
                        ),
                        TextModelContent(
                            text="Current verification context:\n"
                            + json.dumps(context, sort_keys=True, separators=(",", ":"))
                        ),
                        ImageModelContent(
                            screenshot=planning_request.observation.redacted_screenshot
                        ),
                    ),
                ),
            ),
            response_contract=build_model_response_contract(
                "action-readiness-decision-v1",
                ActionReadinessDecision.model_json_schema(),
            ),
            sensitivity=planning_request.observation.redacted_screenshot.sensitivity,
            retention=planning_request.observation.redacted_screenshot.retention,
        )
        validation_context = {
            "allowed_work_items": known_work_items,
            "completed_work_items": completed_work_items,
            "required_control_evidence": _required_control_evidence(planning_request),
            **(
                {
                    "interactive_control_names": tuple(
                        element.name
                        for element in planning_request.current_screen_analysis.elements
                        if element_may_be_targeted(element)
                    ),
                }
                if planning_request.current_screen_analysis is not None
                else {}
            ),
        }
        response = self._model_client.complete(
            request,
            ActionReadinessDecision,
            validation_context=validation_context,
        )
        if response.disposition is not ModelCallDisposition.COMPLETED:
            return _failed_assessment(
                verification_id,
                proposal.proposal_id,
                response.reason or "The action-verifier model call did not complete.",
            )
        try:
            decision = ActionReadinessDecision.model_validate(
                response.structured_output, context=validation_context
            )
        except ValidationError:
            return _failed_assessment(
                verification_id,
                proposal.proposal_id,
                "The action-verifier response was invalid.",
            )
        known_work_item_set = set(known_work_items)
        reported_work_items = set(decision.completed_work_items)
        if decision.proposal_work_item is not None:
            reported_work_items.add(decision.proposal_work_item)
        if decision.restarts_completed_work_item is not None:
            reported_work_items.add(decision.restarts_completed_work_item)
        completed_work_item_set = set(completed_work_items)
        if (
            (
                decision.disposition is ActionReadinessDisposition.REFUSE
                and not (
                    decision.unresolved_controls
                    or decision.restarts_completed_work_item is not None
                )
            )
            or (
                decision.disposition is not ActionReadinessDisposition.REFUSE
                and decision.unresolved_controls
            )
            or decision.disposition is ActionReadinessDisposition.UNAVAILABLE
            or not reported_work_items.issubset(known_work_item_set)
            or (known_work_item_set and decision.proposal_work_item is None)
            or (
                decision.restarts_completed_work_item is not None
                and (
                    decision.disposition is not ActionReadinessDisposition.REFUSE
                    or decision.restarts_completed_work_item not in completed_work_item_set
                )
            )
        ):
            return _failed_assessment(
                verification_id,
                proposal.proposal_id,
                "The action-verifier response did not satisfy its evidence contract.",
            )
        return ActionReadinessAssessment(
            verification_id=verification_id,
            proposal_id=proposal.proposal_id,
            **decision.model_dump(),
        )


def _action_verifier_proposal_context(proposal: ActionProposal) -> dict[str, object]:
    """Project only the proposal fields used by the readiness decision."""
    return {
        key: value
        for key, value in {
            "action": proposal.action.value,
            "target_element_ref": proposal.target_element_ref,
            "target_expression": proposal.target_expression,
            "expected_outcome": proposal.expected_outcome,
            "reason": proposal.reason,
        }.items()
        if value is not None
    }


def _action_verifier_screen_context(
    analysis: CurrentScreenAnalysis,
) -> dict[str, object]:
    """Project interactive hypotheses without grounding or runtime bookkeeping."""
    return {
        "elements": tuple(
            {
                key: value
                for key, value in {
                    "element_ref": element.element_ref,
                    "name": element.name,
                    "kind": element.kind.semantic.value,
                    "affordance": element_affordance(element).value,
                    "state": element.state,
                    "confidence": element.confidence.value,
                    "workflow_role": (
                        element.workflow_role.value if element.workflow_role is not None else None
                    ),
                }.items()
                if value is not None
            }
            for element in analysis.elements
            if element_may_be_targeted(element)
        )
    }


def _required_control_evidence(
    request: PlanningRequest,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Bind exact trusted values to the analyzed controls that they identify."""
    application_feedback = tuple(
        entry.receipt.reason
        for entry in request.history
        if entry.receipt is not None and entry.receipt.reason is not None
    )
    evidence_by_basis = {
        RequiredControlBasis.OBJECTIVE_OR_OPEN_WORK.value: (
            request.objective,
            *request.open_work,
        ),
        RequiredControlBasis.TRUSTED_INSTRUCTION.value: request.known_facts,
        RequiredControlBasis.APPLICATION_REFUSAL.value: application_feedback,
        RequiredControlBasis.INPUT_CONTRACT.value: request.declared_value_names,
    }
    if request.current_screen_analysis is None:
        return {}
    return {
        element.name: {
            basis: tuple(
                evidence
                for evidence in evidence_values
                if evidence_identifies_control(element.name, evidence)
            )
            for basis, evidence_values in evidence_by_basis.items()
        }
        for element in request.current_screen_analysis.elements
        if element_may_be_targeted(element)
    }


def evidence_identifies_control(control_name: str, evidence: str) -> bool:
    """Match one visible identity without language-specific vocabulary."""
    control_tokens = _identity_tokens(control_name)
    evidence_tokens = _identity_tokens(evidence)
    width = len(control_tokens)
    token_match = bool(width) and any(
        evidence_tokens[index : index + width] == control_tokens
        for index in range(len(evidence_tokens) - width + 1)
    )
    return token_match or (
        any(not token.isascii() for token in control_tokens)
        and "".join(control_tokens) in "".join(evidence_tokens)
    )


def _identity_tokens(value: str) -> tuple[str, ...]:
    normalized_value = unicodedata.normalize("NFKC", value).casefold()
    separated = "".join(character if character.isalnum() else " " for character in normalized_value)
    return tuple(separated.split())


def _failed_assessment(
    verification_id: str,
    proposal_id: str,
    reason: str,
) -> ActionReadinessAssessment:
    return ActionReadinessAssessment(
        verification_id=verification_id,
        proposal_id=proposal_id,
        disposition=ActionReadinessDisposition.UNAVAILABLE,
        reason=reason,
        completed_work_items=(),
        proposal_work_item=None,
        restarts_completed_work_item=None,
    )
