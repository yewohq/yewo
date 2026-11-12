"""Observation-bound records for one screenshot-derived screen hypothesis."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from yewo.agent.actuation.records import MAXIMUM_ACTION_TARGET_EXPRESSION_LENGTH
from yewo.agent.screen_analysis.catalog import (
    ScreenElementAffordance,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
    ScreenWorkflowRole,
    screen_element_catalog,
)
from yewo.artifacts import RecordIdentifier, Sha256Digest
from yewo.grounding import NormalizedRegion, ScreenAnalysisGroundingCandidate
from yewo.observations import Observation, PresentationKey


# The grounder searches on these details, so both element models must request them.
VISUAL_CUES_GUIDANCE = (
    "Visible details that let a visual grounder distinguish this element from similar elements "
    "in the screenshot. Use distinctive shape, iconography, styling, nearby landmarks, or "
    "spatial relationships. Always give at least one detail. The name already carries the "
    "visible identity, so never repeat it here."
)

MAXIMUM_SCREEN_ELEMENT_NAME_LENGTH = 200
MAXIMUM_SCREEN_ELEMENT_STATE_LENGTH = 200
MAXIMUM_SCREEN_ELEMENT_VISUAL_CUE_LENGTH = 200
MAXIMUM_SCREEN_ROLE_ALTERNATIVES = 10


def _truncate_visual_cue(value: object) -> object:
    return value[:MAXIMUM_SCREEN_ELEMENT_VISUAL_CUE_LENGTH] if isinstance(value, str) else value


ScreenVisualCue = Annotated[
    str,
    BeforeValidator(_truncate_visual_cue),
    Field(min_length=1, max_length=MAXIMUM_SCREEN_ELEMENT_VISUAL_CUE_LENGTH),
]


class ScreenElementConfidence(StrEnum):
    """Confidence supported by visible screenshot evidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def screen_role_affordance(role: ScreenSemanticRole) -> ScreenElementAffordance:
    """Return the conventional affordance without claiming runtime actionability."""
    return screen_element_catalog().role_definition(role).interaction_class


def element_affordance(element: ScreenElement) -> ScreenElementAffordance:
    """Return the affordance that honors an analyzer hedge toward interaction."""
    affordance = screen_role_affordance(element.kind.semantic)
    alternative_affordances = {
        screen_role_affordance(alternative) for alternative in element.role_alternatives
    }
    if any(alternative is not affordance for alternative in alternative_affordances):
        return ScreenElementAffordance.CONTEXT_DEPENDENT
    return affordance


def element_may_be_targeted(element: ScreenElement) -> bool:
    """Allow an established control or one the analyzer could not rule out."""
    return element_affordance(element) is not ScreenElementAffordance.NON_INTERACTIVE


class ScreenElement(BaseModel):
    """One visible semantic element with an observation-scoped identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    element_ref: RecordIdentifier
    name: str = Field(
        min_length=1,
        max_length=MAXIMUM_SCREEN_ELEMENT_NAME_LENGTH,
        description=(
            "The element's stable identity, taken from its own visible label or caption. "
            "Never the value the element currently holds, and unchanged when that value "
            "changes. An input takes the identity of the label that names it."
        ),
    )
    kind: ScreenElementKind
    state: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAXIMUM_SCREEN_ELEMENT_STATE_LENGTH,
        description=(
            "The element's current visible value or condition, such as the text inside an "
            "input, a selected value, or checked, empty, expanded, or disabled. Leave unset "
            "only when the element shows no value or condition."
        ),
    )
    options: tuple[str, ...] = Field(default=(), max_length=100)
    confidence: ScreenElementConfidence
    role_alternatives: tuple[ScreenSemanticRole, ...] = Field(
        default=(), max_length=MAXIMUM_SCREEN_ROLE_ALTERNATIVES
    )
    workflow_role: ScreenWorkflowRole | None = Field(
        default=None,
        description=(
            "An observation-scoped hypothesis that this control advances or cancels the current "
            "workflow stage. This is not evidence that activating it succeeded."
        ),
    )
    input_surface_key: str | None = Field(default=None, min_length=1, max_length=100)
    parent_ref: RecordIdentifier | None = None
    position_in_parent: int | None = Field(default=None, ge=1)
    visual_cues: tuple[ScreenVisualCue, ...] = Field(
        default=(), max_length=20, description=VISUAL_CUES_GUIDANCE
    )
    region_hint: NormalizedRegion | None = Field(
        default=None,
        description=(
            "Bounds of this exact element on this screenshot. Coordinates are "
            "normalized from zero to one with the origin at the top-left. This is "
            "observation-scoped analyzer evidence. A high-confidence interactive region can "
            "resolve one action point. It is not durable control geometry."
        ),
    )

    @model_validator(mode="after")
    def _validate_alternatives(self) -> Self:
        screen_element_catalog().validate_kind(self.kind)
        if self.kind.semantic in self.role_alternatives:
            raise ValueError("A screen semantic role cannot also be an alternative.")
        if len(set(self.role_alternatives)) != len(self.role_alternatives):
            raise ValueError("A screen semantic-role alternative must be unique.")
        return self


class CurrentScreenAnalysis(BaseModel):
    """A model-authored screen hypothesis bound to one exact observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    analysis_id: RecordIdentifier
    observation_id: RecordIdentifier
    screenshot_sha256: Sha256Digest
    presentation: PresentationKey
    screen_summary: str = Field(min_length=1, max_length=120)
    elements: tuple[ScreenElement, ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def _validate_element_references(self) -> Self:
        references = tuple(element.element_ref for element in self.elements)
        if len(set(references)) != len(references):
            raise ValueError("Each current-screen element reference must be unique.")
        surface_keys = tuple(
            element.input_surface_key
            for element in self.elements
            if element.input_surface_key is not None
        )
        if len(set(surface_keys)) != len(surface_keys):
            raise ValueError("One visible surface cannot support multiple screen elements.")
        elements_by_ref = {element.element_ref: element for element in self.elements}
        for element in self.elements:
            if element.parent_ref is not None and element.parent_ref not in elements_by_ref:
                raise ValueError("Each parent reference must identify a current-screen element.")
        sibling_positions: set[tuple[RecordIdentifier, int]] = set()
        for element in self.elements:
            parent_ref = element.parent_ref
            if parent_ref is None:
                if element.position_in_parent is not None:
                    raise ValueError("An element position requires a parent reference.")
                continue
            if parent_ref == element.element_ref:
                raise ValueError("A screen element cannot be its own parent.")
            if element.position_in_parent is not None:
                position = (parent_ref, element.position_in_parent)
                if position in sibling_positions:
                    raise ValueError("Sibling positions must be unique within one parent.")
                sibling_positions.add(position)
            visited = {element.element_ref}
            ancestor_ref: RecordIdentifier | None = parent_ref
            while ancestor_ref is not None:
                if ancestor_ref in visited:
                    raise ValueError("Screen element parent references cannot form a cycle.")
                visited.add(ancestor_ref)
                ancestor_ref = elements_by_ref[ancestor_ref].parent_ref
        return self

    def matches(self, observation: Observation) -> bool:
        """Return whether this analysis describes the exact supplied observation."""
        return (
            self.observation_id == observation.observation_id
            and self.screenshot_sha256 == observation.screenshot.sha256
            and self.presentation == observation.presentation_key
        )


def screen_analysis_context(analysis: CurrentScreenAnalysis) -> dict[str, object]:
    """Project one compact screen map without copying analyzer evidence prose."""
    return {
        "analysis_id": analysis.analysis_id,
        "observation_id": analysis.observation_id,
        "screen_summary": analysis.screen_summary,
        "elements": tuple(
            {
                key: value
                for key, value in {
                    "element_ref": element.element_ref,
                    "name": element.name,
                    "kind": {
                        "visual": element.kind.visual.value,
                        "semantic": element.kind.semantic.value,
                    },
                    "affordance": element_affordance(element).value,
                    "state": element.state,
                    "confidence": element.confidence.value,
                    "role_alternatives": tuple(item.value for item in element.role_alternatives),
                    "workflow_role": (
                        element.workflow_role.value if element.workflow_role is not None else None
                    ),
                    "parent_ref": element.parent_ref,
                    "position_in_parent": element.position_in_parent,
                    "visual_cues": element.visual_cues,
                }.items()
                if value not in (None, (), [])
            }
            for element in analysis.elements
        ),
    }


def grounding_expression_for_element(
    element: ScreenElement,
    parent: ScreenElement | None = None,
) -> str:
    """Describe one analyzed element to the visual grounder without durable coordinates."""
    parts = [_visual_target_phrase(element)]
    if parent is not None:
        parent_role = parent.kind.semantic.value.replace("_", " ")
        parts.append(f"inside the {parent_role} {parent.name!r}")
    if element.position_in_parent is not None:
        parts.append(f"at position {element.position_in_parent} in that container")
    if element.state is not None:
        parts.append(f"with visible state {element.state!r}")
    if element.visual_cues:
        parts.append("identified by " + "; ".join(element.visual_cues))
    return ", ".join(parts)[:MAXIMUM_ACTION_TARGET_EXPRESSION_LENGTH]


def _visual_target_phrase(element: ScreenElement) -> str:
    role = element.kind.semantic.value.replace("_", " ")
    if element.kind.visual is ScreenVisualKind.TEXT:
        row_roles = {
            ScreenSemanticRole.NAVIGATION_ITEM: "navigation row",
            ScreenSemanticRole.MENU_ITEM: "menu row",
            ScreenSemanticRole.LIST_ITEM: "list row",
            ScreenSemanticRole.OPTION: "option row",
        }
        container = row_roles.get(element.kind.semantic)
        return (
            f"the visible text {element.name!r} in its {container}"
            if container is not None
            else f"the visible text {element.name!r} for the {role}"
        )
    if element.kind.visual is ScreenVisualKind.ICON:
        return f"the visible icon for the {role} {element.name!r}"
    if element.kind.visual is ScreenVisualKind.TEXT_AND_ICON:
        return f"the {role} {element.name!r} showing visible text and an icon"
    visual = element.kind.visual.value.replace("_", " ")
    return f"the visible {visual} for {element.name!r}"


def resolve_element_expression(
    analysis: CurrentScreenAnalysis | None,
    element_ref: str | None,
) -> str | None:
    """Resolve an observation-scoped interactive reference into a TARS description."""
    if element_ref is None:
        return None
    if analysis is None:
        raise ValueError("An element_ref requires current screen analysis.")
    element = next((item for item in analysis.elements if item.element_ref == element_ref), None)
    if element is None:
        raise ValueError("The element_ref is not present in current screen analysis.")
    if not element_may_be_targeted(element):
        raise ValueError("The element_ref does not identify an interactive element.")
    parent = next(
        (item for item in analysis.elements if item.element_ref == element.parent_ref),
        None,
    )
    return grounding_expression_for_element(element, parent)


def interactive_element_references(
    analysis: CurrentScreenAnalysis | None,
) -> tuple[RecordIdentifier, ...]:
    """Return references the analyzer established or could not rule out."""
    return tuple(
        element.element_ref
        for element in (analysis.elements if analysis is not None else ())
        if element_may_be_targeted(element)
    )


def screen_analysis_grounding_candidate(
    analysis: CurrentScreenAnalysis | None,
    observation: Observation,
    element_ref: str | None,
) -> ScreenAnalysisGroundingCandidate | None:
    """Return exact-observation analyzer evidence that can safely resolve one point."""
    if analysis is None or element_ref is None or not analysis.matches(observation):
        return None
    element = next((item for item in analysis.elements if item.element_ref == element_ref), None)
    if (
        element is None
        or element.region_hint is None
        or element.confidence is not ScreenElementConfidence.HIGH
        or element_affordance(element) is not ScreenElementAffordance.INTERACTIVE
    ):
        return None
    return ScreenAnalysisGroundingCandidate(
        analysis_id=analysis.analysis_id,
        element_ref=element.element_ref,
        screenshot_sha256=analysis.screenshot_sha256,
        region=element.region_hint,
    )
