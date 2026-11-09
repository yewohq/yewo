"""Model-backed semantic analysis of one redacted screenshot."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.agent.screen_analysis.catalog import (
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenWorkflowRole,
    screen_element_catalog,
)
from yewo.agent.screen_analysis.prompt import (
    SCREEN_ANALYZER_USER_PROMPT,
    screen_analyzer_system_prompt,
)
from yewo.agent.screen_analysis.records import (
    MAXIMUM_SCREEN_ELEMENT_NAME_LENGTH,
    MAXIMUM_SCREEN_ELEMENT_STATE_LENGTH,
    ScreenVisualCue,
    VISUAL_CUES_GUIDANCE,
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
)
from yewo.artifacts import new_record_identifier
from yewo.grounding import NormalizedRegion
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
from yewo.observations import Observation


class ScreenAnalysisError(RuntimeError):
    """A screen-analysis request did not yield a usable hypothesis."""


class ScreenAnalysisCapacityError(ScreenAnalysisError):
    """The selected profile exhausted capacity before producing its contract."""

    def __init__(self, profile_id: str, response_contract_id: str) -> None:
        super().__init__("Screen analysis exhausted its reasoning capacity.")
        self.profile_id = profile_id
        self.response_contract_id = response_contract_id


SCREEN_ANALYSIS_RESPONSE_CONTRACT_ID = "screen-analysis-hypothesis-v1"


_PARENT_SCOPED_ROLES = frozenset(
    {
        ScreenSemanticRole.NAVIGATION_ITEM,
        ScreenSemanticRole.MENU_ITEM,
        ScreenSemanticRole.OPTION,
    }
)


class ScreenElementHypothesis(BaseModel):
    """Model-authored element data before runtime identity is assigned."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    analysis_key: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "An identity unique within this response. It does not establish identity across "
            "screenshots."
        ),
    )
    name: str = Field(min_length=1, max_length=MAXIMUM_SCREEN_ELEMENT_NAME_LENGTH)
    kind: ScreenElementKind
    state: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAXIMUM_SCREEN_ELEMENT_STATE_LENGTH,
    )
    options: tuple[str, ...] = Field(default=(), max_length=100)
    confidence: ScreenElementConfidence
    role_alternatives: tuple[ScreenSemanticRole, ...] = Field(
        default=(),
        max_length=10,
        description=(
            "Other semantic roles the screenshot visibly supports for this element. Each one must "
            "have its own visible evidence, and none may repeat the role in kind.semantic. Leave "
            "this empty when the visible evidence establishes the chosen role: an alternative "
            "whose interaction nature differs from that role marks the element "
            "interaction-uncertain for everything downstream."
        ),
    )
    workflow_role: ScreenWorkflowRole | None = Field(
        default=None,
        description=(
            "Set advance_stage only when visible evidence indicates that activating this exact "
            "control advances, submits, saves, accepts, or completes the current workflow stage. "
            "Set cancel_stage only when visible evidence indicates abandonment or cancellation. "
            "Use any visible language or iconography. Leave this unset when the role is uncertain."
        ),
    )
    input_surface_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "A response-local identity for one visual input_surface. Set this only when "
            "kind.visual is input_surface. It does not establish identity across screenshots. "
            "Leave it unset for every other visual kind."
        ),
    )
    parent_key: str | None = Field(default=None, min_length=1, max_length=100)
    position_in_parent: int | None = Field(default=None, ge=1)
    visual_cues: tuple[ScreenVisualCue, ...] = Field(
        default=(), max_length=20, description=VISUAL_CUES_GUIDANCE
    )
    region_hint: NormalizedRegion = Field(
        description=(
            "Bounds of this exact visible element on the supplied screenshot. "
            "Use normalized zero-to-one coordinates with the origin at the top-left. Exclude "
            "unrelated neighboring elements. For an interactive element, place the region "
            "center inside the actionable surface. This is observation-bound location evidence."
        ),
    )

    @model_validator(mode="after")
    def _normalize_optional_model_bookkeeping(self) -> Self:
        """Keep safe visible evidence when optional bookkeeping is malformed."""
        updates: dict[str, object] = {}
        requires_input_key = (
            screen_element_catalog().visual_definition(self.kind.visual).requires_unique_surface
        )
        if requires_input_key and self.input_surface_key is None:
            updates["input_surface_key"] = self.analysis_key
        elif not requires_input_key and self.input_surface_key is not None:
            updates["input_surface_key"] = None
        if (
            self.confidence is not ScreenElementConfidence.HIGH
            and self.kind.semantic is not ScreenSemanticRole.UNKNOWN
            and not self.role_alternatives
        ):
            claimed_role = self.kind.semantic
            updates["kind"] = ScreenElementKind(
                visual=self.kind.visual,
                semantic=ScreenSemanticRole.UNKNOWN,
            )
            updates["role_alternatives"] = (claimed_role,)
        return self.model_copy(update=updates) if updates else self


class ScreenAnalysisHypothesis(BaseModel):
    """Closed structured output requested from the analyzer model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    screen_summary: str = Field(min_length=1, max_length=120)
    elements: tuple[ScreenElementHypothesis, ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def _validate_element_relationships(self) -> Self:
        keys = tuple(element.analysis_key for element in self.elements)
        if len(set(keys)) != len(keys):
            raise ValueError("Each analysis key must be unique.")
        elements_by_key = {element.analysis_key: element for element in self.elements}
        claimed_surface_keys: set[str] = set()
        sibling_positions: set[tuple[str, int]] = set()
        normalized_elements: list[ScreenElementHypothesis] = []
        for element in self.elements:
            visual_definition = screen_element_catalog().visual_definition(element.kind.visual)
            if visual_definition.requires_unique_surface:
                if element.input_surface_key is None:
                    raise ValueError("An input surface requires an input surface key.")
                if element.input_surface_key in claimed_surface_keys:
                    raise ValueError("One visible input surface cannot support multiple elements.")
                claimed_surface_keys.add(element.input_surface_key)
            if element.parent_key is None or element.parent_key not in elements_by_key:
                updates: dict[str, object] = {
                    "parent_key": None,
                    "position_in_parent": None,
                }
                if element.kind.semantic in _PARENT_SCOPED_ROLES:
                    claimed_role = element.kind.semantic
                    updates["kind"] = ScreenElementKind(
                        visual=element.kind.visual,
                        semantic=ScreenSemanticRole.UNKNOWN,
                    )
                    updates["role_alternatives"] = tuple(
                        dict.fromkeys((*element.role_alternatives, claimed_role))
                    )
                element = element.model_copy(update=updates)
                normalized_elements.append(element)
                continue
            if element.parent_key == element.analysis_key:
                raise ValueError("An element cannot be its own parent.")
            if element.position_in_parent is not None:
                sibling_position = (element.parent_key, element.position_in_parent)
                if sibling_position in sibling_positions:
                    element = element.model_copy(update={"position_in_parent": None})
                else:
                    sibling_positions.add(sibling_position)
            normalized_elements.append(element)
        normalized_by_key = {element.analysis_key: element for element in normalized_elements}
        for element in normalized_elements:
            visited = {element.analysis_key}
            parent_key = element.parent_key
            while parent_key is not None:
                if parent_key in visited:
                    raise ValueError("Element parent keys cannot form a cycle.")
                visited.add(parent_key)
                parent_key = normalized_by_key[parent_key].parent_key
        normalized_tuple = tuple(normalized_elements)
        if normalized_tuple != self.elements:
            return self.model_copy(update={"elements": normalized_tuple})
        return self


class ScreenAnalysisSource(Protocol):
    """Analyze a redacted observation without choosing an action."""

    def required_model_requests(self, observation: Observation) -> int: ...

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis: ...


class ModelScreenAnalyzer:
    """Ask one configured visual model for a semantic screen hypothesis."""

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

    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def capacity_identity(self) -> tuple[str, str]:
        """Return the profile and validated response contract used by this source."""
        return self._model_profile_id, SCREEN_ANALYSIS_RESPONSE_CONTRACT_ID

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        request = ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=self._model_profile_id,
            role=ModelRole.SCREEN_ANALYZER,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text=screen_analyzer_system_prompt()),),
                    cache_prefix_end=True,
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(
                        TextModelContent(text=SCREEN_ANALYZER_USER_PROMPT),
                        TextModelContent(
                            text="Observation context:\n"
                            + json.dumps(
                                {"platform": observation.device.platform.value},
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        ImageModelContent(screenshot=observation.redacted_screenshot),
                    ),
                ),
            ),
            response_contract=build_model_response_contract(
                SCREEN_ANALYSIS_RESPONSE_CONTRACT_ID,
                ScreenAnalysisHypothesis.model_json_schema(),
            ),
            sensitivity=observation.redacted_screenshot.sensitivity,
            retention=observation.redacted_screenshot.retention,
        )
        response = self._model_client.complete(request, ScreenAnalysisHypothesis)
        if response.disposition is ModelCallDisposition.REASONING_EXHAUSTED:
            raise ScreenAnalysisCapacityError(*self.capacity_identity())
        if response.disposition is not ModelCallDisposition.COMPLETED:
            raise ScreenAnalysisError(response.reason or "Screen analysis did not complete.")
        if response.structured_output is None:
            raise ScreenAnalysisError("Screen analysis returned no structured output.")
        try:
            hypothesis = ScreenAnalysisHypothesis.model_validate(response.structured_output)
            element_refs = {
                element.analysis_key: f"element-{index}"
                for index, element in enumerate(hypothesis.elements, start=1)
            }
            elements = tuple(
                ScreenElement(
                    element_ref=element_refs[element.analysis_key],
                    name=element.name[:MAXIMUM_SCREEN_ELEMENT_NAME_LENGTH],
                    state=(
                        element.state[:MAXIMUM_SCREEN_ELEMENT_STATE_LENGTH]
                        if element.state is not None
                        else None
                    ),
                    evidence=_stored_element_trace(element),
                    parent_ref=(
                        element_refs[element.parent_key] if element.parent_key is not None else None
                    ),
                    **element.model_dump(exclude={"analysis_key", "name", "state", "parent_key"}),
                )
                for element in hypothesis.elements
            )
            analysis = CurrentScreenAnalysis(
                analysis_id=self._identifier_factory(),
                observation_id=observation.observation_id,
                screenshot_sha256=observation.redacted_screenshot.sha256,
                presentation=observation.presentation_key,
                screen_summary=hypothesis.screen_summary,
                elements=elements,
            )
        except ValidationError as error:
            raise ScreenAnalysisError("Screen analysis returned an invalid hypothesis.") from error
        return analysis


def _stored_element_trace(element: ScreenElementHypothesis) -> str:
    """Retain a compact record trace without asking the model for duplicate prose."""
    visual = element.kind.visual.value.replace("_", " ")
    return f"Visible {visual} {element.name!r}."
