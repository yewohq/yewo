"""Typed contracts for the canonical screenshot-element catalog."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScreenSemanticRole(StrEnum):
    """Closed names of screenshot-visible element roles."""

    HEADING = "heading"
    LABEL = "label"
    TEXT = "text"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    SLIDER = "slider"
    DROPDOWN = "dropdown"
    DATE_PICKER = "date_picker"
    TIME_PICKER = "time_picker"
    FILE_PICKER = "file_picker"
    BUTTON = "button"
    LINK = "link"
    MENU_ITEM = "menu_item"
    OPTION = "option"
    CHIP = "chip"
    TAB = "tab"
    NAVIGATION_ITEM = "navigation_item"
    IMAGE = "image"
    ICON = "icon"
    LIST_ITEM = "list_item"
    CARD = "card"
    VIDEO = "video"
    PROGRESS_INDICATOR = "progress_indicator"
    BADGE = "badge"
    TOOLTIP = "tooltip"
    DIVIDER = "divider"
    CONTENT_REGION = "content_region"
    UNKNOWN = "unknown"


class ScreenVisualKind(StrEnum):
    """Closed names for the visible surface used to locate an element."""

    TEXT = "text"
    ICON = "icon"
    TEXT_AND_ICON = "text_and_icon"
    INPUT_SURFACE = "input_surface"
    CONTROL_SURFACE = "control_surface"
    IMAGE = "image"
    VIDEO = "video"
    CONTAINER = "container"
    INDICATOR = "indicator"
    DIVIDER = "divider"
    UNKNOWN = "unknown"


class ScreenElementAffordance(StrEnum):
    """Whether one element kind conventionally presents a direct affordance."""

    INTERACTIVE = "interactive"
    NON_INTERACTIVE = "non_interactive"
    CONTEXT_DEPENDENT = "context_dependent"


class ScreenWorkflowRole(StrEnum):
    """An observation-scoped hypothesis about one workflow control."""

    ADVANCE_STAGE = "advance_stage"
    CANCEL_STAGE = "cancel_stage"


class ScreenElementKind(BaseModel):
    """The visible locating surface and semantic role of one element."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    visual: ScreenVisualKind
    semantic: ScreenSemanticRole


class ScreenSemanticRoleDefinition(BaseModel):
    """Visible evidence and affordance semantics for one element kind."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: ScreenSemanticRole
    allowed_visual_kinds: frozenset[ScreenVisualKind] = Field(min_length=1)
    interaction_class: ScreenElementAffordance
    description: str = Field(min_length=1)
    visible_requirements: tuple[str, ...] = Field(min_length=1)


class ScreenVisualKindDefinition(BaseModel):
    """Canonical meaning of one screenshot-visible surface kind."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: ScreenVisualKind
    description: str = Field(min_length=1)
    visible_requirements: tuple[str, ...] = Field(min_length=1)
    requires_unique_surface: bool = False


class ScreenElementCatalog(BaseModel):
    """Canonical screenshot-grounded vocabulary and classification rules."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[2] = 2
    name: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    classification_rules: tuple[str, ...] = Field(min_length=1)
    overlap_rules: tuple[str, ...] = Field(min_length=1)
    interaction_classes: dict[ScreenElementAffordance, str]
    visual_kinds: tuple[ScreenVisualKindDefinition, ...] = Field(min_length=1)
    semantic_roles: tuple[ScreenSemanticRoleDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_closed_vocabulary(self) -> Self:
        semantic_roles = tuple(item.name for item in self.semantic_roles)
        if len(set(semantic_roles)) != len(semantic_roles):
            raise ValueError("Each screen semantic role must have one definition.")
        if set(semantic_roles) != set(ScreenSemanticRole):
            raise ValueError("Each screen semantic role must have one catalog definition.")
        visual_kinds = tuple(item.name for item in self.visual_kinds)
        if len(set(visual_kinds)) != len(visual_kinds):
            raise ValueError("Each screen visual kind must have one definition.")
        if set(visual_kinds) != set(ScreenVisualKind):
            raise ValueError("Each screen visual kind must have one catalog definition.")
        if set(self.interaction_classes) != set(ScreenElementAffordance):
            raise ValueError("The catalog must define every element affordance.")
        return self

    def role_definition(self, role: ScreenSemanticRole) -> ScreenSemanticRoleDefinition:
        """Return the canonical definition for one semantic role."""
        return next(item for item in self.semantic_roles if item.name is role)

    def visual_definition(self, visual: ScreenVisualKind) -> ScreenVisualKindDefinition:
        """Return the canonical definition for one visible surface kind."""
        return next(item for item in self.visual_kinds if item.name is visual)

    def validate_kind(self, kind: ScreenElementKind) -> None:
        """Reject a visual presentation that cannot represent its semantic role."""
        definition = self.role_definition(kind.semantic)
        if kind.visual not in definition.allowed_visual_kinds:
            raise ValueError(
                f"{kind.visual.value!r} is not an allowed visual kind for {kind.semantic.value!r}."
            )
