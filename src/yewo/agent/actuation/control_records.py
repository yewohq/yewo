"""Typed model observations of controls visible on an application screen."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VisibleControlKind(StrEnum):
    """Coarse UI role reported by the action verifier."""

    TEXT_FIELD = "text_field"
    TEXT = "text"
    BUTTON = "button"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    SLIDER = "slider"
    DROPDOWN = "dropdown"
    OPTION = "option"
    LINK = "link"
    TAB = "tab"
    ROW = "row"
    MENU_ITEM = "menu_item"
    ICON = "icon"
    OTHER = "other"


class VisibleControl(BaseModel):
    """One visible control description. It is knowledge, not completion proof."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    kind: VisibleControlKind
    state: str | None = Field(default=None, min_length=1, max_length=200)
    options: tuple[str, ...] = Field(default=(), max_length=100)
    matches_proposal_target: bool = False
