"""Shared typed records for cross-platform device interaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Self


class AndroidKey(IntEnum):
    """Supported Android hardware and navigation keys."""

    HOME = 3
    BACK = 4
    A = 29
    DPAD_UP = 19
    DPAD_DOWN = 20
    DPAD_LEFT = 21
    DPAD_RIGHT = 22
    DPAD_CENTER = 23
    TAB = 61
    ENTER = 66
    DELETE = 67
    ESCAPE = 111

    @classmethod
    def parse_name(cls, value: str) -> Self:
        """Resolve one case-insensitive key name through the canonical vocabulary."""
        try:
            return cls[value.upper()]
        except KeyError as error:
            raise ValueError("The Android key name is unsupported.") from error


@dataclass(frozen=True)
class TextInputConstraint:
    """A platform-established constraint on text accepted by the focused field."""

    is_numeric: bool
    allows_sign: bool = False
    allows_decimal: bool = False

    def accepts(self, value: str) -> bool:
        """Return whether text satisfies the known character-level constraint."""
        if not self.is_numeric:
            return True
        sign = "-?" if self.allows_sign else ""
        number = r"(?:\d+(?:[.,]\d*)?|[.,]\d+)" if self.allows_decimal else r"\d+"
        return re.fullmatch(sign + number, value) is not None

    def describe(self) -> str:
        """Describe the accepted value shape so a planner can satisfy it."""
        if not self.is_numeric:
            return "The focused field accepts text."
        allowed = "a decimal number" if self.allows_decimal else "whole digits only"
        refused = "" if self.allows_decimal else " A decimal point is not accepted."
        sign = (
            " A leading minus sign is accepted."
            if self.allows_sign
            else " A leading sign is not accepted."
        )
        return f"The focused field accepts {allowed}.{refused}{sign}"
