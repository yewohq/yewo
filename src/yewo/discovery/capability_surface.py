"""Decide locally whether a reached screen can exercise a capability.

Reaching the screen that contains a capability is not evidence that the
capability can be exercised. A form reached with no editable control, or with
a control nothing can name again, cannot carry a scenario. These predicates
are deterministic and hold no app labels, routes, or coordinates, so one
policy serves every application.
"""

from __future__ import annotations

from dataclasses import dataclass

from yewo.discovery.control_semantics import (
    ControlInteraction,
    ControlSurface,
)


@dataclass(frozen=True)
class SurfaceShortfall:
    """Why an observed surface cannot exercise a capability yet."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class CapabilitySurfacePredicate:
    """The interaction surface one capability needs, as counts and not labels.

    Expressed as minimums because a screen may offer more than a capability
    needs. A journey declares what it must interact with. The application
    decides how much else is present.
    """

    predicate_id: str
    minimum_editable_controls: int = 0
    minimum_activatable_controls: int = 0
    requires_reacquirable_controls: bool = True

    def shortfalls(self, surface: ControlSurface) -> tuple[SurfaceShortfall, ...]:
        """Every reason this observed surface falls short, in a stable order."""
        activatable = tuple(
            control
            for control in surface.controls
            if control.interaction is ControlInteraction.ACTIVATE
        )
        found = []
        if len(surface.editable_controls) < self.minimum_editable_controls:
            found.append(
                SurfaceShortfall(
                    f"the screen offers {len(surface.editable_controls)} editable "
                    f"control(s) and the capability needs "
                    f"{self.minimum_editable_controls}"
                )
            )
        if len(activatable) < self.minimum_activatable_controls:
            found.append(
                SurfaceShortfall(
                    f"the screen offers {len(activatable)} control(s) to activate "
                    f"and the capability needs {self.minimum_activatable_controls}"
                )
            )
        if self.requires_reacquirable_controls and surface.unreacquirable_controls:
            found.append(
                SurfaceShortfall(
                    f"{len(surface.unreacquirable_controls)} control(s) carry no "
                    "name and no anchor, so a later scenario cannot find them"
                )
            )
        return tuple(found)

    def is_satisfied_by(self, surface: ControlSurface) -> bool:
        """Whether this observation may close its capability as reached."""
        return not self.shortfalls(surface)
