"""What one run records, so a path can be learned from it afterwards.

The loop hands over each step and the screen it was decided on. Nothing
here reads a device, walks a trace or writes a file. It owns the successful
segments and the rules for keeping them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from yewo.agent.actuation.actions import Action
from yewo.agent.memory.replay_store import ReplayTrace, ScreenSignature, ValueNames
from yewo.agent.run.postconditions import ValueVerificationPolicy
from yewo.agent.perception.elements import GUIElement, ScanInfo


Screen = Tuple[Sequence[GUIElement], ScanInfo]
"""A reading reduced to what matching needs: the elements and the scan."""


class ScreenReading(Protocol):
    """One reading of the screen, as the loop hands it over."""

    @property
    def gui_elements(self) -> Sequence[GUIElement]: ...

    @property
    def scan_info(self) -> Optional[ScanInfo]: ...

    @property
    def current_phash(self) -> Optional[str]: ...

    @property
    def raw_img_bytes(self) -> bytes: ...

    @property
    def observation_id(self) -> str: ...


@dataclass
class DurableReplay:
    """One scenario's saved path, and everything needed to walk it.

    Built by the caller that knows the scenario, because the run loop knows
    a goal and a device and nothing about locks or journeys. Handed down so
    the loop only has to ask whether there is a path and walk it.
    """

    trace: Optional[ReplayTrace]
    values: Dict[str, str]
    names: ValueNames
    verification_policies: Dict[str, ValueVerificationPolicy] = field(default_factory=dict)
    key: Optional[bytes] = None

    recorded_steps: List[Tuple[Optional[Action], ScreenReading]] = field(default_factory=list)
    """Every step this run took, with the screen it was decided on.

    Cached steps and planned ones share one list. A failed action closes a
    segment with a no-action entry. Later successful actions start another
    segment instead of being discarded.

    The last entry carries no action. It is the reading taken after the
    final step, which is that step's destination and nothing else's source.
    """

    action_arrival_ms: List[int] = field(default_factory=list)
    """Action dispatch through immediate destination settlement, per step."""

    selected_arrival_ms: List[int] = field(default_factory=list)
    """Arrival durations aligned with the transitions selected for persistence."""

    position: int = 0
    """Which saved step to try next.

    A cursor rather than a loop counter. The run plans a step the saved
    path could not supply, and the search for the next one starts here and
    moves forward. It never moves back: a path that revisits a screen is
    the loop the detector exists to catch.
    """

    pending_destination: Optional["ScreenSignature"] = None
    """Where the last cached step was supposed to arrive.

    Checked against the next step's own reading rather than a second one
    taken on the spot. A step's destination is the next step's source, so
    reading it twice costs a detection pass and answers the same question.
    """

    cached_step_count: int = 0
    """How many steps this run walked from the file rather than planned.

    Read by whoever numbers what the auditor found, because a planned step
    counts itself from one.
    """

    path_completed: bool = False
    """Whether every saved action reached its expected destination."""

    target_descriptions: List[Optional[str]] = field(default_factory=list)
    """The semantic target for each recorded action, aligned with the records."""

    screen_descriptions: List[Optional[str]] = field(default_factory=list)
    """The secret-safe screen name for each recorded reading."""

    recorded_segment_starts: List[bool] = field(default_factory=list)
    """Whether each recorded action starts after a failed-action gap."""

    next_step_starts_segment: bool = False
    """Whether the next successful action starts after a failed action."""

    selected_target_descriptions: List[Optional[str]] = field(default_factory=list)
    selected_source_descriptions: List[Optional[str]] = field(default_factory=list)
    selected_destination_descriptions: List[Optional[str]] = field(default_factory=list)
    selected_segment_starts: List[bool] = field(default_factory=list)
    """Metadata aligned with the transitions selected for persistence."""

    pending_arrival_description: Optional[str] = None
    """The planner's name for a terminal screen awaiting its final reading."""

    current_artifact: Optional[str] = None
    """The build this run is executing, for the strict local fast path."""

    @property
    def is_worth_saving(self) -> bool:
        """Whether this run's path should replace what is on disk.

        A refusal part way through is not a reason to keep the old file.
        The run can retain successful segments before and after the refused
        action. Refusing to write them would leave a path on disk that no
        longer fits.
        """
        return any(action is not None for action, _ in self.recorded_steps)

    @property
    def has_path(self) -> bool:
        return self.trace is not None and bool(self.trace.transitions)

    @property
    def is_same_build(self) -> bool:
        """Whether the saved target geometry belongs to this exact artifact."""
        return bool(
            self.trace is not None
            and self.trace.learned_from
            and self.current_artifact
            and self.trace.learned_from == self.current_artifact
        )

    def record_step(
        self,
        action: Action,
        screen: ScreenReading,
        arrival_ms: int = 0,
        target_description: Optional[str] = None,
        screen_description: Optional[str] = None,
    ) -> None:
        """Keep one action, its source screen, and observed settlement time."""
        self.recorded_steps.append((action, screen))
        self.target_descriptions.append(target_description)
        self.screen_descriptions.append(screen_description)
        self.recorded_segment_starts.append(self.next_step_starts_segment)
        self.next_step_starts_segment = False
        self.action_arrival_ms.append(max(0, int(arrival_ms)))

    def close_learning_segment(
        self, screen: ScreenReading, description: Optional[str] = None
    ) -> None:
        """Close the current segment before a failed action.

        The failed action is not recorded. A later successful action starts a
        new segment from the screen observed after recovery.
        """
        self.record_arrival(screen, description)
        self.next_step_starts_segment = True

    def reject_last_step(self) -> None:
        """Remove an action whose post-settlement verification failed."""
        if not self.recorded_steps or self.recorded_steps[-1][0] is None:
            self.next_step_starts_segment = True
            return
        _action, source = self.recorded_steps.pop()
        description = self.screen_descriptions.pop()
        self.target_descriptions.pop()
        self.recorded_segment_starts.pop()
        self.action_arrival_ms.pop()
        self.close_learning_segment(source, description)

    def describe_arrival(self, description: Optional[str]) -> None:
        """Keeps the name of a terminal screen until its final reading."""
        if self.recorded_steps and self.recorded_steps[-1][0] is not None:
            self.pending_arrival_description = description

    def record_arrival(self, screen: ScreenReading, description: Optional[str] = None) -> None:
        """Keeps the screen the last action produced.

        Nothing else reads it. A step's destination is the next step's
        source, so the final step has none until the run takes one more
        reading. Without this the last action of every scenario is dropped,
        and a scenario of one action learns nothing at all.
        """
        if not self.is_worth_saving:
            return
        if self.recorded_steps[-1][0] is None:
            return
        self.recorded_steps.append((None, screen))
        self.target_descriptions.append(None)
        self.screen_descriptions.append(description or self.pending_arrival_description)
        self.recorded_segment_starts.append(False)
        self.pending_arrival_description = None
        self.action_arrival_ms.append(0)

    def select_learned_steps(self) -> Tuple[List[Action], List[Tuple[Screen, Screen]]]:
        """The steps and their before-and-after screens, ready to learn from.

        One pair for every successful step whose destination was read. A
        failed action leaves a segment boundary instead of discarding later
        successful transitions.
        """
        actions = [action for action, _ in self.recorded_steps]
        screens = [screen for _, screen in self.recorded_steps]
        self.selected_arrival_ms = []
        self.selected_target_descriptions = []
        self.selected_source_descriptions = []
        self.selected_destination_descriptions = []
        self.selected_segment_starts = []
        steps: List[Action] = []
        screen_pairs: List[Tuple[Screen, Screen]] = []
        for position, (source, destination) in enumerate(zip(screens, screens[1:], strict=False)):
            action = actions[position]
            if action is None or source.scan_info is None or destination.scan_info is None:
                continue
            screen_pairs.append(
                (
                    (source.gui_elements, source.scan_info),
                    (destination.gui_elements, destination.scan_info),
                )
            )
            steps.append(action)
            self.selected_arrival_ms.append(self.action_arrival_ms[position])
            self.selected_target_descriptions.append(self.target_descriptions[position])
            self.selected_source_descriptions.append(self.screen_descriptions[position])
            self.selected_destination_descriptions.append(self.screen_descriptions[position + 1])
            self.selected_segment_starts.append(self.recorded_segment_starts[position])
        return steps, screen_pairs
