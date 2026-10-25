"""What one run carries from step to step.

Each group belongs to a policy that decides when a screen is going nowhere, and
each is keyed by screen, because none of it means anything on a different one.
"""

from dataclasses import dataclass, field
from typing import Optional

from yewo.agent.memory.history import AgentHistory
from yewo.agent.grounding import marks
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.memory.stall import Stall
from yewo.agent.run.record import StepLog
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.run.postconditions import (
    PendingEnteredValue,
    PostconditionVerification,
    ValueVerificationPolicy,
)


@dataclass
class RunContext:
    history: AgentHistory
    loop_detector: RollingLoopDetector
    prev_phash: Optional[str] = None
    last_external_nav: bool = False
    drift_warning: Optional[str] = None
    steps_executed: int = 0
    current_phash: Optional[str] = None
    prev_element_count: Optional[int] = None
    keyboard_open: Optional[bool] = None
    last_action_type: Optional[str] = None
    screen_height: Optional[int] = None
    pending_loop_nudge: Optional[str] = None
    stall: Stall = field(default_factory=Stall)
    """Steps in a row that never reached the device, and where."""

    keyboard_dismissed: set[str] = field(default_factory=set)
    """Screens whose keyboard was already pulled down once to look behind it."""

    steps: StepLog = field(default_factory=StepLog)
    """One record per step, built as the step goes, carried out on the result.

    Kept here for the same reason the receipts are. A step can end the run
    part way through. What it reached before that is what a reader of a
    failed scenario needs.

    The default log has no scrubber, which is what a test gets. A run builds
    one with its own.
    """

    capability_receipts: list = field(default_factory=list)
    """What the device reported about each capability a step applied.

    Kept because a scenario can declare a capability it needs and then never
    reach it. Without the receipts the gate cannot tell that from a scenario
    that applied one and had it confirmed, so both would pass.
    """

    recovered: set[str] = field(default_factory=set)
    """Screens already handed to a recovery planner."""

    recovery_looks: int = 0
    """Model calls recovery made, counted apart from the run's own steps.

    Adding them to steps_executed would mix two counts: the loop numbers its
    history by step, and a scenario that was rescued would appear to have run
    steps it never ran.
    """

    text_strikes: dict[str, list[marks.Box]] = field(default_factory=dict)
    """Per screen, elements that took no text once. One is not yet proof.

    Focus that did not move means the tap reached nothing focusable. A picture
    of a field does that, and so does a tap into the padding of a real one. A
    second failure at the same place separates them.
    """

    refused_marks: dict[str, list[marks.Box]] = field(default_factory=dict)
    """Per screen, where the elements that took no text sit.

    Held by place, not by number: see marks.is_same_element.
    """

    declared_values: dict[str, str] = field(default_factory=dict)
    value_verifications: dict[str, ValueVerificationPolicy] = field(default_factory=dict)
    value_names: ValueNames = field(default_factory=ValueNames)
    pending_entered_value: Optional[PendingEnteredValue] = None
    postcondition_verifications: list[PostconditionVerification] = field(default_factory=list)
    postcondition_blocked: bool = False
    postcondition_recovery_required: bool = False
    observation_blocked: bool = False
