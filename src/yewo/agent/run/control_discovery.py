"""Typed state for deterministic recovery of a visually missing control."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


Box = tuple[int, int, int, int]
MAX_PROBE_ATTEMPTS = 10


class ControlDiscoveryState(str, Enum):
    REQUESTED = "requested"
    PROBING = "probing"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass
class ControlDiscoveryTransaction:
    """One bounded attempt to recover one target absent from perception."""

    source_phash: str
    region: str
    description: str
    intent: str
    anchor_above: Optional[Box] = None
    anchor_below: Optional[Box] = None
    state: ControlDiscoveryState = ControlDiscoveryState.REQUESTED
    probe_budget: int = MAX_PROBE_ATTEMPTS
    probed_cell_indexes: set[int] = field(default_factory=set)
    verified_bounds: Optional[Box] = None
    failure_reason: Optional[str] = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source_phash, self.description, self.region

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            ControlDiscoveryState.RESOLVED,
            ControlDiscoveryState.FAILED,
        )

    @property
    def probes_remaining(self) -> int:
        return max(0, self.probe_budget - len(self.probed_cell_indexes))

    def begin_probing(self) -> None:
        if self.state not in (
            ControlDiscoveryState.REQUESTED,
            ControlDiscoveryState.PROBING,
        ):
            raise ValueError(f"Cannot probe a control discovery in state {self.state.value}.")
        self.state = ControlDiscoveryState.PROBING

    def record_probe(self, index: int, bounds: Optional[Box]) -> None:
        if self.state is not ControlDiscoveryState.PROBING:
            raise ValueError("A control probe requires the probing state.")
        if index in self.probed_cell_indexes:
            raise ValueError("A control-discovery cell cannot be probed twice.")
        if self.probes_remaining == 0:
            raise ValueError("The control-discovery probe budget is exhausted.")
        self.probed_cell_indexes.add(index)
        if bounds is not None:
            self.verified_bounds = bounds
            self.state = ControlDiscoveryState.VERIFYING

    def resolve(self, bounds: Box) -> None:
        if self.state is not ControlDiscoveryState.VERIFYING:
            raise ValueError("Only a verifying control discovery can resolve.")
        self.verified_bounds = bounds
        self.state = ControlDiscoveryState.RESOLVED

    def fail(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError("A terminal control discovery is immutable.")
        self.failure_reason = reason
        self.state = ControlDiscoveryState.FAILED
