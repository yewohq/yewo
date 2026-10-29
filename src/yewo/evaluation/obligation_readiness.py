"""Readiness measurement for strict obligation routing in locked journeys."""

from pathlib import Path

from pydantic import BaseModel

from yewo.testing.lock_file import Lock
from yewo.testing.obligations import (
    ObligationController,
    ObligationEvent,
    ObligationReadiness,
    ScenarioContract,
    aggregate_readiness,
)
from typing import cast


class LockedJourneyReadiness(BaseModel):
    scenario_count: int
    obligation_count: int
    measurement: ObligationReadiness

    @property
    def strict_ready(self) -> bool:
        return self.scenario_count > 0 and self.measurement.strict_ready


def measure_locked_journey(path: Path) -> LockedJourneyReadiness:
    """Measure every locked obligation without starting a device or model."""
    lock = Lock.model_validate_json(path.read_text(encoding="utf-8"))
    measurements = []
    obligation_count = 0
    for entry in lock.entries:
        scenario = entry.scenario()
        controller = ObligationController.from_scenario(
            cast(ScenarioContract, scenario)
        )
        obligation_count += len(controller.contracts)
        for event in ObligationEvent:
            controller.route(event)
        measurements.append(controller.readiness())
    return LockedJourneyReadiness(
        scenario_count=len(lock.entries),
        obligation_count=obligation_count,
        measurement=aggregate_readiness(measurements),
    )
