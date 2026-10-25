"""Append-only measurement telemetry for a run."""

import json
import logging
from pathlib import Path

from yewo.testing.models import Measurement
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


class MeasurementLog:
    """Writes scrubbed samples without turning telemetry loss into run failure."""

    def __init__(
        self,
        telemetry_dir: Path,
        scrubber: SecretScrubber | None = None,
    ) -> None:
        self.telemetry_dir = telemetry_dir
        self._scrubber = scrubber

    @property
    def path(self) -> Path:
        return self.telemetry_dir / "measurements.jsonl"

    def append(self, measurement: Measurement) -> bool:
        """Append one sample and return whether it reached disk."""
        try:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            row = measurement.model_dump(mode="json", exclude_none=True)
            body = json.dumps(row, sort_keys=True)
            if self._scrubber is not None:
                body = self._scrubber.scrub(body)
            with self.path.open("a", encoding="utf-8") as log:
                log.write(body + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            logger.warning("Could not write measurement telemetry: %s", exc)
            return False
