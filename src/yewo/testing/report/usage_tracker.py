"""What every model call cost, and what it spent that on.

Totals the token counts, cache counts, latency and cost across a run, and keeps
one row per call carrying the whole response. Counts two kinds of call
separately: answers cut off at the token limit, and questions asked twice after
a schema mismatch.

Written three times over, because a run can stop at three different moments.
Each call is appended to a log as it happens. Each scenario leaves a rolled-up
file. The run ends with the report. The log costs one line per call. Rewriting
the roll-up per call would rewrite a megabyte to add four kilobytes.

A response quotes the values a journey declared secret, because the model
writes its reasoning in its own words. The scrubber masks them before anything
reaches disk.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from typing import Any, Dict

from yewo.agent.run.record import CallRow, StepRecord
from yewo.llm.usage import CallRecord
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)

TRUNCATED = "length"


@dataclass
class UsageTotals:
    """What a run spent, one call at a time.

    These field names go straight into the report. They match the names already
    written to telemetry on disk, so an older file stays readable.
    """

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    cost_unknown_calls: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_latency_ms: float = 0.0

    truncated_calls: int = 0
    """Answers cut off at the token limit. They read as malformed, not as wrong."""

    retried_calls: int = 0
    """Questions asked twice, because the first answer did not fit the schema."""

    components: Dict[str, Any] = field(default_factory=dict)

    def add(self, call: CallRecord) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += call.prompt_tokens
        self.total_completion_tokens += call.completion_tokens
        self.total_tokens += call.total_tokens
        self.total_cached_tokens += call.cached_tokens
        self.total_reasoning_tokens += call.reasoning_tokens
        self.total_latency_ms += call.latency_ms or 0.0

        if call.cost is None:
            self.cost_unknown_calls += 1
        else:
            self.total_estimated_cost_usd += call.cost

        if call.finish_reason == TRUNCATED:
            self.truncated_calls += 1
        if call.attempt > 1:
            self.retried_calls += 1


class LLMUsageTracker:
    """Manages and serializes LLM token and cost telemetry during testing."""

    def __init__(
        self, output_dir: Path = Path("assets"), scrubber: SecretScrubber | None = None
    ):
        self.output_dir = output_dir
        self._scrubber = scrubber
        self.totals = UsageTotals()
        self._step: Optional[StepRecord] = None
        self._outside: Dict[str, list[CallRow]] = {}

    @property
    def log_path(self) -> Path:
        """Where each call is appended as it happens."""
        return self.output_dir / "llm_calls.jsonl"

    def _append(self, component_name: str, call: CallRecord) -> None:
        """Keeps one call, before anything else can go wrong.

        A crash inside a scenario takes the roll-up for that scenario with it,
        and those are the calls around whatever caused the crash. Appending
        loses only the call in flight.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            row = self._scrubbed({"component": component_name} | asdict(call))
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            logger.warning("Could not log the call: %s", exc)

    def mark(self, step: Optional[StepRecord]) -> None:
        """Says which step the calls from now on belong to.

        Set by whoever runs the loop. A call is made several layers below it,
        by a planner that has no reason to know which step it is on. So the
        attribution is made here, rather than carried through the planner.
        """
        self._step = step

    def add_call(self, component_name: str, call: CallRecord) -> None:
        """Callback to record a single LLM call for a specific component."""
        self._append(component_name, call)
        row = CallRow.of(component_name, call)
        if self._step is not None:
            self._step.calls.append(row)
        else:
            # The auditor and the route judge answer once the loop is over,
            # so their calls belong to the scenario and to no step in it.
            self._outside.setdefault(component_name, []).append(row)
        self.totals.add(call)
        if call.finish_reason == TRUNCATED:
            logger.warning(
                "Answer truncated at the token limit (%s). It reads as "
                "malformed rather than wrong, so raise the limit before "
                "rewording the schema.",
                call.generation_id,
            )

        component = self.totals.components.setdefault(
            component_name, {"models": [], "calls": []}
        )
        if call.model not in component["models"]:
            component["models"].append(call.model)
        component["calls"].append(asdict(call) | {"cost_known": call.cost is not None})

    def _scrubbed(self, value):
        """Masks declared secrets anywhere in the records, before anything writes them."""
        if self._scrubber is None:
            return value
        if isinstance(value, str):
            return self._scrubber.scrub(value)
        if isinstance(value, dict):
            return {key: self._scrubbed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._scrubbed(item) for item in value]
        return value

    def _write(self, filepath: Path) -> Path:
        """Writes the totals, through a temporary file so a reader never sees half."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = filepath.with_suffix(filepath.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._scrubbed(asdict(self.totals)), indent=2), encoding="utf-8"
        )
        temporary.replace(filepath)
        return filepath

    def checkpoint(self) -> Optional[Path]:
        """Writes what has been spent so far, to one file that keeps its name.

        A run that dies between scenarios otherwise loses every call made by
        the ones that passed. Written beside the partial reports, which exist
        for the same reason.
        """
        try:
            return self._write(self.output_dir / "llm_usage_partial.json")
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            logger.warning("Could not checkpoint LLM usage: %s", exc)
            return None

    def calls_outside_steps(self, component: str) -> list[CallRow]:
        """What this component spent while no step was open.

        Without these, the calls recorded under a scenario would not add up
        to what the scenario cost. A reader would have to know which judges
        answer after the loop to notice.
        """
        return list(self._outside.get(component, []))

    def take_calls_outside_steps(self, components: tuple[str, ...]) -> list[CallRow]:
        """Remove and return scenario-level calls for the named components."""
        taken = []
        for component in components:
            taken.extend(self._outside.pop(component, []))
        return taken

    def summary(self) -> dict:
        """What was spent, without the per-component breakdown.

        The manifest carries each call under the step that made it, so the
        breakdown there would be the same facts counted twice.
        """
        counted = asdict(self.totals)
        counted.pop("components", None)
        return counted

    def save_report(self) -> Path:
        """Serializes usage records to a JSON file in the assets directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self._write(self.output_dir / f"llm_usage_{timestamp}.json")
        logger.info("LLM usage telemetry successfully saved to %s", filepath)
        return filepath
