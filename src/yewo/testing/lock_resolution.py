"""Resolving which scenarios come from the lock and which must be generated.

The data is in `lock_file.py`. This is the policy, kept apart so one module holds
the file's shape and the other holds what a run does about it.

Three states of the lock, and two commands.

- **No lock.** `yewo test` generates it, writes it and runs. `--locked` fails.
- **A fresh lock.** Both read it and run. Neither writes.
- **A stale lock.** `yewo test` regenerates those cells and runs, loudly.
  `--locked` fails.

That is `npm install` against `npm ci`. Local work never thinks about it. CI
asks for a hard failure rather than a silently different test.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from yewo.errors import UsageError
from yewo.testing.generator import (
    GeneratedSuite,
    UncoveredCell,
    with_journey_preconditions,
    with_values_filled,
)
from yewo.testing.lock_file import Lock, LockEntry, contract_sha256, read, write
from yewo.testing.models import GeneratedTestCase, UserJourney
from yewo.testing.generation_evidence import ScenarioGenerationEvidence
from yewo.testing.taxonomy import CoverageCell, sort_cells_by_priority

logger = logging.getLogger(__name__)

Generate = Callable[[UserJourney, list[CoverageCell]], GeneratedSuite]
"""Asks for one scenario per cell. The generator, or a stand-in for it."""


@dataclass
class LockResolution:
    """The suite a run will execute, and where each scenario came from."""

    scenarios: list[GeneratedTestCase] = field(default_factory=list)
    uncovered: list[UncoveredCell] = field(default_factory=list)
    reused_count: int = 0
    generated_count: int = 0
    lock_sha256: Optional[str] = None
    lock_written: bool = False

    @property
    def note(self) -> str:
        """One line a reader of the log can tell the three states apart by."""
        if not self.reused_count and not self.generated_count:
            return "no scenario was produced"
        parts = []
        if self.reused_count:
            parts.append(f"{self.reused_count} read from the lock")
        if self.generated_count:
            parts.append(f"{self.generated_count} generated")
        return ", ".join(parts)


def _ordered_coverage_cells(journey: UserJourney) -> list[CoverageCell]:
    return sort_cells_by_priority(list(journey.coverage))


def staleness(
    lock: Optional[Lock],
    journey: UserJourney,
    workflow: str,
    refresh: bool = False,
    generation_evidence: ScenarioGenerationEvidence | None = None,
) -> dict[str, str]:
    """Why each declared cell cannot be read from the lock, by cell key.

    A cell with no entry and a cell whose question changed are different
    problems, so they are reported as different sentences. A cell that is
    absent from this mapping is fresh.

    `refresh` reports every cell as needing generation, which is what
    `yewo lock --all` asks for.
    """
    locked_entries = lock.entries_for(workflow, journey.id) if lock is not None else {}
    problems = {}
    for cell in _ordered_coverage_cells(journey):
        entry = locked_entries.get(cell.key)
        if refresh:
            problems[cell.key] = "asked for again"
        elif entry is None:
            problems[cell.key] = "not in the lock"
        elif entry.contract_sha256 != contract_sha256(
            journey,
            cell,
            generation_evidence.evidence_sha256 if generation_evidence is not None else "",
        ):
            problems[cell.key] = "the journey or verified app evidence changed since it was locked"
    return problems


def require_lock(
    journey: UserJourney,
    path: Optional[Path],
    workflow: str = "",
    generation_evidence: ScenarioGenerationEvidence | None = None,
) -> None:
    """Refuses a missing or stale lock before anything costly happens.

    `--locked` is what CI passes, and a stale lock there has to fail in a
    second rather than after a device boots. Reading the file costs nothing,
    so it is read first.

    Raises:
        UsageError: If the lock does not hold every declared cell.
    """
    lock = read(path) if path is not None else None
    _refuse(
        path,
        lock,
        staleness(
            lock,
            journey,
            workflow,
            generation_evidence=generation_evidence,
        ),
    )


def suite_for(
    journey: UserJourney,
    generate: Generate,
    path: Optional[Path] = None,
    workflow: str = "",
    locked: bool = False,
    refresh: bool = False,
    generation_evidence: ScenarioGenerationEvidence | None = None,
) -> LockResolution:
    """The scenarios to run, read from the lock wherever the lock still holds.

    Args:
        journey: The journey whose declared cells are covered.
        generate: Asked for the cells the lock cannot answer.
        path: Where the lock is. None runs without one, which is what a
            journey file outside a project does.
        workflow: Which workflow's copy of this journey, so one file can lock
            the same journey against two apps.
        locked: Refuse rather than generate. CI passes this.
        refresh: Ask again for every declared cell, which is what
            `yewo lock --all` does.

    Returns:
        The resolution, in declared priority order.

    Raises:
        UsageError: If `locked` and the lock is absent, or does not hold every
            declared cell for this journey.
    """
    lock = read(path) if path is not None else None
    problems = staleness(
        lock,
        journey,
        workflow,
        refresh=refresh,
        generation_evidence=generation_evidence,
    )

    if locked:
        _refuse(path, lock, problems)

    locked_entries = lock.entries_for(workflow, journey.id) if lock is not None else {}
    missing = [cell for cell in _ordered_coverage_cells(journey) if cell.key in problems]
    produced = generate(journey, missing) if missing else GeneratedSuite()

    fresh = {scenario.cell: scenario for scenario in produced.locked}
    filled = {scenario.cell: scenario for scenario in produced.scenarios}

    resolution = LockResolution(uncovered=list(produced.uncovered))
    keeping: list[LockEntry] = []
    # By identity, not by cell. Two scenarios can carry one cell key, and
    # dropping the second as a duplicate would lose a scenario that was
    # generated and paid for.
    taken: set[str] = set()
    for cell in _ordered_coverage_cells(journey):
        if cell.key in filled:
            resolution.scenarios.append(filled[cell.key])
            resolution.generated_count += 1
            taken.add(filled[cell.key].scenario_uuid)
            keeping.append(
                LockEntry.of(
                    fresh[cell.key],
                    journey,
                    workflow,
                    contract_sha256(
                        journey,
                        cell,
                        generation_evidence.evidence_sha256
                        if generation_evidence is not None
                        else "",
                    ),
                    generation_evidence,
                )
            )
            continue
        entry = locked_entries.get(cell.key)
        if entry is None:
            continue
        if cell.key in problems:
            # The lock holds this cell and the lock is out of date for it.
            # Generation was asked for a replacement and produced none, so
            # the cell is uncovered. Running the old scenario would run a
            # test the file no longer describes, and count it as reused.
            # The generator already recorded why it produced nothing.
            #
            # The entry is kept rather than dropped. It carries the contract
            # it was written under, so the next run reads it as stale again
            # and refuses it again. Dropping it would delete a reviewed test
            # because a provider was down for one call.
            keeping.append(entry)
            logger.warning(
                "Cell %s is stale and was not regenerated. It does not run. "
                "Its entry stays in the lock, and is refused again next run.",
                cell.key,
            )
            continue
        resolution.scenarios.append(
            with_journey_preconditions(
                with_values_filled(entry.scenario(), journey.values), journey
            )
        )
        resolution.reused_count += 1
        keeping.append(entry)

    # A scenario for a cell this journey does not declare is still a scenario
    # that was paid for, so it runs. It is not locked: there is no declared
    # cell to compute a contract against, so nothing could tell it was stale.
    for scenario in produced.scenarios:
        if scenario.scenario_uuid in taken:
            continue
        logger.warning(
            "Scenario %s answers %s, which this journey does not declare. "
            "It runs, and it is not locked.",
            scenario.test_id,
            scenario.cell,
        )
        resolution.scenarios.append(scenario)
        resolution.generated_count += 1

    if path is not None:
        resolution.lock_sha256, resolution.lock_written = _persist(
            path, lock, journey, workflow, keeping
        )
    return resolution


def _refuse(path: Optional[Path], lock: Optional[Lock], problems: dict[str, str]) -> None:
    """Ends the run rather than generating, which is what `--locked` asks for.

    Raises:
        UsageError: Always, unless the lock holds every declared cell.
    """
    if path is None:
        raise UsageError("--locked needs a journey file on disk to find yewo.lock beside.")
    if lock is None:
        raise UsageError(f"No lock at {path}. Run 'yewo lock' and commit it.")
    if problems:
        listed = "\n".join(f"  {cell}: {why}" for cell, why in sorted(problems.items()))
        raise UsageError(
            f"{path} does not cover every declared cell:\n{listed}\n"
            f"Run 'yewo lock' and commit the result."
        )


def _persist(
    path: Path,
    lock: Optional[Lock],
    journey: UserJourney,
    workflow: str,
    keeping: list[LockEntry],
) -> tuple[Optional[str], bool]:
    """Writes the lock where this run changed what it should hold.

    Entries belonging to another journey or workflow are carried over
    untouched. Entries for cells this file no longer declares are dropped. A
    lock is whole or it is a bug, and an entry nothing asks for is neither.

    Decided by comparing the file that would be written with the one that is
    there. So a run that read the lock and generated nothing leaves it alone.
    A run that only dropped a cell still writes.
    """
    if not keeping:
        # Every cell failed. Writing now would delete the entries this run
        # could not replace, and a lock cannot be rebuilt from a report.
        return (lock.sha256() if lock is not None else None), False

    others = [
        entry
        for entry in (lock.entries if lock is not None else [])
        if not (entry.journey == journey.id and entry.workflow == workflow)
    ]
    updated = Lock(entries=[*others, *keeping])
    if lock is not None and updated.sha256() == lock.sha256():
        return lock.sha256(), False

    write(path, updated)
    logger.info("Wrote %s (%d entries).", path, len(updated.entries))
    return updated.sha256(), True
