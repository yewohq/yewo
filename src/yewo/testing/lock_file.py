"""The `yewo.lock` file — the test, committed, so two runs can be compared.

A cell key is not a test. `security.negative.0` can name a scenario about a
rejected password this week. Next week it can name one about a locked
account. Both are faithful to the same journey file. One passed and one
failed, and whether the app broke is unanswerable: the file is identical and
the test is not the same test.

So the yaml locks the question, and this file locks the test. It holds the
walk, the expected outcomes and the setup, keyed by `scenario_uuid`, and it
is reviewed as a commit. `yewo.yaml` is `pyproject.toml` and this is
`uv.lock`.

Two things it deliberately does not hold.

- **No resolved value.** An entry keeps `{{rejected_password}}`, not `abc`. A
  lock holding values would hold a credential, and would change with the
  environment rather than with the test.
- **Nothing derived.** `steps_instructions`, `required_capabilities`, and
  `required_metrics` are read from the steps or outcomes by a validator.
  Storing them would let a stored value disagree with the derivation that
  produced it.

It locks the test, not the outcome. The agent may still walk the same steps
differently, and the auditor still judges every run.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from yewo import __version__
from yewo.agent.device.launch import SetupMode
from yewo.errors import UsageError
from yewo.testing.models import (
    ExpectedOutcome,
    GeneratedTestCase,
    ScenarioDraft,
    UserJourney,
)
from yewo.testing.steps import Step
from yewo.testing.lifecycle import SetupOrigin
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)

SCHEMA = 1
"""The version of this file's shape. Read before anything else in it."""

LOCK_NAME = "yewo.lock"
"""What the file is called, beside the journey file it locks."""


def _extract_journey_contract(journey: UserJourney) -> dict:
    """What the journey asks, with no value in it.

    Names only. A value is excluded on purpose: a hash of a six-character
    password is recovered offline in seconds, and a lock is committed. The
    prose is scrubbed first, because a file may write a declared secret into
    its own description.
    """
    scrubber = SecretScrubber()
    scrubber.learn_secrets(journey.secrets)
    contract = {
        "journey": journey.id,
        "name": scrubber.scrub(journey.name),
        "description": scrubber.scrub(journey.description),
        "security_rules": [scrubber.scrub(rule) for rule in journey.security_rules],
        "values": sorted(journey.values),
        "refused": sorted(journey.refused),
        "actors": {
            name: sorted(actor.values)
            for name, actor in sorted(journey.actors.items())
        },
    }
    if journey.preconditions.app_state is not None or journey.preconditions.capabilities:
        contract["preconditions"] = scrubber.scrub(
            json.dumps(
                journey.preconditions.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return contract


def _digest(payload: dict) -> str:
    """One hash of one canonical spelling, so two callers never disagree."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def contract_sha256(journey: UserJourney, cell: CoverageCell) -> str:
    """What the file asked for at this cell, so a changed question reads as stale."""
    return _digest(
        {**_extract_journey_contract(journey), "cell": cell.key, "priority": cell.priority.value}
    )


def journey_sha256(journey: UserJourney) -> str:
    """The journey as a whole, for a run manifest to record what it was given."""
    return _digest(
        {
            **_extract_journey_contract(journey),
            "coverage": [cell.key for cell in journey.coverage],
        }
    )


class LockEntry(BaseModel):
    """One scenario, as the lock holds it."""

    scenario_uuid: str
    workflow: str = ""
    journey: str
    cell: str = Field(description="The cell key this scenario answers, as a label.")
    test_id: str
    category: Category
    scenario_type: ScenarioType
    priority: Priority
    contract_sha256: str
    scenario_name: str
    setup: SetupMode
    steps: List[Step]
    expected_outcomes: List[ExpectedOutcome]

    @classmethod
    def of(
        cls, scenario: GeneratedTestCase, journey: UserJourney, workflow: str, digest: str
    ) -> "LockEntry":
        """The entry for one scenario, taken before any value was substituted."""
        return cls(
            scenario_uuid=scenario.scenario_uuid,
            workflow=workflow,
            journey=journey.id,
            cell=scenario.cell,
            test_id=scenario.test_id,
            category=scenario.category,
            scenario_type=scenario.scenario_type,
            priority=scenario.priority,
            contract_sha256=digest,
            scenario_name=scenario.scenario_name,
            setup=scenario.setup,
            steps=list(scenario.steps),
            expected_outcomes=list(scenario.expected_outcomes),
        )

    def cell_of(self) -> CoverageCell:
        """The cell this entry answers, rebuilt from its own axes."""
        return CoverageCell(
            category=self.category,
            scenario_type=self.scenario_type,
            priority=self.priority,
            index=int(self.cell.rsplit(".", 1)[-1]),
        )

    def scenario(self) -> GeneratedTestCase:
        """The scenario as a run holds it, still carrying the declared names."""
        draft = ScenarioDraft.model_construct(
            scenario_name=self.scenario_name,
            steps=list(self.steps),
            expected_outcomes=list(self.expected_outcomes),
        )
        return GeneratedTestCase.of(
            self.cell_of(),
            draft,
            scenario_uuid=self.scenario_uuid,
            setup=self.setup,
            setup_origin=SetupOrigin.LOCKED,
        )


class Lock(BaseModel):
    """Every locked scenario in one file, so a partial lock is a bug and not a state."""

    schema_version: int = SCHEMA
    yewo_version: str = __version__
    entries: List[LockEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _order_is_the_files_own(self) -> "Lock":
        """Sorts the entries, so the file does not depend on run order.

        Two journeys in one file are locked by whichever ran last. Left in
        arrival order, each run would reorder the file and rewrite it, and
        every commit would carry a diff that changed nothing.
        """
        self.entries.sort(key=lambda entry: (entry.workflow, entry.journey, entry.cell))
        return self

    def entries_for(self, workflow: str, journey: str) -> dict[str, LockEntry]:
        """The entries for one journey of one workflow, by cell key."""
        return {
            entry.cell: entry
            for entry in self.entries
            if entry.journey == journey and entry.workflow == workflow
        }

    def sha256(self) -> str:
        """The lock as a whole, for a run manifest to record what it ran."""
        return hashlib.sha256(_serialised(self).encode("utf-8")).hexdigest()


def _serialised(lock: Lock) -> str:
    """The file's text. One spelling, so a hash and a write never disagree."""
    return json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=False) + "\n"


def path_beside(journey_file: Path) -> Path:
    """Where the lock for this journey file lives."""
    return journey_file.resolve().parent / LOCK_NAME


def read(path: Path) -> Optional[Lock]:
    """The lock at this path, or None where there is none.

    Raises:
        UsageError: If the file exists and cannot be read as a lock. A
            malformed lock is never treated as an absent one, because that
            would silently regenerate the tests it was keeping.
    """
    if not path.is_file():
        return None
    try:
        lock = Lock.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise UsageError(f"{path} is not a readable lock: {exc}") from None
    if lock.schema_version != SCHEMA:
        raise UsageError(
            f"{path} was written for lock schema {lock.schema_version}, and this "
            f"Yewo reads schema {SCHEMA}. Run 'yewo lock' to rewrite it."
        )
    return lock


def write(path: Path, lock: Lock) -> None:
    """Writes the lock whole, through a temporary file.

    Raises:
        ArtifactWriteFailed: If the file could not be written.
    """
    from yewo.errors import ArtifactWriteFailed

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(_serialised(lock), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ArtifactWriteFailed(f"Could not write {path}: {exc}") from exc
