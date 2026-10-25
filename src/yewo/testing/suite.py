"""Reading a yewo.yaml into the workflows it describes.

A workflow owns an app, a device and an environment. The journeys inside it
are only the tests, so one file can cover several apps without a journey
having to say where it runs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from yewo.config import SETTING_ALIASES, STRUCTURE_KEYS, ConfigError, expand_device
from yewo.testing.journey_env import (
    resolve,
    secret_displays,
    substitute,
    verification_policies,
)
from yewo.testing.models import Actor, UserJourney
from yewo.agent.run.postconditions import ValueVerificationPolicy
from yewo.testing.taxonomy import DEFAULT_COVERAGE, CoverageCell, index_coverage_cells
from yewo.schema import CoverageSpec, YewoFile, format_validation_errors

logger = logging.getLogger(__name__)

PATH_SETTINGS = ("app_path", "output_dir")
"""Settings read as paths, resolved against the file that named them."""


@dataclass(frozen=True)
class Workflow:
    """One app, one device, and the journeys to run against them."""

    name: str
    settings: dict[str, Any] = field(default_factory=dict)
    journeys: list[UserJourney] = field(default_factory=list)


def _normalise(block: dict[str, Any], yaml_location: str) -> dict[str, Any]:
    """Settings from one level of the file, under the names Yewo resolves."""
    expanded = expand_device(block, yaml_location)
    settings = {}
    for key, value in expanded.items():
        if key in STRUCTURE_KEYS:
            continue
        settings[SETTING_ALIASES.get(key, key)] = value
    return settings


def _anchor(settings: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolves path settings against the directory holding the file.

    A workflow names its app beside itself, so `yewo test` finds the same
    binary wherever it is run from.
    """
    anchored = dict(settings)
    for name in PATH_SETTINGS:
        raw = anchored.get(name)
        if raw and not Path(str(raw)).is_absolute():
            anchored[name] = str((base / str(raw)).resolve())
    return anchored


def _cells(coverage_data: Any, yaml_location: str) -> list[CoverageCell]:
    """The cells one `coverage:` block asks for, one per category and type.

    Returns the default set where nothing is declared, so a file gets a
    useful suite without having to describe one.

    Raises:
        ConfigError: If the block is not a list, is empty, or is not valid.
    """
    if coverage_data is None:
        return list(DEFAULT_COVERAGE)
    if not isinstance(coverage_data, list):
        raise ConfigError(f"'coverage' in {yaml_location} must be a list of categories.")
    # An empty list is refused rather than read as "cover nothing". A run with
    # no cells writes a report of zero scenarios, and zero failures reads to
    # CI as a pass.
    if not coverage_data:
        raise ConfigError(
            f"'coverage' in {yaml_location} is empty. Remove the key to use the "
            f"default cells, or name the categories to cover."
        )
    pairs = []
    for item in coverage_data:
        try:
            spec = CoverageSpec.model_validate(item)
        except ValidationError as exc:
            problems = "; ".join(format_validation_errors(exc))
            raise ConfigError(f"'coverage' in {yaml_location} is not valid: {problems}") from None
        pairs.extend(
            (spec.category, wanted, spec.priority) for wanted in spec.types
        )
    # Indexed here rather than defaulted on the cell: a file may name the
    # same pair twice, and two cells sharing a key would share a test id.
    return index_coverage_cells(pairs)


def _actors(
    actors_data: Any, yaml_location: str
) -> tuple[
    dict[str, Actor],
    dict[str, str],
    dict[str, str],
    dict[str, ValueVerificationPolicy],
]:
    """Each actor, its values, and the policies attached to those values.

    An actor holds the app-specific values that establish its context. A
    secret is masked under the path the file wrote. A name Yewo invented would
    appear in a report that nobody could trace back to the file.

    Raises:
        ConfigError: If the block, or an actor inside it, is not valid.
    """
    if actors_data is None:
        return {}, {}, {}, {}
    if not isinstance(actors_data, dict):
        raise ConfigError(f"'actors' in {yaml_location} must be a mapping of name to actor.")
    actors: dict[str, Actor] = {}
    secrets: dict[str, str] = {}
    displays: dict[str, str] = {}
    verifications: dict[str, ValueVerificationPolicy] = {}
    for name, block in actors_data.items():
        if not isinstance(block, dict) or not block:
            raise ConfigError(
                f"'actors.{name}' in {yaml_location} must be a mapping of name to value."
            )
        try:
            values, secret_values, _ = resolve({"env": block})
        except ValueError as problem:
            raise ConfigError(
                f"'actors.{name}' in {yaml_location} is not valid: {problem}"
            ) from None
        actors[str(name)] = Actor(values=values)
        secrets.update(
            {
                f"actors.{name}.{key}": value
                for key, value in secret_values.items()
            }
        )
        displays.update(
            {
                f"actors.{name}.{key}": value
                for key, value in secret_displays({"env": block}).items()
            }
        )
        verifications.update(
            {
                f"actors.{name}.{key}": value
                for key, value in verification_policies({"env": block}).items()
            }
        )
    return actors, secrets, displays, verifications


def _merge_actor_values(values: dict[str, str], actors: dict[str, Actor]) -> dict[str, str]:
    """One namespace of every value a scenario may name.

    An actor's values join it under a dotted name. A step can then name the
    value without knowing where it came from.
    """
    combined_values = dict(values)
    for actor_name, actor in actors.items():
        combined_values.update(
            {
                f"actors.{actor_name}.{key}": value
                for key, value in actor.values.items()
            }
        )
    return combined_values


def _journey(
    data: dict[str, Any],
    shared_env: dict[str, Any],
    index: int,
    coverage: Any = None,
    actors: Any = None,
    yaml_location: str = "the file",
) -> UserJourney:
    """One test with its workflow environment, coverage, and actors."""
    merged = dict(data)
    merged["env"] = {**shared_env, **(data.get("env") or {})}
    declared, secrets, refused = resolve(merged)
    declared_actors, actor_secrets, actor_displays, actor_verifications = _actors(
        actors, yaml_location
    )
    values = _merge_actor_values(declared, declared_actors)
    description = data.get("journey", data.get("description", ""))
    return UserJourney(
        id=str(data.get("id", f"journey_{index:02d}")),
        name=str(data.get("name", "User Journey")),
        description=substitute(description, values),
        security_rules=[
            substitute(rule, values) for rule in data.get("security_rules", [])
        ],
        secrets={**secrets, **actor_secrets},
        secret_displays={**secret_displays(merged), **actor_displays},
        value_verifications={
            **verification_policies(merged),
            **actor_verifications,
        },
        values=values,
        refused=refused,
        actors=declared_actors,
        coverage=_cells(coverage, yaml_location),
    )


def _journeys_of(
    block: dict[str, Any],
    shared_env: dict[str, Any],
    yaml_location: str,
    coverage: Any = None,
    actors: Any = None,
) -> list[UserJourney]:
    listed = block.get("journeys")
    if listed is None and "journey" in block:
        listed = [block]
    if listed is None:
        raise ConfigError(f"{yaml_location} has no journeys.")
    if not isinstance(listed, list):
        raise ConfigError(f"'journeys' in {yaml_location} must be a list.")
    return [
        _journey(item, shared_env, index + 1, coverage, actors, yaml_location)
        for index, item in enumerate(listed)
    ]


def _shared_env(block: dict[str, Any]) -> dict[str, Any]:
    declared = block.get("env") or {}
    if not isinstance(declared, dict):
        raise ConfigError("'env' must be a mapping of name to value.")
    return declared


def _validate(raw: dict[str, Any], path: Path) -> None:
    """Refuses a file Yewo cannot read, naming every key that is wrong.

    Every problem at once, so one pass over the file fixes them all.

    Raises:
        ConfigError: If the file does not match the yewo.yaml schema.
    """
    try:
        YewoFile.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  {line}" for line in format_validation_errors(exc)
        )
        raise ConfigError(f"{path} cannot be read:\n{problems}") from None


def load_workflows(path: Path, raw: Any = None) -> list[Workflow]:
    """Every workflow in a yewo.yaml, in the order the file lists them.

    A file with no `workflows:` is one unnamed workflow, so a journey file
    written before workflows existed still runs.

    Raises:
        ConfigError: If the file cannot be read as workflows.
    """
    if raw is None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.resolve().parent

    if isinstance(raw, list):
        return [Workflow("", {}, [_journey(item, {}, i + 1) for i, item in enumerate(raw)])]
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping or a list of journeys.")
    _validate(raw, path)

    top = _normalise(raw, str(path))
    shared = _shared_env(raw)
    top_coverage = raw.get("coverage")
    top_actors = raw.get("actors")

    declared = raw.get("workflows")
    if declared is None:
        return [
            Workflow(
                "",
                _anchor(top, base),
                _journeys_of(raw, shared, str(path), top_coverage, top_actors),
            )
        ]
    if not isinstance(declared, dict):
        raise ConfigError(f"'workflows' in {path} must be a mapping of name to workflow.")

    workflows = []
    for name, block in declared.items():
        yaml_location = f"workflow '{name}'"
        if not isinstance(block, dict):
            raise ConfigError(f"{yaml_location} in {path} must be a mapping.")
        settings = {**top, **_normalise(block, yaml_location)}
        env = {**shared, **_shared_env(block)}
        # A workflow owns one app, so its coverage replaces the file's rather
        # than adding to it. Half a declared set would be worse than either.
        coverage = block.get("coverage", top_coverage)
        actors = {**(top_actors or {}), **(block.get("actors") or {})} or None
        workflows.append(
            Workflow(
                str(name),
                _anchor(settings, base),
                _journeys_of(block, env, yaml_location, coverage, actors),
            )
        )
    return workflows


def select(workflows: list[Workflow], wanted: Optional[str]) -> list[Workflow]:
    """The workflows a run covers, or all of them.

    Raises:
        ConfigError: If a name was asked for and the file does not have it.
    """
    if not wanted:
        return workflows
    names = [name.strip() for name in wanted.split(",") if name.strip()]
    by_name = {workflow.name: workflow for workflow in workflows}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ConfigError(
            f"No workflow named {', '.join(missing)}. "
            f"This file has: {', '.join(by_name) or '(none named)'}"
        )
    return [by_name[name] for name in names]
