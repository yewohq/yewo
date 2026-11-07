"""Load and resolve a strict ``yewo.yaml`` project declaration."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re

from pydantic import ValidationError
import yaml

from yewo.project_config.capabilities import (
    MediaInjectCapabilitySpec,
    ScenarioPreconditionsSpec,
)
from yewo.project_config.records import (
    CoverageCell,
    CoverageCategory,
    CoveragePriority,
    CoverageSpec,
    DeclaredValueSpec,
    EnvironmentValue,
    JourneySpec,
    LoadedYewoProject,
    ModelOverridesSpec,
    ResolvedJourney,
    ResolvedSettings,
    ResolvedTarget,
    ResolvedValue,
    ResolvedWorkflow,
    ScenarioType,
    SettingsSpec,
    TargetSpec,
    ValueRedaction,
    WorkflowSpec,
    YewoFileSpec,
)


PROJECT_FILE_NAME = "yewo.yaml"
_ENVIRONMENT_REFERENCE = re.compile(r"\$\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}")
_VALUE_REFERENCE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
_SETTINGS_FIELDS = frozenset(SettingsSpec.model_fields)


class YewoProjectError(ValueError):
    """A public Yewo project declaration could not be loaded safely."""


def find_yewo_project(start: Path | None = None) -> Path | None:
    """Find the nearest ``yewo.yaml`` in this directory or one of its parents."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def load_yewo_project(path: Path) -> LoadedYewoProject:
    """Parse one YAML file and resolve its inherited workflows and journeys."""
    source_path = path.expanduser().resolve()
    payload = _read_yaml(source_path)
    if isinstance(payload, list):
        payload = {"journeys": payload}
    if not isinstance(payload, Mapping):
        raise YewoProjectError("A Yewo project file must contain a mapping or journey list.")
    try:
        declaration = YewoFileSpec.model_validate(dict(payload))
        top_settings = _settings_values(declaration)
        settings = _resolved_settings(top_settings, source_path.parent)
        workflows = _resolve_workflows(declaration, top_settings, source_path.parent)
    except ValidationError as error:
        raise YewoProjectError(_format_validation_error(source_path, error)) from None
    return LoadedYewoProject(
        source_path=source_path,
        settings=settings,
        workflows=workflows,
    )


def load_nearest_yewo_project(start: Path | None = None) -> LoadedYewoProject:
    """Load the nearest project file or report that none exists."""
    path = find_yewo_project(start)
    if path is None:
        raise YewoProjectError(
            "No yewo.yaml was found. Create it in this directory or one of its parents."
        )
    return load_yewo_project(path)


def select_workflows(
    project: LoadedYewoProject,
    requested: str | None,
) -> tuple[ResolvedWorkflow, ...]:
    """Return workflows named by a comma-separated selector, preserving its order."""
    if requested is None:
        return project.workflows
    names = tuple(name.strip() for name in requested.split(",") if name.strip())
    if not names:
        raise YewoProjectError("The workflow selector is empty.")
    workflows_by_name = {workflow.name: workflow for workflow in project.workflows}
    missing = tuple(name for name in names if name not in workflows_by_name)
    if missing:
        available = ", ".join(workflows_by_name) or "unnamed"
        raise YewoProjectError(
            f"Unknown workflow(s): {', '.join(missing)}. Available workflows: {available}."
        )
    return tuple(workflows_by_name[name] for name in names)


def _read_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise YewoProjectError(f"Could not read the Yewo project file: {path}.") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise YewoProjectError(f"The Yewo project file is not valid YAML{location}.") from None


def _format_validation_error(path: Path, error: ValidationError) -> str:
    problems: list[str] = []
    for issue in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in issue["loc"]) or "file"
        message = str(issue.get("msg", "is invalid"))
        problems.append(f"{location}: {message}")
    return f"{path} cannot be read:\n  " + "\n  ".join(problems)


def _resolve_workflows(
    declaration: YewoFileSpec,
    top_settings: dict[str, object],
    base_path: Path,
) -> tuple[ResolvedWorkflow, ...]:
    if declaration.workflows is None:
        if (
            declaration.journeys is None
            and declaration.journey is None
            and declaration.description is None
        ):
            return ()
        workflow = _inline_workflow(declaration)
        return (
            _resolve_workflow(
                "",
                workflow,
                declaration,
                top_settings,
                base_path,
            ),
        )
    return tuple(
        _resolve_workflow(name, workflow, declaration, top_settings, base_path)
        for name, workflow in declaration.workflows.items()
    )


def _inline_workflow(declaration: YewoFileSpec) -> WorkflowSpec:
    payload = declaration.model_dump(mode="python", exclude_none=True)
    payload.pop("workflows", None)
    return WorkflowSpec.model_validate(payload)


def _resolve_workflow(
    name: str,
    declaration: WorkflowSpec,
    project: YewoFileSpec,
    top_settings: dict[str, object],
    base_path: Path,
) -> ResolvedWorkflow:
    workflow_settings = _merge_settings(top_settings, _settings_values(declaration))
    settings = _resolved_settings(workflow_settings, base_path)
    env = {**project.env, **declaration.env}
    secret_names = _secret_names(project.secrets) | _secret_names(declaration.secrets)
    actors = {**(project.actors or {}), **(declaration.actors or {})}
    coverage = declaration.coverage if declaration.coverage is not None else project.coverage
    preconditions = (
        declaration.preconditions or project.preconditions or ScenarioPreconditionsSpec()
    )
    security_rules = _unique((*project.security_rules, *declaration.security_rules))
    checks = _unique((*project.checks, *declaration.checks))
    journeys = _journey_specs(declaration)
    resolved_journeys = tuple(
        _resolve_journey(
            journey,
            index,
            env,
            secret_names,
            actors,
            coverage,
            preconditions,
            security_rules,
            checks,
            base_path,
        )
        for index, journey in enumerate(journeys, start=1)
    )
    _validate_unique_journey_ids(name, resolved_journeys)
    policy_waivers = {
        **(project.policy.waive if project.policy else {}),
        **(declaration.policy.waive if declaration.policy else {}),
    }
    target_specs = declaration.targets if declaration.targets is not None else project.targets
    targets = tuple(
        _resolve_target(target, workflow_settings, base_path) for target in target_specs or ()
    )
    return ResolvedWorkflow(
        name=name,
        settings=settings,
        journeys=resolved_journeys,
        policy_waivers=policy_waivers,
        targets=targets,
    )


def _journey_specs(workflow: WorkflowSpec) -> tuple[JourneySpec, ...]:
    if workflow.journeys is not None:
        return workflow.journeys
    return (
        JourneySpec(
            id=workflow.id,
            name=workflow.name,
            journey=workflow.journey,
            description=workflow.description,
            security_rules=(),
            checks=(),
            env={},
            preconditions=None,
        ),
    )


def _resolve_journey(
    declaration: JourneySpec,
    index: int,
    shared_env: dict[str, EnvironmentValue],
    secret_names: set[str],
    actor_specs: dict[str, dict[str, EnvironmentValue]],
    coverage_specs: tuple[CoverageSpec, ...] | None,
    inherited_preconditions: ScenarioPreconditionsSpec,
    inherited_security_rules: tuple[str, ...],
    inherited_checks: tuple[str, ...],
    base_path: Path,
) -> ResolvedJourney:
    all_declared_values = {**shared_env, **declaration.env}
    actor_value_names = {
        value_name for actor_values in actor_specs.values() for value_name in actor_values
    }
    missing_secrets = secret_names - set(all_declared_values) - actor_value_names
    if missing_secrets:
        raise YewoProjectError(
            "env or actors do not declare secret value(s): "
            + ", ".join(sorted(missing_secrets))
            + "."
        )
    declared_values = _resolve_values(
        all_declared_values,
        secret_names.intersection(all_declared_values),
        "env",
    )
    actors = {
        actor_name: _resolve_values(
            values,
            secret_names.intersection(values),
            f"actors.{actor_name}",
        )
        for actor_name, values in actor_specs.items()
    }
    objective = declaration.journey or declaration.description
    if objective is None:
        raise YewoProjectError("A resolved journey must contain an objective.")
    available_names = set(declared_values)
    available_names.update(
        f"actors.{actor_name}.{value_name}"
        for actor_name, values in actors.items()
        for value_name in values
    )
    security_rules = _unique((*inherited_security_rules, *declaration.security_rules))
    checks = _unique((*inherited_checks, *declaration.checks))
    _validate_references(objective, (*security_rules, *checks), available_names)
    preconditions = _anchor_preconditions(
        declaration.preconditions or inherited_preconditions,
        base_path,
    )
    return ResolvedJourney(
        journey_id=declaration.id or f"journey_{index:02d}",
        name=declaration.name or "User Journey",
        objective=objective,
        security_rules=security_rules,
        checks=checks,
        declared_values=declared_values,
        actors=actors,
        coverage=_coverage_cells(coverage_specs),
        preconditions=preconditions,
    )


def _resolve_values(
    declarations: dict[str, EnvironmentValue],
    listed_secrets: set[str],
    location: str,
) -> dict[str, ResolvedValue]:
    missing = listed_secrets - set(declarations)
    if missing:
        raise YewoProjectError(
            f"{location} does not declare secret value(s): {', '.join(sorted(missing))}."
        )
    resolved: dict[str, ResolvedValue] = {}
    for name, declaration in declarations.items():
        spec = (
            declaration
            if isinstance(declaration, DeclaredValueSpec)
            else DeclaredValueSpec(value=declaration)
        )
        value = _expand_environment(name, str(spec.value), location)
        is_secret = spec.secret or name in listed_secrets
        resolved[name] = ResolvedValue(
            name=name,
            value=value,
            secret=is_secret,
            refused=spec.refused,
            redaction=spec.redaction if is_secret else ValueRedaction.HIDE,
            verify_as=spec.verify_as,
            normalizers=spec.normalizers,
            visible_suffix=spec.visible_suffix,
            masked_length=spec.masked_length,
        )
    return resolved


def _expand_environment(name: str, value: str, location: str) -> str:
    referenced = set(_ENVIRONMENT_REFERENCE.findall(value))
    missing = sorted(
        variable for variable in referenced if not os.environ.get(variable, "").strip()
    )
    if missing:
        raise YewoProjectError(
            f"{location}.{name} requires unset or empty environment variable(s): "
            f"{', '.join(missing)}."
        )
    return _ENVIRONMENT_REFERENCE.sub(lambda match: os.environ[match.group(1)], value)


def _validate_references(
    objective: str,
    requirements: tuple[str, ...],
    available_names: set[str],
) -> None:
    referenced = {
        name for text in (objective, *requirements) for name in _VALUE_REFERENCE.findall(text)
    }
    missing = sorted(referenced - available_names)
    if missing:
        raise YewoProjectError(
            "Journey text references undeclared value(s): " + ", ".join(missing) + "."
        )


def _coverage_cells(specs: tuple[CoverageSpec, ...] | None) -> tuple[CoverageCell, ...]:
    if specs is None:
        return (
            CoverageCell(
                category=CoverageCategory.FUNCTIONAL,
                scenario_type=ScenarioType.HAPPY_PATH,
                priority=CoveragePriority.P0,
                index=0,
            ),
        )
    if not specs:
        raise YewoProjectError("Coverage must not be empty.")
    counts: dict[tuple[object, object], int] = {}
    cells: list[CoverageCell] = []
    for spec in specs:
        for scenario_type in spec.types:
            key = (spec.category, scenario_type)
            index = counts.get(key, 0)
            counts[key] = index + 1
            cells.append(
                CoverageCell(
                    category=spec.category,
                    scenario_type=scenario_type,
                    priority=spec.priority,
                    index=index,
                )
            )
    return tuple(cells)


def _settings_values(declaration: SettingsSpec) -> dict[str, object]:
    payload = declaration.model_dump(mode="python", exclude_none=True)
    values = {name: payload[name] for name in _SETTINGS_FIELDS if name in payload}
    device = values.pop("device", None)
    emulator = values.pop("emulator", None)
    if isinstance(device, dict):
        if device.get("name") is not None:
            values["device_name"] = device["name"]
        for name in ("platform", "headless"):
            if device.get(name) is not None:
                values[name] = device[name]
    elif device is not None:
        values["device_name"] = device
    elif emulator is not None:
        values["device_name"] = emulator
    app = values.pop("app", None)
    app_path = values.pop("app_path", None)
    if app is not None or app_path is not None:
        values["app_path"] = app if app is not None else app_path
    record = values.pop("record", None)
    record_session = values.pop("record_session", None)
    if record is not None or record_session is not None:
        values["record"] = record if record is not None else record_session
    return values


def _merge_settings(
    inherited: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    merged = {**inherited, **overrides}
    inherited_models = inherited.get("models")
    override_models = overrides.get("models")
    if isinstance(inherited_models, dict) and isinstance(override_models, dict):
        merged["models"] = {**inherited_models, **override_models}
    return merged


def _resolved_settings(values: dict[str, object], base_path: Path) -> ResolvedSettings:
    payload = {
        name: _expand_setting_environment(value, f"settings.{name}")
        for name, value in values.items()
    }
    for name in ("app_path", "output_dir", "memory_root"):
        raw = payload.get(name)
        if raw is not None:
            path = Path(str(raw)).expanduser()
            payload[name] = path.resolve() if path.is_absolute() else (base_path / path).resolve()
    models = payload.get("models")
    if isinstance(models, dict):
        payload["models"] = ModelOverridesSpec.model_validate(models)
    return ResolvedSettings.model_validate(payload)


def _expand_setting_environment(value: object, location: str) -> object:
    if isinstance(value, str):
        referenced = set(_ENVIRONMENT_REFERENCE.findall(value))
        missing = sorted(
            variable for variable in referenced if not os.environ.get(variable, "").strip()
        )
        if missing:
            raise YewoProjectError(
                f"{location} requires unset or empty environment variable(s): {', '.join(missing)}."
            )
        return _ENVIRONMENT_REFERENCE.sub(lambda match: os.environ[match.group(1)], value)
    if isinstance(value, dict):
        return {
            name: _expand_setting_environment(item, f"{location}.{name}")
            for name, item in value.items()
        }
    return value


def _resolve_target(
    target: TargetSpec,
    workflow_settings: dict[str, object],
    base_path: Path,
) -> ResolvedTarget:
    overrides: dict[str, object] = {
        "platform": target.platform,
        "device_name": target.device,
    }
    if target.headless is not None:
        overrides["headless"] = target.headless
    if target.app is not None or target.app_path is not None:
        overrides["app_path"] = target.app if target.app is not None else target.app_path
    return ResolvedTarget(
        target_id=target.id,
        settings=_resolved_settings(_merge_settings(workflow_settings, overrides), base_path),
    )


def _anchor_preconditions(
    preconditions: ScenarioPreconditionsSpec,
    base_path: Path,
) -> ScenarioPreconditionsSpec:
    capabilities = tuple(
        capability.model_copy(
            update={
                "local_path": str((base_path / capability.local_path).resolve())
                if not Path(capability.local_path).is_absolute()
                else str(Path(capability.local_path).resolve())
            }
        )
        if isinstance(capability, MediaInjectCapabilitySpec)
        else capability
        for capability in preconditions.capabilities
    )
    return preconditions.model_copy(update={"capabilities": capabilities})


def _secret_names(value: tuple[str, ...] | str) -> set[str]:
    return {value} if isinstance(value, str) else set(value)


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def _validate_unique_journey_ids(
    workflow_name: str,
    journeys: tuple[ResolvedJourney, ...],
) -> None:
    identifiers = [journey.journey_id for journey in journeys]
    if len(identifiers) != len(set(identifiers)):
        label = workflow_name or "the unnamed workflow"
        raise YewoProjectError(f"Journey identifiers must be unique in {label}.")
