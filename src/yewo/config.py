"""Resolved settings for every Yewo command.

Precedence, strongest first:

1. an explicit CLI flag
2. a YEWO_* environment variable
3. ./yewo.yaml in the working directory or any parent
4. the built-in defaults

yewo.yaml holds the settings and the journeys together.

Each value remembers where it came from, so `yewo config` can explain itself.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from yewo.schema import (
    Settings as SettingsSpec,
    YewoFile as YewoFileSpec,
    format_validation_errors,
)
from yewo.defaults import (
    DEFAULT_APP_PATH,
    DEFAULT_DEVICE,
    DEFAULT_GROUNDING,
    DEFAULT_RECORD,
    DEFAULT_DURABLE_REPLAY,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_PERCEPTION_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATFORM,
    DEFAULT_SETTLE_TIME,
)

PROJECT_CONFIG_FILENAME = "yewo.yaml"
ENV_PREFIX = "YEWO_"

DEFAULT_ORIGIN = "built-in default"

logger = logging.getLogger(__name__)


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


SETTING_ALIASES = {"app": "app_path", "emulator": "device", "record_session": "record"}
"""Spellings a file may use for a setting Yewo knows by another name."""

SETTINGS: frozenset[str] = frozenset(
    name for name in SettingsSpec.model_fields if name not in SETTING_ALIASES
)
"""Every setting, taken from the schema that validates a yewo.yaml.

Naming them twice meant `yewo config set platform androdi` could be accepted
here and refused by the loader reading the same file.
"""

BUILT_IN: dict[str, Any] = {
    "platform": DEFAULT_PLATFORM,
    "device": DEFAULT_DEVICE,
    "model": DEFAULT_MODEL,
    "perception_model": DEFAULT_PERCEPTION_MODEL,
    "fast_model": None,
    "app_path": DEFAULT_APP_PATH,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "max_steps": DEFAULT_MAX_STEPS,
    "settle_time": DEFAULT_SETTLE_TIME,
    "headless": False,
    "grounding": DEFAULT_GROUNDING,
    "record": DEFAULT_RECORD,
    "durable_replay": DEFAULT_DURABLE_REPLAY,
    "allow_secret_exposure": False,
}


STRUCTURE_KEYS: frozenset[str] = frozenset(
    set(YewoFileSpec.model_fields) - set(SettingsSpec.model_fields)
)
"""Keys that describe what to test, not how Yewo runs.

Taken from the schema, because a key listed in one place and forgotten here
is reported as a misspelled setting while the run applies it.
"""

DEVICE_BLOCK = {"name": "device", "platform": "platform", "headless": "headless"}
"""What a `device:` block may name, and the setting each key resolves to.

One word for one thing: `device` names the target in a block, and the rest of
Yewo resolves it as the flat `device` setting.
"""


class ConfigError(Exception):
    """Raised for an unknown setting or a value that will not coerce."""


@dataclass
class Config:
    """Effective settings, with the origin of each value."""

    values: dict[str, Any] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    def get(self, name: str) -> Any:
        return self.values[name]

    def origin(self, name: str) -> str:
        return self.origins.get(name, DEFAULT_ORIGIN)

    def platforms(self, requested: Optional[str] = None) -> tuple[str, ...]:
        """Which platforms a command covers.

        Without an explicit choice, follow the configured target rather than
        assuming both. An Android-only user should not be told that a missing
        iOS driver blocks them.
        """
        choice = requested or self.values["platform"]
        return ("android", "ios") if choice == "all" else (choice,)

    def resolve(self, name: str, override: Any = None) -> Any:
        """An explicit CLI value wins. Otherwise the resolved setting applies."""
        if override is None or override is False:
            return self.values[name]
        return override


def coerce(name: str, raw: Any) -> Any:
    """One setting's value, checked against the schema that defines it."""
    if name not in SETTINGS:
        raise ConfigError(f"Unknown setting '{name}'. Known: {', '.join(sorted(SETTINGS))}")
    if raw is None:
        return None
    try:
        return SettingsSpec.model_validate({name: raw}).model_dump()[name]
    except ValidationError as exc:
        reason = "; ".join(format_validation_errors(exc))
        raise ConfigError(f"Invalid value for '{name}': {raw!r} ({reason})") from None


def expand_device(loaded: dict[str, Any], where: str) -> dict[str, Any]:
    """Turns a `device:` block into the flat settings Yewo resolves.

    A bare string is the device name on its own.

    Args:
        loaded: The mapping read from a file or a workflow.
        where: What to name in a message about an unusable value.
    """
    raw = loaded.get("device")
    if raw is None or isinstance(raw, str):
        return loaded
    if not isinstance(raw, dict):
        raise ConfigError(f"'device' in {where} must be a name or a mapping, not {type(raw).__name__}")

    expanded = {key: value for key, value in loaded.items() if key != "device"}
    unknown = sorted(key for key in raw if key not in DEVICE_BLOCK)
    if unknown:
        logger.warning(
            "Ignoring unknown key(s) in the device block of %s: %s. Known keys: %s",
            where,
            ", ".join(str(key) for key in unknown),
            ", ".join(sorted(DEVICE_BLOCK)),
        )
    for key, setting in DEVICE_BLOCK.items():
        if key not in raw:
            continue
        if setting in expanded:
            logger.warning(
                "%s sets '%s' twice: the device block wins over the top-level value.",
                where,
                setting,
            )
        expanded[setting] = raw[key]
    return expanded


def read_file(path: Path) -> dict[str, Any]:
    """Reads one config file, ignoring unknown keys so a newer file stays usable.

    Yewo names the unknown keys instead of dropping them in silence. A
    misspelled setting that does nothing looks exactly like one Yewo applied.
    """
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping of settings")
    loaded = expand_device(loaded, str(path))
    loaded = {SETTING_ALIASES.get(key, key): value for key, value in loaded.items()}
    known = {key: value for key, value in loaded.items() if key in SETTINGS}
    unknown = sorted(
        key for key in loaded if key not in SETTINGS and key not in STRUCTURE_KEYS
    )
    if unknown:
        logger.warning(
            "Ignoring unknown setting(s) in %s: %s. Known settings: %s",
            path,
            ", ".join(str(key) for key in unknown),
            ", ".join(sorted(SETTINGS)),
        )
    return known


def find_project_config(start: Optional[Path] = None) -> Optional[Path]:
    """Looks for yewo.yaml in this directory and its parents."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _env_overrides() -> dict[str, tuple[Any, str]]:
    found = {}
    for name in SETTINGS:
        variable = f"{ENV_PREFIX}{name.upper()}"
        raw = os.getenv(variable)
        if raw is not None:
            found[name] = (raw, variable)
    return found


def load_config(start: Optional[Path] = None) -> Config:
    """Builds the effective configuration from every layer."""
    config = Config(values=dict(BUILT_IN))
    config.origins = {name: DEFAULT_ORIGIN for name in BUILT_IN}

    path = find_project_config(start)
    if path is not None:
        settings = read_file(path)
        if settings:
            config.files.append(path)
        for name, raw in settings.items():
            config.values[name] = coerce(name, raw)
            config.origins[name] = str(path)

    for name, (raw, variable) in _env_overrides().items():
        config.values[name] = coerce(name, raw)
        config.origins[name] = variable

    return config


def config_path(start: Optional[Path] = None) -> Path:
    """The yewo.yaml a write targets, whether or not it exists yet."""
    existing = find_project_config(start)
    return existing or (start or Path.cwd()) / PROJECT_CONFIG_FILENAME


class _BlockDumper(yaml.SafeDumper):
    """Writes multi-line text as a block scalar, the way a person would."""


def _block_scalar(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockDumper.add_representer(str, _block_scalar)


def write_document(path: Path, document: dict[str, Any]) -> None:
    """Saves a config file, keeping it readable.

    A journey is a paragraph of prose. Written as a quoted scalar it becomes
    one long line, so it goes back as a block. Key order is kept for the same
    reason: this file is read and edited by hand.

    Comments do not survive, so a file that carries journeys says so.
    """
    if any(key in document for key in STRUCTURE_KEYS):
        logger.warning("Rewriting %s. Any comments in it are lost.", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(document, Dumper=_BlockDumper, sort_keys=False, width=100),
        encoding="utf-8",
    )


def read_document(path: Path) -> dict[str, Any]:
    """A whole config file, unfiltered.

    A write has to keep what it does not understand. The same file holds the
    journeys, so filtering to known settings before writing back would delete
    every test in it.
    """
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping of settings")
    return loaded


def _block_key(document: dict[str, Any], name: str) -> Optional[str]:
    """Which key of this file's device block holds a setting, if it has one."""
    device = document.get("device")
    if not isinstance(device, dict):
        return None
    for key, setting in DEVICE_BLOCK.items():
        if setting == name:
            return key
    return None


def _place_setting(document: dict[str, Any], name: str, value: Any) -> None:
    """Writes one setting where this file already keeps it.

    A file that describes its target as a block keeps it there, rather than
    gaining a second spelling of the same setting beside it.
    """
    key = _block_key(document, name)
    if key is None:
        document[name] = value
        return
    document["device"][key] = value


def _remove_setting(document: dict[str, Any], name: str) -> None:
    """Removes one setting from wherever this file keeps it.

    Only that setting. Removing the device by name must leave the platform
    and the rest of the block it shares.
    """
    key = _block_key(document, name)
    if key is None:
        document.pop(name, None)
        return
    document["device"].pop(key, None)


def set_value(name: str, raw: Any, start: Optional[Path] = None) -> Path:
    """Writes one setting into yewo.yaml, creating the file if needed."""
    value = coerce(name, raw)
    path = config_path(start)
    document = read_document(path)
    _place_setting(document, name, value)
    write_document(path, document)
    return path


def unset_value(name: str, start: Optional[Path] = None) -> Path:
    """Removes one setting from yewo.yaml, keeping everything else in it."""
    if name not in SETTINGS:
        raise ConfigError(f"Unknown setting '{name}'")
    path = config_path(start)
    document = read_document(path)
    _remove_setting(document, name)
    if document:
        write_document(path, document)
    elif path.is_file():
        path.unlink()
    return path
