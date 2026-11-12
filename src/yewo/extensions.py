"""Components a separately installed package may provide or wrap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, TypeVar

EXTENSION_GROUP = "yewo.extensions"

_Component = TypeVar("_Component")


class ExtensionDiagnosticError(RuntimeError):
    """An extension failure whose message is safe for the run trace."""


class RunResource(Protocol):
    """Extension-owned work whose lifetime is one real run."""

    def start(self) -> None:
        """Start work without waiting for it to finish."""
        ...

    def close(self) -> None:
        """Release resources after every consumer has closed."""
        ...


@dataclass(frozen=True)
class DeclaredValue:
    """One value this run typed into the application."""

    label: str
    value: str

    def __repr__(self) -> str:
        """Show the label, never the value."""
        return f"DeclaredValue(label={self.label!r})"


class DeclaredValues:
    """The values this run has typed so far."""

    def __init__(self) -> None:
        """Start with nothing typed."""
        self._values: list[DeclaredValue] = []

    def add(self, label: str, value: str) -> None:
        """Record one value that reached the device."""
        if not label or not value:
            raise ValueError("A declared value needs a label and a value.")
        entry = DeclaredValue(label=label, value=value)
        if entry not in self._values:
            self._values.append(entry)

    def all(self) -> tuple[DeclaredValue, ...]:
        """Everything typed so far."""
        return tuple(self._values)


@dataclass(frozen=True)
class RunContext:
    """What an extension is given for one run.

    `declared_values` is live: an extension holding it sees each value as the run types it,
    rather than a copy taken before anything had been typed.
    """

    artifact_root: Path
    run_id: str
    declared_values: DeclaredValues


class ExtensionRegistry:
    """What installed packages provide, and what they wrap."""

    def __init__(self) -> None:
        """Start empty."""
        self._provided: dict[type, Any] = {}
        self._wrappers: dict[type, list[Callable[[Any], Any]]] = {}
        self._run_resources: list[RunResource] = []
        self._started_run_resources: list[RunResource] = []
        self._failures: list[str] = []

    def provide(self, component: type[_Component], instance: _Component) -> None:
        """Fill a slot. The first package to claim it keeps it."""
        self._provided.setdefault(component, instance)

    def find(self, component: type[_Component]) -> _Component | None:
        """Return what fills a slot, or None."""
        return self._provided.get(component)

    def provides(self, component: type) -> bool:
        """Whether anything fills this slot, by providing it or by wrapping it."""
        return component in self._provided or bool(self._wrappers.get(component))

    def wrap(
        self, component: type[_Component], wrapper: Callable[[_Component], _Component]
    ) -> None:
        """Register a decorator for a component this project builds itself."""
        self._wrappers.setdefault(component, []).append(wrapper)

    def wrapped(self, component: type[_Component], instance: _Component) -> _Component:
        """Apply every registered decorator, in registration order."""
        for wrapper in self._wrappers.get(component, ()):
            instance = wrapper(instance)
        return instance

    def add_run_resource(self, resource: RunResource) -> None:
        """Register extension work that starts only for a real run."""
        self._run_resources.append(resource)

    def start_run_resources(self) -> None:
        """Start each registered resource once and roll back a partial start."""
        if self._started_run_resources:
            return
        try:
            for resource in self._run_resources:
                self._started_run_resources.append(resource)
                resource.start()
        except (Exception, KeyboardInterrupt):
            self.close_run_resources()
            raise

    def close_run_resources(self) -> None:
        """Close every started resource in reverse registration order."""
        first_error: Exception | KeyboardInterrupt | None = None
        while self._started_run_resources:
            resource = self._started_run_resources.pop()
            try:
                resource.close()
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def record_failure(self, description: str) -> None:
        """Note a package that could not be loaded."""
        self._failures.append(description)

    def filled(self) -> tuple[str, ...]:
        """Name every component an installed package provides or wraps."""
        names = {component.__name__ for component in self._provided}
        names |= {component.__name__ for component in self._wrappers}
        return tuple(sorted(names))

    def failures(self) -> tuple[str, ...]:
        """Packages that failed to load."""
        return tuple(self._failures)


def load_extensions(
    context: RunContext,
    group: str = EXTENSION_GROUP,
) -> ExtensionRegistry:
    """Build the registry from what this environment has installed.

    This records a package that fails to load rather than raising, so a broken one cannot stop
    a run. It records rather than ignores it, so it never looks like one nobody installed.

    This runs more than once per run. The pre-run notice needs to know what is installed
    before anything touches a device, and asks with a placeholder context. So `setup` has to be
    cheap, and must do nothing with its context except hold it.
    """
    registry = ExtensionRegistry()
    for entry in entry_points(group=group):
        try:
            entry.load()(registry, context)
        except Exception as error:  # noqa: BLE001 - another package's failure is not this run's
            registry.record_failure(f"{entry.name}: {error}")
    return registry
