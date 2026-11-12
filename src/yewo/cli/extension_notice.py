"""Tell a caller when what they configured has nothing to carry it out."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from yewo.devices.logs.reader import DeviceLogReader
from yewo.devices.recording.recorder import ScreenRecorder
from yewo.extensions import ExtensionRegistry
from yewo.observations.service import ObservationSource

CLOUD_URL = "https://tryyewo.dev/cloud"


@dataclass(frozen=True)
class Capability:
    """One thing a package may do, the components it needs, and what its absence costs.

    A row here, not a branch and not a sentence in the renderer. One package claims its
    components together, so one capability covers all of them however many there are.
    """

    name: str
    components: tuple[type, ...]
    without_it: str
    needed_when: Callable[[RunDeclarations], bool]

    def is_missing(self, registry: ExtensionRegistry, declared: RunDeclarations) -> bool:
        """Whether this run needs this capability and no installed package claims it."""
        return self.needed_when(declared) and not all(
            registry.provides(component) for component in self.components
        )


@dataclass(frozen=True)
class RunDeclarations:
    """What this run asked for, in the terms its project file uses."""

    secret_labels: Sequence[str] = ()

    @property
    def summary(self) -> str:
        """What is at stake, named without naming any value."""
        count = len(self.secret_labels)
        return f"{count} secret value{'' if count == 1 else 's'}"


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="declared-value masking",
        components=(ObservationSource, ScreenRecorder, DeviceLogReader),
        without_it=(
            "every screenshot, the recording, and the device log keep the declared\n"
            "      values as the device produced them. Each screenshot also goes to\n"
            "      the model"
        ),
        needed_when=lambda declared: bool(declared.secret_labels),
    ),
)


def missing_notice(
    registry: ExtensionRegistry,
    declared: RunDeclarations,
) -> str | None:
    """Return what to say before a run, or None when there is nothing to say.

    Silent unless the caller asked for something this install cannot do. A run that declared
    nothing sees nothing, so the message keeps meaning something when it does appear.
    """
    missing = tuple(item for item in CAPABILITIES if item.is_missing(registry, declared))
    if not missing:
        return None
    listed = "\n\n".join(f"    {item.name}\n      {item.without_it}" for item in missing)
    return (
        f"! This run declares {declared.summary}.\n"
        f"  Nothing installed acts on {'them' if len(declared.secret_labels) > 1 else 'it'}:\n"
        "\n"
        f"{listed}\n"
        "\n"
        f"  Yewo Cloud provides {'these' if len(missing) > 1 else 'this'}. {CLOUD_URL}"
    )


def failure_notice(registry: ExtensionRegistry) -> str | None:
    """Return what to say about an installed package that could not load."""
    failures = registry.failures()
    if not failures:
        return None
    listed = "\n".join(f"    {item}" for item in failures)
    return (
        "! An installed extension could not start, so whatever it provides is not\n"
        f"  running. This run behaves as though it were never installed.\n{listed}"
    )
