"""Application boundary that reuses compatible knowledge or starts discovery."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from yewo.discovery.artifacts import DiscoveryCompatibilityKey
from yewo.discovery.contracts import DiscoveryRun
from yewo.discovery.controller import DiscoveryController
from yewo.discovery.knowledge import AppMap
from yewo.discovery.runtime import (
    DiscoveryProposer,
    DiscoveryResult,
    DiscoveryRuntime,
)


class DiscoveryRepository(Protocol):
    def find_compatible_app_map(
        self, compatibility: DiscoveryCompatibilityKey
    ) -> AppMap | None: ...

    def save_discovery_run(self, run: DiscoveryRun) -> object: ...


@dataclass(frozen=True)
class DiscoveryCoordinationResult:
    """Exactly one source of mechanics: an existing map or a new run."""

    app_map: AppMap | None = None
    discovery: DiscoveryResult | None = None

    def __post_init__(self) -> None:
        if (self.app_map is None) == (self.discovery is None):
            raise ValueError("Coordination requires exactly one knowledge source.")


class DiscoveryCoordinator:
    """Start bounded discovery only when exact compatible knowledge is absent."""

    def __init__(self, repository: DiscoveryRepository):
        self.repository = repository

    def ensure_knowledge(
        self,
        compatibility: DiscoveryCompatibilityKey,
        run_factory: Callable[[], DiscoveryRun],
        runtime: DiscoveryRuntime,
        proposer: DiscoveryProposer,
        *,
        scenario_owned_values: Sequence[str] = (),
        interrupted: Callable[[], bool] = lambda: False,
    ) -> DiscoveryCoordinationResult:
        app_map = self.repository.find_compatible_app_map(compatibility)
        if app_map is not None:
            return DiscoveryCoordinationResult(app_map=app_map)

        run = run_factory()
        if not run.header.compatibility.is_compatible_with(compatibility):
            raise ValueError("The discovery run does not match the requested compatibility.")
        discovery = DiscoveryController(
            run, scenario_owned_values=scenario_owned_values
        ).run_to_terminal(
            runtime,
            proposer,
            interrupted=interrupted,
        )
        self.repository.save_discovery_run(discovery.run)
        return DiscoveryCoordinationResult(discovery=discovery)
