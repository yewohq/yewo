"""The measurements Yewo can name in a scenario.

This catalogue is outside ``testing`` because a metric is a product contract,
not an implementation detail of one test runner.  A scenario names a metric;
the target decides whether it can collect it, and a collector later records
samples using the same unit and direction.
"""

from collections.abc import Iterable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from yewo.agent.device.platform import Platform


class MetricDirection(str, Enum):
    """Which side of a threshold is better for a metric."""

    LOWER = "lower"
    HIGHER = "higher"


class MetricUnit(str, Enum):
    """The unit stored with a metric sample."""

    MILLISECONDS = "ms"
    POINTS = "pt"
    COUNT = "count"


class MetricSource(str, Enum):
    """The collection boundary that produced a metric sample."""

    DEVICE = "device"
    ACCESSIBILITY_AUDIT = "accessibility_audit"


class TargetKind(str, Enum):
    """A target form that can determine whether collection is available."""

    EMULATOR = "emulator"
    PHYSICAL = "physical"


class Metric(str, Enum):
    """A closed name for a measurable scenario requirement."""

    APP_LAUNCH_TTID_MS = "app_launch_ttid_ms"
    MIN_TOUCH_TARGET = "min_touch_target"
    UNLABELLED_ACTIONABLE_COUNT = "unlabelled_actionable_count"


class MetricDefinition(Enum):
    """The immutable collection contract for one metric."""

    APP_LAUNCH_TTID_MS = (
        Metric.APP_LAUNCH_TTID_MS,
        MetricUnit.MILLISECONDS,
        MetricDirection.LOWER,
        frozenset({Platform.ANDROID, Platform.IOS}),
        frozenset({TargetKind.EMULATOR, TargetKind.PHYSICAL}),
        MetricSource.DEVICE,
        False,
        False,
    )
    MIN_TOUCH_TARGET = (
        Metric.MIN_TOUCH_TARGET,
        MetricUnit.POINTS,
        MetricDirection.HIGHER,
        frozenset({Platform.IOS}),
        frozenset({TargetKind.EMULATOR, TargetKind.PHYSICAL}),
        MetricSource.ACCESSIBILITY_AUDIT,
        True,
        False,
    )
    UNLABELLED_ACTIONABLE_COUNT = (
        Metric.UNLABELLED_ACTIONABLE_COUNT,
        MetricUnit.COUNT,
        MetricDirection.LOWER,
        frozenset({Platform.IOS}),
        frozenset({TargetKind.EMULATOR, TargetKind.PHYSICAL}),
        MetricSource.ACCESSIBILITY_AUDIT,
        False,
        False,
    )

    def __init__(
        self,
        metric: Metric,
        unit: MetricUnit,
        direction: MetricDirection,
        platforms: frozenset[Platform],
        target_kinds: frozenset[TargetKind],
        source: MetricSource,
        has_published_floor: bool,
        has_live_collector: bool,
    ) -> None:
        self.metric = metric
        self.unit = unit
        self.direction = direction
        self.platforms = platforms
        self.target_kinds = target_kinds
        self.source = source
        self.has_published_floor = has_published_floor
        self.has_live_collector = has_live_collector


def definition_for(metric: Metric) -> MetricDefinition:
    """Return the catalogue row for a metric."""
    return MetricDefinition[metric.name]


def metrics_without_live_collectors(metrics: Iterable[Metric]) -> list[Metric]:
    """Return requested metrics that scenario execution cannot collect yet."""
    return [metric for metric in metrics if not definition_for(metric).has_live_collector]


class Measurement(BaseModel):
    """One scrubbed numeric observation from a declared collection source."""

    metric: Metric
    value: float = Field(ge=0, allow_inf_nan=False)
    unit: MetricUnit
    source: MetricSource
    platform: Platform
    target_kind: TargetKind | None = None
    collected_at: datetime
    scenario_uuid: str | None = None

    @model_validator(mode="after")
    def _validate_catalogue_contract(self) -> "Measurement":
        definition = definition_for(self.metric)
        if self.unit is not definition.unit:
            raise ValueError(
                f"{self.metric.value} samples use {definition.unit.value}, not "
                f"{self.unit.value}."
            )
        if self.source is not definition.source:
            raise ValueError(
                f"{self.metric.value} samples come from {definition.source.value}, "
                f"not {self.source.value}."
            )
        if self.platform not in definition.platforms:
            raise ValueError(
                f"{self.metric.value} is not collectable on {self.platform.value}."
            )
        if self.target_kind is not None and self.target_kind not in definition.target_kinds:
            raise ValueError(
                f"{self.metric.value} is not collectable on a "
                f"{self.target_kind.value} target."
            )
        return self
