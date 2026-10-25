"""Target-aware collection of scenario measurements."""

import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Optional, Protocol

from yewo.agent.device.launch import LaunchTiming, SetupMode
from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.platform import Platform
from yewo.errors import MeasurementUnavailable
from yewo.metrics import (
    Measurement,
    MeasurementPurpose,
    MeasurementRelation,
    Metric,
    MetricSource,
    MetricUnit,
    TargetKind,
)
from yewo.testing.judge.capabilities import CapabilityTarget
from yewo.testing.report.measurement_log import MeasurementLog

logger = logging.getLogger(__name__)

IOS_ACCESSIBILITY_RELEASE = (17,)
IOS_TOUCH_TARGET_FLOOR_PT = 44.0
IOS_ACTIONABLE_TYPES = frozenset(
    {
        "Button",
        "Link",
        "TextField",
        "SecureTextField",
        "Switch",
        "Slider",
        "Stepper",
        "Picker",
        "DatePicker",
        "SegmentedControl",
    }
)


class MeasurementDriver(Protocol):
    """The one Appium command used by the accessibility collector."""

    def execute_script(self, script: str, args: dict) -> object: ...


class MeasurementDevice(Protocol):
    """The device facts and command needed by live collectors."""

    platform: Platform
    os_version: Optional[str]
    model: Optional[str]
    device_name: str
    driver: MeasurementDriver
    serial: Optional[str]
    app_id: Optional[str]
    package_id: Optional[str]


def collector_refusal(
    metric: Metric,
    target: CapabilityTarget,
    setup: SetupMode,
) -> Optional[str]:
    """Return why this exact target cannot collect a metric."""
    if metric is Metric.APP_LAUNCH_TTID_MS:
        if target.platform is Platform.IOS:
            return (
                "app_launch_ttid_ms has no accepted iOS collector. Appium command "
                "duration is not an application first-frame measurement"
            )
        if setup is SetupMode.PRESERVE:
            return (
                "app_launch_ttid_ms needs a process launch, but preserve forbids "
                "restarting the app"
            )
        return None
    if target.platform is not Platform.IOS:
        return f"{metric.value} has no collector on {target.platform.value}"
    if target.release is not None and target.release < IOS_ACCESSIBILITY_RELEASE:
        return f"{metric.value} needs iOS 17 or newer"
    return None


def _target_kind(target: CapabilityTarget) -> Optional[TargetKind]:
    if target.emulator is None:
        return None
    return TargetKind.EMULATOR if target.emulator else TargetKind.PHYSICAL


def _audit_type(issue: dict) -> str:
    return str(issue.get("auditType") or "").lower()


def _rect(issue: dict) -> Optional[dict]:
    attributes = issue.get("elementAttributes")
    if not isinstance(attributes, dict):
        return None
    rect = attributes.get("rect")
    return rect if isinstance(rect, dict) else None


def _element_type(issue: dict) -> Optional[str]:
    attributes = issue.get("elementAttributes")
    if not isinstance(attributes, dict):
        return None
    value = attributes.get("type")
    if not isinstance(value, str) or not value:
        return None
    return value.removeprefix("XCUIElementType")


def _dimension(rect: dict, name: str) -> Optional[float]:
    value = rect.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 else None


def accessibility_values(
    payload: object,
    metrics: Iterable[Metric],
) -> dict[Metric, tuple[float, MeasurementRelation]]:
    """Derive bounded numeric evidence from an XCUITest audit response."""
    if not isinstance(payload, list) or any(not isinstance(issue, dict) for issue in payload):
        raise MeasurementUnavailable("The iOS accessibility audit returned an invalid payload.")
    requested = set(metrics)
    known_types = ("hitregion", "sufficientelementdescription")
    if any(not any(name in _audit_type(issue) for name in known_types) for issue in payload):
        raise MeasurementUnavailable(
            "The iOS accessibility audit returned an unknown issue type."
        )
    values: dict[Metric, tuple[float, MeasurementRelation]] = {}
    if Metric.UNLABELLED_ACTIONABLE_COUNT in requested:
        description_issues = [
            issue
            for issue in payload
            if "sufficientelementdescription" in _audit_type(issue)
        ]
        types = [_element_type(issue) for issue in description_issues]
        if any(element_type is None for element_type in types):
            raise MeasurementUnavailable(
                "An element-description issue had no element type."
            )
        count = sum(element_type in IOS_ACTIONABLE_TYPES for element_type in types)
        values[Metric.UNLABELLED_ACTIONABLE_COUNT] = (
            float(count),
            MeasurementRelation.EXACT,
        )
    if Metric.MIN_TOUCH_TARGET in requested:
        hit_issues = [issue for issue in payload if "hitregion" in _audit_type(issue)]
        sizes = []
        for issue in hit_issues:
            rect = _rect(issue)
            width = _dimension(rect or {}, "width")
            height = _dimension(rect or {}, "height")
            if width is None or height is None:
                raise MeasurementUnavailable(
                    "A hit-region audit issue had no numeric element rectangle."
                )
            sizes.append(min(width, height))
        values[Metric.MIN_TOUCH_TARGET] = (
            (min(sizes), MeasurementRelation.EXACT)
            if sizes
            else (IOS_TOUCH_TARGET_FLOOR_PT, MeasurementRelation.AT_LEAST)
        )
    return values


def janky_frame_percent(output: str) -> float:
    """Read the worst app-scoped janky-frame percentage from gfxinfo."""
    values = [
        float(value)
        for value in re.findall(r"(?m)^Janky frames:\s*\d+\s*\(([0-9.]+)%\)\s*$", output)
    ]
    if not values:
        raise MeasurementUnavailable("Android gfxinfo reported no janky-frame percentage.")
    return max(values)


class ScenarioMeasurementCollector:
    """Collect and persist measurements for one scenario."""

    def __init__(
        self,
        device: MeasurementDevice,
        target: CapabilityTarget,
        scenario_uuid: str,
        setup: SetupMode,
        required_metrics: Iterable[Metric],
        log: Optional[MeasurementLog] = None,
    ) -> None:
        self.device = device
        self.target = target
        self.scenario_uuid = scenario_uuid
        self.setup = setup
        self.required_metrics = frozenset(required_metrics)
        self.log = log
        self.samples: list[Measurement] = []
        self.failures: list[str] = []
        self._screen_ids: set[str] = set()
        self._iteration_by_metric: dict[Metric, int] = {}
        self._frame_window_started = False

    def _purpose(self, metric: Metric) -> MeasurementPurpose:
        return (
            MeasurementPurpose.REQUIRED
            if metric in self.required_metrics
            else MeasurementPurpose.DIAGNOSTIC
        )

    def _record(
        self,
        metric: Metric,
        value: float,
        relation: MeasurementRelation,
        source: MetricSource,
        collector: str,
        *,
        launch_type: Optional[str] = None,
        screen: Optional[str] = None,
    ) -> None:
        iteration = self._iteration_by_metric.get(metric, 0) + 1
        self._iteration_by_metric[metric] = iteration
        sample = Measurement(
            metric=metric,
            value=value,
            relation=relation,
            purpose=self._purpose(metric),
            unit=(
                MetricUnit.MILLISECONDS
                if metric is Metric.APP_LAUNCH_TTID_MS
                else MetricUnit.POINTS
                if metric is Metric.MIN_TOUCH_TARGET
                else MetricUnit.PERCENT
                if metric is Metric.JANKY_FRAME_PERCENT
                else MetricUnit.COUNT
            ),
            source=source,
            collector=collector,
            platform=self.device.platform,
            target_kind=_target_kind(self.target),
            os_version=self.device.os_version,
            device_model=self.device.model or self.device.device_name,
            app_data_state=self.setup.value,
            launch_type=launch_type,
            screen=screen,
            iteration=iteration,
            collected_at=datetime.now(timezone.utc),
            scenario_uuid=self.scenario_uuid,
        )
        self.samples.append(sample)
        if self.log is not None:
            self.log.append(sample)

    def record_launch(self, timing: LaunchTiming) -> None:
        """Record Android's platform-reported cold-process launch time."""
        self._record(
            Metric.APP_LAUNCH_TTID_MS,
            timing.duration_ms,
            MeasurementRelation.EXACT,
            MetricSource.DEVICE,
            timing.source,
            launch_type=timing.launch_type,
        )

    def record_failure(self, reason: str) -> None:
        """Keep a bounded diagnostic reason without failing the UI walk."""
        if reason not in self.failures:
            self.failures.append(reason)
            logger.warning("Measurement collection failed: %s", reason)

    def begin_frame_diagnostics(self) -> None:
        """Reset Android's app-scoped frame interval before visible UI work."""
        if self.device.platform is not Platform.ANDROID or self.log is None:
            return
        app_id = self.device.app_id or self.device.package_id
        if not self.device.serial or not app_id:
            self.record_failure("Android frame telemetry has no target serial or app id.")
            return
        try:
            AdbControl.shell(
                self.device.serial,
                "dumpsys",
                "gfxinfo",
                app_id,
                "reset",
                doing=f"reset frame telemetry for {app_id}",
            )
            self._frame_window_started = True
        except Exception as exc:  # noqa: BLE001 - adb boundary
            self.record_failure(str(exc))

    def finish_frame_diagnostics(self) -> None:
        """Record Android jank for the bounded scenario interval."""
        if not self._frame_window_started:
            return
        app_id = self.device.app_id or self.device.package_id
        output = AdbControl.ask(
            self.device.serial or "",
            "dumpsys",
            "gfxinfo",
            app_id or "",
        )
        try:
            if output is None:
                raise MeasurementUnavailable("Android gfxinfo returned no frame telemetry.")
            self._record(
                Metric.JANKY_FRAME_PERCENT,
                janky_frame_percent(output),
                MeasurementRelation.EXACT,
                MetricSource.DEVICE,
                "Android dumpsys gfxinfo",
            )
        except MeasurementUnavailable as exc:
            self.record_failure(str(exc))
        finally:
            self._frame_window_started = False

    def observe_screen(self, screen_id: object, step: int) -> None:
        """Run requested iOS accessibility audits once per observed screen."""
        metrics = self.required_metrics.intersection(
            {Metric.MIN_TOUCH_TARGET, Metric.UNLABELLED_ACTIONABLE_COUNT}
        )
        key = str(screen_id or f"step:{step}")
        if self.device.platform is not Platform.IOS or not metrics or key in self._screen_ids:
            return
        self._screen_ids.add(key)
        audit_types = []
        if Metric.MIN_TOUCH_TARGET in metrics:
            audit_types.append("XCUIAccessibilityAuditTypeHitRegion")
        if Metric.UNLABELLED_ACTIONABLE_COUNT in metrics:
            audit_types.append("XCUIAccessibilityAuditTypeSufficientElementDescription")
        try:
            payload = self.device.driver.execute_script(
                "mobile: performAccessibilityAudit",
                {"auditTypes": audit_types},
            )
            for metric, (value, relation) in accessibility_values(payload, metrics).items():
                self._record(
                    metric,
                    value,
                    relation,
                    MetricSource.ACCESSIBILITY_AUDIT,
                    "Appium XCUITest performAccessibilityAudit",
                    screen=key,
                )
        except Exception as exc:  # noqa: BLE001 - Appium SDK boundary
            self.record_failure(str(exc))
