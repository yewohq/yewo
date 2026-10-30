"""Reusable boundaries for device-log service tests."""

from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from yewo.applications import ApplicationDescriptor
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android import ANDROID_LOG_SOURCES
from yewo.devices.logs import (
    DeviceLogBatch,
    DeviceLogCapturePolicy,
    DeviceLogCaptureService,
    DeviceLogCheckpointKind,
    DeviceLogRedactionResult,
    DeviceLogSource,
    DeviceLogStore,
    DeviceLogTextEntry,
    DeviceLogTextScrubber,
)


CAPTURE_TIME = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
LOGCAT_SOURCE = ANDROID_LOG_SOURCES[0]


def text_batch(
    *entries: str,
    observed_item_count: int | None = None,
    filtered_item_count: int = 0,
    may_have_omitted_earlier_items: bool = False,
) -> DeviceLogBatch:
    return DeviceLogBatch(
        tuple(DeviceLogTextEntry(entry) for entry in entries),
        len(entries) if observed_item_count is None else observed_item_count,
        filtered_item_count=filtered_item_count,
        may_have_omitted_earlier_items=may_have_omitted_earlier_items,
    )


class ReaderStub:
    def __init__(
        self,
        baselines: dict[str, DeviceLogBatch | None | Exception],
        reads: dict[str, tuple[DeviceLogBatch | Exception, ...]],
        *,
        coarse_sources: frozenset[str] = frozenset(),
    ) -> None:
        self._baselines = baselines
        self._reads = {name: deque(values) for name, values in reads.items()}
        self._coarse_sources = coarse_sources
        self.read_log_types: list[str] = []
        self.acknowledged_batches: list[DeviceLogBatch] = []

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        value = self._baselines[log_type]
        if isinstance(value, Exception):
            raise value
        return value

    def read(self, log_type: str) -> DeviceLogBatch:
        self.read_log_types.append(log_type)
        value = self._reads[log_type].popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        return not (
            log_type in self._coarse_sources and checkpoint_kind is DeviceLogCheckpointKind.ACTION
        )

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        del log_type
        self.acknowledged_batches.append(batch)


def redact_text(text: str) -> DeviceLogRedactionResult:
    count_found = text.count("SECRET")
    return DeviceLogRedactionResult(
        text.replace("SECRET", "[redacted]"),
        "test-policy-v1",
        count_found,
    )


def build_service(
    reader: ReaderStub,
    root: Path,
    *,
    sources: tuple[DeviceLogSource, ...] = (LOGCAT_SOURCE,),
    platform: DevicePlatform = DevicePlatform.ANDROID,
    policy: DeviceLogCapturePolicy | None = None,
    store: DeviceLogStore | None = None,
    scrubber: DeviceLogTextScrubber = redact_text,
    time_factory: Callable[[], datetime] = lambda: CAPTURE_TIME,
) -> DeviceLogCaptureService:
    identifiers = (f"record_{number}" for number in count(1))
    artifacts = (f"artifact_{number}" for number in count(1))
    device = DeviceDescriptor(
        platform=platform,
        target_id="test-target",
        connection_id="test-connection",
        display_name="test device",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="ready",
    )
    application = ApplicationDescriptor(platform=platform, application_id="com.example.app")
    return DeviceLogCaptureService(
        reader,
        device,
        application,
        sources,
        store or DeviceLogStore(root, identifier_factory=artifacts.__next__),
        scrubber,
        policy=policy,
        identifier_factory=identifiers.__next__,
        time_factory=time_factory,
    )
