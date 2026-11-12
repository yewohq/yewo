"""Bounded device-log capture across explicit execution boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from yewo.applications.records import ApplicationDescriptor
from yewo.artifacts.records import (
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    new_record_identifier,
    validate_record_identifier,
)
from yewo.devices.inventory import DeviceDescriptor
from yewo.devices.logs.artifact_producer import (
    DeviceLogArtifactProducer,
    DeviceLogBatchPersistence,
)
from yewo.devices.logs.capture import DeviceLogCapturePolicy, DeviceLogCaptureStart
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import DeviceLogBatch, DeviceLogReader
from yewo.devices.logs.records import (
    DeviceLogBoundary,
    DeviceLogBoundarySource,
    DeviceLogBoundaryState,
    DeviceLogCheckpointKind,
    DeviceLogDrainReceipt,
    DeviceLogDrainState,
    DeviceLogSource,
    DeviceLogSourceReceipt,
    validate_device_log_checkpoint,
)
from yewo.devices.logs.store import DeviceLogStore
from yewo.devices.diagnostics import ApplicationDiagnosticClassification


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class _SourceBaseline:
    boundary_source: DeviceLogBoundarySource
    batch: DeviceLogBatch | None


class DeviceLogCaptureService:
    """Establish, drain, and finish one incremental device-log interval."""

    def __init__(
        self,
        reader: DeviceLogReader,
        device: DeviceDescriptor,
        application: ApplicationDescriptor,
        sources: Sequence[DeviceLogSource],
        store: DeviceLogStore,
        *,
        policy: DeviceLogCapturePolicy | None = None,
        identifier_factory: Callable[[], str] = new_record_identifier,
        time_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        if device.platform is not application.platform:
            raise ValueError("The device-log device and application must use the same platform.")
        if not sources:
            raise ValueError("Device-log capture requires at least one source.")
        log_types = [source.log_type for source in sources]
        if len(log_types) != len(set(log_types)):
            raise ValueError("Device-log capture sources must be unique.")
        self._reader = reader
        self._device = device
        self._application = application
        self._sources = tuple(sources)
        self._store = store
        self._artifact_producer = DeviceLogArtifactProducer(
            store,
            policy or DeviceLogCapturePolicy(),
        )
        self._identifier_factory = identifier_factory
        self._time_factory = time_factory
        self._active_boundary: DeviceLogBoundary | None = None
        self._next_drain_index = 0
        self._completed_checkpoints: set[tuple[DeviceLogCheckpointKind, str]] = set()

    def begin(self) -> DeviceLogCaptureStart:
        """Persist a source baseline and any startup entries."""
        if self._active_boundary is not None:
            raise DeviceLogCaptureError("A device-log boundary is already active.")
        boundary_id = self._new_identifier("boundary")
        started_at = self._capture_time("boundary")
        baselines = tuple(self._begin_source(source) for source in self._sources)
        boundary = self._build_boundary(boundary_id, started_at, baselines)
        self._store.persist_boundary(boundary)
        startup_drain_id = self._new_identifier("drain")
        try:
            startup_receipt = self._persist_drain(
                boundary,
                drain_index=0,
                checkpoint_kind=DeviceLogCheckpointKind.SESSION_START,
                checkpoint_id=None,
                started_at=started_at,
                source_drains=tuple(
                    self._artifact_producer.persist_batch(
                        boundary,
                        startup_drain_id,
                        0,
                        source,
                        baseline.batch,
                    )
                    if baseline.boundary_source.state is DeviceLogBoundaryState.READY
                    else self._artifact_producer.unavailable_source(
                        source,
                        baseline.boundary_source.reason,
                        infrastructure_failure=(
                            baseline.boundary_source.state is DeviceLogBoundaryState.FAILED
                        ),
                    )
                    for source, baseline in zip(self._sources, baselines, strict=True)
                ),
                drain_id=startup_drain_id,
            )
        except BaseException as error:
            try:
                self._store.discard_boundary(boundary)
            except DeviceLogCaptureError as rollback_error:
                error.add_note(f"Boundary rollback failed: {rollback_error}")
            raise
        self._active_boundary = boundary
        self._next_drain_index = 1
        return DeviceLogCaptureStart(boundary, startup_receipt)

    def drain(
        self,
        boundary: DeviceLogBoundary,
        checkpoint_kind: DeviceLogCheckpointKind,
        checkpoint_id: str | None,
    ) -> DeviceLogDrainReceipt:
        """Persist entries produced since the previous acknowledged read."""
        self._require_active(boundary)
        try:
            validated_checkpoint_id = validate_device_log_checkpoint(
                checkpoint_kind,
                checkpoint_id,
            )
        except (ValidationError, ValueError) as error:
            raise DeviceLogCaptureError("The device-log checkpoint is invalid.") from error
        if checkpoint_kind is DeviceLogCheckpointKind.SESSION_START:
            raise DeviceLogCaptureError("The caller cannot request a session-start log drain.")
        checkpoint_key = (
            checkpoint_kind,
            validated_checkpoint_id or checkpoint_kind.value,
        )
        if checkpoint_key in self._completed_checkpoints:
            raise DeviceLogCaptureError("The device-log checkpoint was already drained.")
        drain_id = self._new_identifier("drain")
        started_at = self._capture_time("drain start")
        source_drains = tuple(
            self._drain_source(
                boundary,
                drain_id,
                self._next_drain_index,
                checkpoint_kind,
                source,
            )
            for source in self._sources
        )
        receipt = self._persist_drain(
            boundary,
            drain_index=self._next_drain_index,
            checkpoint_kind=checkpoint_kind,
            checkpoint_id=validated_checkpoint_id,
            started_at=started_at,
            source_drains=source_drains,
            drain_id=drain_id,
        )
        self._completed_checkpoints.add(checkpoint_key)
        self._next_drain_index += 1
        return receipt

    def finish(self, boundary: DeviceLogBoundary) -> DeviceLogDrainReceipt:
        """Perform the final drain and close the active boundary."""
        try:
            return self.drain(boundary, DeviceLogCheckpointKind.SESSION_END, None)
        finally:
            if self._active_boundary == boundary:
                self._active_boundary = None

    def _begin_source(self, source: DeviceLogSource) -> _SourceBaseline:
        try:
            batch = self._reader.begin(source.log_type)
        except DeviceLogCaptureError:
            return _SourceBaseline(
                DeviceLogBoundarySource(
                    source=source,
                    state=DeviceLogBoundaryState.FAILED,
                    baseline_item_count=0,
                    may_have_omitted_earlier_items=False,
                    reason="The initial device-log read failed.",
                ),
                None,
            )
        if batch is None:
            return _SourceBaseline(
                DeviceLogBoundarySource(
                    source=source,
                    state=DeviceLogBoundaryState.UNSUPPORTED,
                    baseline_item_count=0,
                    may_have_omitted_earlier_items=False,
                    reason="The configured reader does not support this source.",
                ),
                None,
            )
        return _SourceBaseline(
            DeviceLogBoundarySource(
                source=source,
                state=DeviceLogBoundaryState.READY,
                baseline_item_count=batch.observed_item_count,
                may_have_omitted_earlier_items=batch.may_have_omitted_earlier_items,
            ),
            batch,
        )

    def _drain_source(
        self,
        boundary: DeviceLogBoundary,
        drain_id: str,
        drain_index: int,
        checkpoint_kind: DeviceLogCheckpointKind,
        source: DeviceLogSource,
    ) -> DeviceLogBatchPersistence:
        boundary_source = next((item for item in boundary.sources if item.source == source), None)
        if boundary_source is None:
            raise DeviceLogCaptureError("The boundary does not contain a configured source.")
        if boundary_source.state is not DeviceLogBoundaryState.READY:
            return self._artifact_producer.unavailable_source(
                source,
                boundary_source.reason,
                infrastructure_failure=(boundary_source.state is DeviceLogBoundaryState.FAILED),
            )
        if not self._reader.captures_checkpoint(source.log_type, checkpoint_kind):
            return DeviceLogBatchPersistence(
                DeviceLogSourceReceipt(
                    source=source,
                    state=DeviceLogDrainState.NOT_SCHEDULED,
                ),
                None,
            )
        try:
            batch = self._reader.read(source.log_type)
        except DeviceLogCaptureError:
            return DeviceLogBatchPersistence(
                DeviceLogSourceReceipt(
                    source=source,
                    state=DeviceLogDrainState.FAILED,
                    reason="The incremental device-log read failed.",
                    diagnostic_classification=(
                        ApplicationDiagnosticClassification.YEWO_INFRASTRUCTURE_FAILURE
                        if source.is_diagnostic
                        else None
                    ),
                ),
                None,
            )
        return self._artifact_producer.persist_batch(
            boundary,
            drain_id,
            drain_index,
            source,
            batch,
        )

    def _persist_drain(
        self,
        boundary: DeviceLogBoundary,
        *,
        drain_index: int,
        checkpoint_kind: DeviceLogCheckpointKind,
        checkpoint_id: str | None,
        started_at: datetime,
        source_drains: tuple[DeviceLogBatchPersistence, ...],
        drain_id: str,
    ) -> DeviceLogDrainReceipt:
        artifacts = tuple(
            artifact
            for source_drain in source_drains
            for artifact in source_drain.receipt.artifacts
        )
        try:
            receipt = DeviceLogDrainReceipt(
                drain_id=drain_id,
                boundary_id=boundary.boundary_id,
                drain_index=drain_index,
                checkpoint_kind=checkpoint_kind,
                checkpoint_id=checkpoint_id,
                started_at=started_at,
                completed_at=self._capture_time("drain completion"),
                sources=tuple(item.receipt for item in source_drains),
            )
            self._store.persist_receipt(receipt)
        except (DeviceLogCaptureError, ValidationError):
            self._artifact_producer.discard_artifacts(artifacts)
            raise
        for source, source_drain in zip(self._sources, source_drains, strict=True):
            if source_drain.batch is not None:
                self._reader.acknowledge(source.log_type, source_drain.batch)
        return receipt

    def _build_boundary(
        self,
        boundary_id: str,
        started_at: datetime,
        baselines: tuple[_SourceBaseline, ...],
    ) -> DeviceLogBoundary:
        try:
            return DeviceLogBoundary(
                boundary_id=boundary_id,
                device=EvidenceDeviceIdentity(
                    target_id=self._device.target_id,
                    platform=self._device.platform,
                    kind=self._device.kind,
                    operating_system_version=self._device.operating_system_version,
                    model=self._device.device_model,
                    architecture=self._device.architecture,
                ),
                application=EvidenceApplicationIdentity(
                    application_id=self._application.application_id,
                    platform=self._application.platform,
                    version=self._application.version,
                    build=self._application.build,
                    artifact_sha256=self._application.artifact_sha256,
                ),
                started_at=started_at,
                sources=tuple(item.boundary_source for item in baselines),
            )
        except ValidationError as error:
            raise DeviceLogCaptureError("The device-log boundary is invalid.") from error

    def _new_identifier(self, record_kind: str) -> str:
        try:
            return validate_record_identifier(self._identifier_factory())
        except ValidationError as error:
            raise DeviceLogCaptureError(
                f"The device-log {record_kind} identifier is invalid."
            ) from error

    def _capture_time(self, record_kind: str) -> datetime:
        captured_at = self._time_factory()
        if captured_at.utcoffset() is None:
            raise DeviceLogCaptureError(
                f"The device-log {record_kind} time must include a timezone."
            )
        return captured_at

    def _require_active(self, boundary: DeviceLogBoundary) -> None:
        if self._active_boundary != boundary:
            raise DeviceLogCaptureError("The device-log boundary is not active.")
