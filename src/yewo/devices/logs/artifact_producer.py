"""Produce bounded device-log artifacts and source receipts."""

from __future__ import annotations

from dataclasses import dataclass

from yewo.devices.logs.capture import DeviceLogCapturePolicy
from yewo.devices.logs.errors import DeviceLogCaptureError, DeviceLogRollbackError
from yewo.devices.logs.formatting import (
    bound_text_entries,
    combine_omission,
    redaction_policy_id,
    utf8_prefix,
)
from yewo.devices.logs.reader import (
    DeviceCrashReport,
    DeviceLogBatch,
    DeviceLogTextEntry,
)
from yewo.devices.logs.records import (
    DeviceLogArtifactKind,
    DeviceLogArtifactReference,
    DeviceLogBoundary,
    DeviceLogDrainState,
    DeviceLogOmission,
    DeviceLogSource,
    DeviceLogSourceReceipt,
)
from yewo.devices.logs.redaction import DeviceLogRedactionResult, DeviceLogTextScrubber
from yewo.devices.logs.store import DeviceLogStore


@dataclass(frozen=True)
class DeviceLogBatchPersistence:
    """One source receipt and the batch eligible for acknowledgement."""

    receipt: DeviceLogSourceReceipt
    batch: DeviceLogBatch | None


class DeviceLogArtifactProducer:
    """Redact, bound, persist, and roll back device-log artifacts."""

    def __init__(
        self,
        store: DeviceLogStore,
        scrub_text: DeviceLogTextScrubber,
        policy: DeviceLogCapturePolicy,
    ) -> None:
        self._store = store
        self._scrub_text = scrub_text
        self._policy = policy

    def persist_batch(
        self,
        boundary: DeviceLogBoundary,
        drain_id: str,
        drain_index: int,
        source: DeviceLogSource,
        batch: DeviceLogBatch | None,
    ) -> DeviceLogBatchPersistence:
        """Persist one source batch and return its durable-receipt input."""
        if batch is None:
            return self.unavailable_source(
                source,
                "The source baseline is unavailable.",
            )
        if not batch.items:
            return self._empty_batch(source, batch)
        try:
            artifacts, retained_count, omission = self._persist_items(
                boundary,
                drain_id,
                drain_index,
                source,
                batch,
            )
        except DeviceLogRollbackError:
            raise
        except Exception:  # noqa: BLE001 - injected redactor and artifact boundary
            return DeviceLogBatchPersistence(
                DeviceLogSourceReceipt(
                    source=source,
                    state=DeviceLogDrainState.FAILED,
                    observed_item_count=batch.observed_item_count,
                    filtered_item_count=batch.filtered_item_count,
                    reason="The device-log items could not be redacted and persisted.",
                ),
                None,
            )
        state = (
            DeviceLogDrainState.CAPTURED
            if artifacts
            else DeviceLogDrainState.NO_RETAINED_ITEMS
        )
        return DeviceLogBatchPersistence(
            DeviceLogSourceReceipt(
                source=source,
                state=state,
                observed_item_count=batch.observed_item_count,
                filtered_item_count=batch.filtered_item_count,
                retained_item_count=retained_count,
                omission=omission,
                artifacts=artifacts,
            ),
            batch,
        )

    def unavailable_source(
        self,
        source: DeviceLogSource,
        reason: str | None,
    ) -> DeviceLogBatchPersistence:
        """Return an unavailable-source receipt without an acknowledged batch."""
        return DeviceLogBatchPersistence(
            DeviceLogSourceReceipt(
                source=source,
                state=DeviceLogDrainState.UNAVAILABLE,
                reason=reason or "The device-log source is unavailable.",
            ),
            None,
        )

    def discard_artifacts(
        self,
        artifacts: tuple[DeviceLogArtifactReference, ...],
    ) -> None:
        """Roll back artifacts and report any evidence that remains."""
        errors: list[DeviceLogCaptureError] = []
        for artifact in artifacts:
            try:
                self._store.discard(artifact)
            except DeviceLogCaptureError as error:
                errors.append(error)
        if errors:
            failure = DeviceLogRollbackError(
                f"Could not roll back {len(errors)} incomplete device-log artifact(s)."
            )
            failure.add_note("; ".join(str(error) for error in errors))
            raise failure from errors[0]

    def _empty_batch(
        self,
        source: DeviceLogSource,
        batch: DeviceLogBatch,
    ) -> DeviceLogBatchPersistence:
        omission = (
            DeviceLogOmission.EARLIER
            if batch.may_have_omitted_earlier_items
            else DeviceLogOmission.NONE
        )
        return DeviceLogBatchPersistence(
            DeviceLogSourceReceipt(
                source=source,
                state=DeviceLogDrainState.NO_RETAINED_ITEMS,
                observed_item_count=batch.observed_item_count,
                filtered_item_count=batch.filtered_item_count,
                omission=omission,
            ),
            batch,
        )

    def _persist_items(
        self,
        boundary: DeviceLogBoundary,
        drain_id: str,
        drain_index: int,
        source: DeviceLogSource,
        batch: DeviceLogBatch,
    ) -> tuple[tuple[DeviceLogArtifactReference, ...], int, DeviceLogOmission]:
        if all(isinstance(item, DeviceLogTextEntry) for item in batch.items):
            return self._persist_text_entries(
                boundary,
                drain_id,
                drain_index,
                source,
                batch,
            )
        if all(isinstance(item, DeviceCrashReport) for item in batch.items):
            return self._persist_crash_reports(
                boundary,
                drain_id,
                drain_index,
                source,
                batch,
            )
        raise DeviceLogCaptureError("A device-log batch mixed incompatible items.")

    def _persist_text_entries(
        self,
        boundary: DeviceLogBoundary,
        drain_id: str,
        drain_index: int,
        source: DeviceLogSource,
        batch: DeviceLogBatch,
    ) -> tuple[tuple[DeviceLogArtifactReference, ...], int, DeviceLogOmission]:
        entries = tuple(item for item in batch.items if isinstance(item, DeviceLogTextEntry))
        redacted = tuple(self._redact(item.text) for item in entries)
        policy_id = redaction_policy_id(redacted)
        content, retained_count, omission = bound_text_entries(
            redacted,
            self._policy.maximum_drain_bytes,
        )
        if batch.may_have_omitted_earlier_items:
            omission = combine_omission(omission, DeviceLogOmission.EARLIER)
        if not content or not retained_count:
            return (), 0, omission
        artifact = self._store.persist_artifact(
            content,
            boundary_id=boundary.boundary_id,
            drain_id=drain_id,
            drain_index=drain_index,
            source=source,
            entry_count=retained_count,
            kind=DeviceLogArtifactKind.TEXT_STREAM,
            source_name=None,
            redaction_policy_id=policy_id,
            source_redaction_count=sum(
                item.redaction_count for item in redacted[-retained_count:]
            ),
            sensitivity=self._policy.sensitivity,
            retention=self._policy.retention,
        )
        return (artifact,), retained_count, omission

    def _persist_crash_reports(
        self,
        boundary: DeviceLogBoundary,
        drain_id: str,
        drain_index: int,
        source: DeviceLogSource,
        batch: DeviceLogBatch,
    ) -> tuple[tuple[DeviceLogArtifactReference, ...], int, DeviceLogOmission]:
        reports = tuple(item for item in batch.items if isinstance(item, DeviceCrashReport))
        artifacts: list[DeviceLogArtifactReference] = []
        retained_bytes = 0
        omission = DeviceLogOmission.NONE
        try:
            for report in reversed(reports):
                redacted = self._redact(report.text)
                content = redacted.text.encode("utf-8", errors="backslashreplace")
                remaining_bytes = self._policy.maximum_drain_bytes - retained_bytes
                if remaining_bytes <= 0:
                    omission = combine_omission(omission, DeviceLogOmission.EARLIER)
                    break
                if len(content) > remaining_bytes:
                    if artifacts:
                        omission = combine_omission(omission, DeviceLogOmission.EARLIER)
                        break
                    content = utf8_prefix(content, remaining_bytes)
                    omission = combine_omission(omission, DeviceLogOmission.LATER)
                if not content:
                    break
                artifacts.append(
                    self._store.persist_artifact(
                        content,
                        boundary_id=boundary.boundary_id,
                        drain_id=drain_id,
                        drain_index=drain_index,
                        source=source,
                        entry_count=1,
                        kind=DeviceLogArtifactKind.CRASH_REPORT,
                        source_name=report.source_name,
                        redaction_policy_id=redacted.policy_id,
                        source_redaction_count=redacted.redaction_count,
                        sensitivity=self._policy.sensitivity,
                        retention=self._policy.retention,
                    )
                )
                retained_bytes += len(content)
        except Exception:
            self.discard_artifacts(tuple(artifacts))
            raise
        artifacts.reverse()
        if len(artifacts) < len(reports):
            omission = combine_omission(omission, DeviceLogOmission.EARLIER)
        return tuple(artifacts), len(artifacts), omission

    def _redact(self, text: str) -> DeviceLogRedactionResult:
        result = self._scrub_text(text)
        if not isinstance(result, DeviceLogRedactionResult):
            raise DeviceLogCaptureError("The device-log scrubber returned an invalid result.")
        return result
