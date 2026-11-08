"""Atomic persistence for reusable application memory."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Generator
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from yewo.artifacts import (
    APP_MEMORY_SCHEMA_VERSION,
    ArtifactRetention,
    AtomicArtifactWriter,
    ArtifactWriteError,
    AppMemorySchemaVersion,
    RecordIdentifier,
    Sha256Digest,
)
from yewo.devices.screen import png_size
from yewo.observations import ScreenshotReference
from yewo.run_narration import current_narrator

from yewo.agent.memory.records import (
    AppMemory,
    MemoryCompatibility,
    ObservationVariant,
    UnsuccessfulAttemptDisposition,
)


class AppMemoryStoreError(RuntimeError):
    """Reusable app memory could not be read or persisted safely."""


class AppMemorySchemaError(AppMemoryStoreError):
    """An app-memory snapshot uses an unsupported persisted schema."""


class _MemoryEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: AppMemorySchemaVersion = APP_MEMORY_SCHEMA_VERSION
    memory_sha256: Sha256Digest
    memory: AppMemory


class AppMemoryStore:
    """Store one atomic current snapshot per compatible application build."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, compatibility: MemoryCompatibility) -> AppMemory | None:
        """Load and verify compatible memory, or return none when absent."""
        application_root = self._application_root(compatibility)
        with self._locked(application_root):
            memory, legacy_attempt_count = self._load_unlocked(application_root, compatibility)
        _narrate_memory_load(memory, legacy_attempt_count)
        return memory

    def save(self, memory: AppMemory, source_run_root: Path) -> AppMemory:
        """Promote redacted screenshots and atomically replace current memory."""
        memory, _removed_attempt_count = _without_legacy_target_rejections(memory)
        application_root = self._application_root(memory.compatibility)
        with self._locked(application_root):
            current, _legacy_attempt_count = self._load_unlocked(
                application_root, memory.compatibility
            )
            if current is not None and not set(current.source_run_ids).issubset(
                memory.source_run_ids
            ):
                raise AppMemoryStoreError(
                    "Refused to replace app memory built from stale source runs."
                )
            promoted = _promote_screenshots(memory, source_run_root, application_root)
            _write_snapshot(application_root, promoted)
            _remove_unreferenced_screenshots(application_root, promoted)
        _narrate_memory_save(promoted)
        return promoted

    def update(
        self,
        compatibility: MemoryCompatibility,
        source_run_root: Path,
        updater: Callable[[AppMemory], AppMemory],
    ) -> AppMemory:
        """Apply one update while holding the complete load-to-save lock."""
        application_root = self._application_root(compatibility)
        with self._locked(application_root):
            current, _legacy_attempt_count = self._load_unlocked(application_root, compatibility)
            if current is None:
                raise AppMemoryStoreError("App memory is unavailable for write-back.")
            updated = updater(current)
            if updated.compatibility != compatibility:
                raise AppMemoryStoreError("Write-back changed app-memory compatibility.")
            promoted = _promote_screenshots(updated, source_run_root, application_root)
            _write_snapshot(application_root, promoted)
            _remove_unreferenced_screenshots(application_root, promoted)
        _narrate_memory_save(promoted)
        return promoted

    def read_observation_screenshot(
        self,
        memory: AppMemory,
        observation_id: RecordIdentifier,
    ) -> tuple[ScreenshotReference, bytes]:
        """Return one verified retained redacted screenshot by observation ID."""
        variant = next(
            (
                variant
                for screen in memory.screens
                for variant in screen.variants
                if variant.observation_id == observation_id
            ),
            None,
        )
        if variant is None:
            raise AppMemoryStoreError("The requested memory observation is unavailable.")
        application_root = self._application_root(memory.compatibility)
        _verify_screenshot(application_root, variant.screenshot)
        try:
            return variant.screenshot, (application_root / variant.screenshot.path).read_bytes()
        except OSError as error:
            raise AppMemoryStoreError("Could not read the memory screenshot.") from error

    def _application_root(self, compatibility: MemoryCompatibility) -> Path:
        identity = _canonical_bytes(compatibility.model_dump(mode="json"))
        return self.root / "applications" / sha256(identity).hexdigest()[:32]

    @contextmanager
    def _locked(self, application_root: Path) -> Generator[None, None, None]:
        descriptor: int | None = None
        try:
            _prepare_root(self.root, application_root)
            lock_path = application_root / ".memory.lock"
            if lock_path.is_symlink():
                raise AppMemoryStoreError("The app-memory lock cannot be symbolic.")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_path, flags, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                descriptor = None
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                yield
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except AppMemoryStoreError:
            raise
        except OSError as error:
            raise AppMemoryStoreError("Could not lock app memory.") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _load_unlocked(
        application_root: Path,
        compatibility: MemoryCompatibility,
    ) -> tuple[AppMemory | None, int]:
        path = application_root / "memory.json"
        if not path.exists():
            return None, 0
        try:
            if path.is_symlink() or not path.is_file():
                raise AppMemoryStoreError("The app-memory snapshot must be a regular file.")
            payload = json.loads(path.read_bytes())
            _validate_supported_schema(payload)
            envelope = _MemoryEnvelope.model_validate(payload)
        except AppMemoryStoreError:
            raise
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValidationError) as error:
            raise AppMemoryStoreError("The app-memory snapshot is invalid.") from error
        content = _canonical_bytes(payload["memory"])
        if sha256(content).hexdigest() != envelope.memory_sha256:
            raise AppMemoryStoreError("The app-memory snapshot digest does not match.")
        if envelope.memory.compatibility != compatibility:
            raise AppMemoryStoreError("The app-memory snapshot is incompatible.")
        _verify_persistent_screenshots(application_root, envelope.memory)
        return _without_legacy_target_rejections(envelope.memory)


def _validate_supported_schema(payload: object) -> None:
    if not isinstance(payload, dict):
        raise AppMemoryStoreError("The app-memory snapshot is invalid.")
    if payload.get("schema_version") != APP_MEMORY_SCHEMA_VERSION:
        raise AppMemorySchemaError(
            "The app-memory schema is unsupported. Rebuild memory with this Yewo version."
        )


def load_discovery_memory(
    store: AppMemoryStore,
    compatibility: MemoryCompatibility,
) -> AppMemory | None:
    """Load supported memory, or ignore a snapshot that requires rebuilding."""
    try:
        return store.load(compatibility)
    except AppMemorySchemaError:
        _narrate_unsupported_memory()
        return None


def _without_legacy_target_rejections(memory: AppMemory) -> tuple[AppMemory, int]:
    retained_attempts = tuple(
        attempt
        for attempt in memory.unsuccessful_attempts
        if attempt.disposition is not UnsuccessfulAttemptDisposition.TARGET_REJECTED
    )
    removed_count = len(memory.unsuccessful_attempts) - len(retained_attempts)
    return memory.model_copy(update={"unsuccessful_attempts": retained_attempts}), removed_count


def _promote_screenshots(
    memory: AppMemory,
    source_run_root: Path,
    application_root: Path,
) -> AppMemory:
    screens = tuple(
        screen.model_copy(
            update={
                "variants": tuple(
                    _promote_variant(variant, source_run_root, application_root)
                    for variant in screen.variants
                )
            }
        )
        for screen in memory.screens
    )
    return memory.model_copy(update={"screens": screens})


def _promote_variant(
    variant: ObservationVariant,
    source_run_root: Path,
    application_root: Path,
) -> ObservationVariant:
    screenshot = variant.screenshot
    if screenshot.retention is ArtifactRetention.PERSISTENT:
        _verify_screenshot(application_root, screenshot)
        return variant
    source_path = source_run_root / screenshot.path
    _verify_source_path(source_run_root, source_path, screenshot)
    relative_path = Path("screenshots") / f"{screenshot.sha256}.png"
    destination = application_root / relative_path
    persistent = screenshot.model_copy(
        update={"path": relative_path, "retention": ArtifactRetention.PERSISTENT}
    )
    if destination.exists():
        _verify_screenshot(application_root, persistent)
    else:
        try:
            AtomicArtifactWriter(application_root).write_file(relative_path, source_path)
        except ArtifactWriteError as error:
            raise AppMemoryStoreError(str(error)) from error
    return variant.model_copy(update={"screenshot": persistent})


def _verify_source_path(
    source_root: Path,
    source_path: Path,
    screenshot: ScreenshotReference,
) -> None:
    try:
        source_path.resolve().relative_to(source_root.resolve())
        if source_path.is_symlink() or not source_path.is_file():
            raise AppMemoryStoreError("A memory screenshot source must be a regular file.")
    except AppMemoryStoreError:
        raise
    except (OSError, ValueError) as error:
        raise AppMemoryStoreError("A memory screenshot source is unavailable.") from error
    _verify_screenshot(source_root, screenshot)


def _verify_persistent_screenshots(root: Path, memory: AppMemory) -> None:
    for screen in memory.screens:
        for variant in screen.variants:
            if variant.screenshot.retention is not ArtifactRetention.PERSISTENT:
                raise AppMemoryStoreError("Persisted app memory contains a run-scoped screenshot.")
            _verify_screenshot(root, variant.screenshot)


def _verify_screenshot(root: Path, screenshot: ScreenshotReference) -> None:
    path = root / screenshot.path
    try:
        path.resolve().relative_to(root.resolve())
        if path.is_symlink() or not path.is_file():
            raise AppMemoryStoreError("A memory screenshot must be a regular file.")
        content = path.read_bytes()
    except AppMemoryStoreError:
        raise
    except (OSError, ValueError) as error:
        raise AppMemoryStoreError("A memory screenshot is unavailable.") from error
    if sha256(content).hexdigest() != screenshot.sha256:
        raise AppMemoryStoreError("A memory screenshot digest does not match.")
    if png_size(content) != (screenshot.width_px, screenshot.height_px):
        raise AppMemoryStoreError("A memory screenshot has invalid dimensions.")


def _write_snapshot(application_root: Path, memory: AppMemory) -> None:
    memory_content = _canonical_bytes(memory.model_dump(mode="json"))
    envelope = _MemoryEnvelope(
        memory_sha256=sha256(memory_content).hexdigest(),
        memory=memory,
    )
    content = _canonical_bytes(envelope.model_dump(mode="json")) + b"\n"
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=application_root,
            prefix=".memory-",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        destination = application_root / "memory.json"
        if destination.is_symlink():
            raise AppMemoryStoreError("The app-memory snapshot cannot be symbolic.")
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_descriptor = os.open(application_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except AppMemoryStoreError:
        raise
    except OSError as error:
        raise AppMemoryStoreError("Could not persist app memory.") from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _remove_unreferenced_screenshots(
    application_root: Path,
    memory: AppMemory,
) -> None:
    screenshots_root = application_root / "screenshots"
    if not screenshots_root.exists():
        return
    try:
        if screenshots_root.is_symlink() or not screenshots_root.is_dir():
            raise AppMemoryStoreError("The memory screenshot directory must be regular.")
        screenshots_root.resolve().relative_to(application_root.resolve())
        referenced = {
            (application_root / variant.screenshot.path).resolve()
            for screen in memory.screens
            for variant in screen.variants
        }
        for path in screenshots_root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise AppMemoryStoreError(
                    "The memory screenshot directory contains an unsafe entry."
                )
            resolved = path.resolve()
            resolved.relative_to(screenshots_root.resolve())
            if resolved not in referenced:
                path.unlink()
    except AppMemoryStoreError:
        raise
    except (OSError, ValueError) as error:
        raise AppMemoryStoreError("Could not apply memory screenshot retention.") from error


def _prepare_root(root: Path, application_root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise AppMemoryStoreError("The app-memory root cannot be symbolic.")
        applications = root / "applications"
        if applications.is_symlink():
            raise AppMemoryStoreError("The applications directory cannot be symbolic.")
        applications.mkdir(exist_ok=True)
        if application_root.is_symlink():
            raise AppMemoryStoreError("The application memory directory cannot be symbolic.")
        application_root.mkdir(exist_ok=True)
        application_root.resolve().relative_to(root.resolve())
    except AppMemoryStoreError:
        raise
    except (OSError, ValueError) as error:
        raise AppMemoryStoreError("Could not prepare the app-memory directory.") from error


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AppMemoryStoreError("App memory is not serializable.") from error


def _narrate_memory_load(memory: AppMemory | None, legacy_attempt_count: int = 0) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    if memory is None:
        narrator.record("agent-memory", "No compatible application memory was found.")
        narrator.trace("agent-memory", "load=absent")
        return
    narrator.record(
        "agent-memory",
        f"Loaded application memory with {len(memory.transitions)} transition record(s).",
    )
    if legacy_attempt_count:
        narrator.record(
            "agent-memory",
            f"Invalidated {legacy_attempt_count} legacy target-rejection record(s).",
        )
    narrator.trace(
        "agent-memory",
        f"load=completed memory={memory.memory_id} screens={len(memory.screens)} "
        f"transitions={len(memory.transitions)} "
        f"invalidated_legacy_target_rejections={legacy_attempt_count}",
    )


def _narrate_unsupported_memory() -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        "agent-memory",
        "Ignored incompatible application memory. Rebuild memory with this Yewo version.",
    )
    narrator.trace("agent-memory", "load=unsupported")


def _narrate_memory_save(memory: AppMemory) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        "agent-memory",
        f"Saved application memory with {len(memory.transitions)} transition record(s).",
    )
    narrator.trace(
        "agent-memory",
        f"save=completed memory={memory.memory_id} screens={len(memory.screens)} "
        f"transitions={len(memory.transitions)}",
    )
