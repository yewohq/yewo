"""Atomic persistence for reusable application memory."""

from __future__ import annotations

from contextlib import contextmanager, suppress
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
    AppMemorySchemaVersion,
    Sha256Digest,
)
from yewo.host import exclusive_file_lock, replace_file_durably
from yewo.run_narration import current_narrator

from yewo.agent.memory.records import (
    AppMemory,
    MemoryCompatibility,
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

    def save(self, memory: AppMemory) -> AppMemory:
        """Atomically replace current memory.

        Nothing is copied out of the run. A variant carries the fingerprint and presentation
        recognition compares, so the screen it describes needs no retained image.
        """
        memory, _removed_attempt_count = _without_target_rejections(memory)
        application_root = self._application_root(memory.compatibility)
        with self._locked(application_root):
            current = self._replaceable_current(application_root, memory.compatibility)
            if current is not None and not set(current.source_run_ids).issubset(
                memory.source_run_ids
            ):
                raise AppMemoryStoreError(
                    "Refused to replace app memory built from stale source runs."
                )
            _write_snapshot(application_root, memory)
        _narrate_memory_save(memory)
        return memory

    def _replaceable_current(
        self,
        application_root: Path,
        compatibility: MemoryCompatibility,
    ) -> AppMemory | None:
        """Read the snapshot this save replaces, treating an unreadable one as absent.

        The stale-source-run guard needs the current snapshot. An obsolete schema cannot be read,
        but discovery can rebuild this derived memory. Treating that snapshot as present would
        prevent the rebuilt current-schema replacement from ever reaching disk.
        """
        try:
            current, _legacy_attempt_count = self._load_unlocked(application_root, compatibility)
        except AppMemorySchemaError:
            _narrate_replaced_memory()
            return None
        return current

    def update(
        self,
        compatibility: MemoryCompatibility,
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
            _write_snapshot(application_root, updated)
        _narrate_memory_save(updated)
        return updated

    def _application_root(self, compatibility: MemoryCompatibility) -> Path:
        identity = _canonical_bytes(compatibility.model_dump(mode="json"))
        return self.root / "applications" / sha256(identity).hexdigest()[:32]

    @contextmanager
    def _locked(self, application_root: Path) -> Generator[None, None, None]:
        try:
            _prepare_root(self.root, application_root)
            lock_path = application_root / ".memory.lock"
            if lock_path.is_symlink():
                raise AppMemoryStoreError("The app-memory lock cannot be symbolic.")
            with exclusive_file_lock(lock_path):
                yield
        except AppMemoryStoreError:
            raise
        except OSError as error:
            raise AppMemoryStoreError("Could not lock app memory.") from error

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
        return _without_target_rejections(envelope.memory)


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


def _without_target_rejections(memory: AppMemory) -> tuple[AppMemory, int]:
    """Drop attempts a target rejection produced, which app memory must not learn from.

    A target rejection is Yewo grounding onto the wrong pixel. The application did nothing, so
    remembering it would teach a lesson about this agent and attribute it to the app.
    """
    retained_attempts = tuple(
        attempt
        for attempt in memory.unsuccessful_attempts
        if attempt.disposition is not UnsuccessfulAttemptDisposition.TARGET_REJECTED
    )
    removed_count = len(memory.unsuccessful_attempts) - len(retained_attempts)
    return memory.model_copy(update={"unsuccessful_attempts": retained_attempts}), removed_count


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
        replace_file_durably(temporary_path, destination)
        temporary_path = None
    except AppMemoryStoreError:
        raise
    except OSError as error:
        raise AppMemoryStoreError("Could not persist app memory.") from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


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


def _narrate_replaced_memory() -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        "agent-memory",
        "Replacing an app-memory snapshot written by an unsupported schema.",
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
