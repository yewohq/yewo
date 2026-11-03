"""Persistence for run evidence and compatible compiled procedures."""

from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Generator, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.replay.reconciliation.records import EvidenceDocument, ReplayProcedure
from yewo.agent.memory.records import MemoryCompatibility
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AtomicArtifactWriter,
    Sha256Digest,
)


class CompiledProcedureStoreError(RuntimeError):
    """Compatible procedure evidence could not be read or persisted safely."""


class StoredProcedureBundle(BaseModel):
    """One compiler-approved procedure and its complete source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    compatibility: MemoryCompatibility
    evidence: EvidenceDocument
    procedure: ReplayProcedure

    @model_validator(mode="after")
    def _validate_links(self) -> Self:
        if self.compatibility.application != self.procedure.application:
            raise ValueError("Stored procedure compatibility does not match the application.")
        if self.evidence.evidence_id != self.procedure.evidence_id:
            raise ValueError("Stored procedure evidence does not match the procedure.")
        return self


class _StoredProcedureEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    bundle_sha256: Sha256Digest
    bundle: StoredProcedureBundle


class ProcedureReplayStore:
    """Persist one procedure attempt and its optional divergence."""

    def __init__(self, root: Path) -> None:
        self._writer = AtomicArtifactWriter(root)

    def persist(self, result: ProcedureReplayResult) -> tuple[Path, ...]:
        """Write the complete result before returning it to the workflow."""
        directory = ARTIFACTS_DIRECTORY / "agent" / "procedure-replay"
        paths = [directory / "attempt.json"]
        self._writer.write(paths[0], result.model_dump_json(indent=2).encode("utf-8"))
        if result.divergence is not None:
            paths.append(directory / "divergence.json")
            self._writer.write(
                paths[-1],
                result.divergence.model_dump_json(indent=2).encode("utf-8"),
            )
        if result.divergences:
            paths.append(directory / "divergences.json")
            content = (
                b"[\n"
                + b",\n".join(
                    item.model_dump_json(indent=2).encode("utf-8") for item in result.divergences
                )
                + b"\n]\n"
            )
            self._writer.write(paths[-1], content)
        return tuple(paths)


class CompiledProcedureStore:
    """Store one current procedure per compatible objective."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(
        self,
        compatibility: MemoryCompatibility,
        objective_id: str,
        objective: str,
    ) -> StoredProcedureBundle | None:
        """Load an exact compatible procedure, or return none when absent."""
        procedure_root = self._procedure_root(compatibility, objective_id, objective)
        with self._locked(procedure_root):
            path = procedure_root / "procedure.json"
            if not path.exists():
                return None
            bundle = _read_bundle(path)
            if (
                bundle.compatibility != compatibility
                or bundle.evidence.task.objective_id != objective_id
                or bundle.evidence.task.objective != objective
            ):
                raise CompiledProcedureStoreError(
                    "The compiled procedure snapshot is incompatible."
                )
            return bundle

    def compatible_candidates(
        self,
        compatibility: MemoryCompatibility,
    ) -> tuple[StoredProcedureBundle, ...]:
        """Load every valid procedure for one exact application build."""
        compatibility_root = self.root / "procedures" / _compatibility_key(compatibility)
        if not compatibility_root.exists():
            return ()
        if compatibility_root.is_symlink() or not compatibility_root.is_dir():
            raise CompiledProcedureStoreError(
                "The compatible procedure collection must be a regular directory."
            )
        bundles: list[StoredProcedureBundle] = []
        try:
            objective_roots = tuple(sorted(compatibility_root.iterdir()))
        except OSError as error:
            raise CompiledProcedureStoreError(
                "Could not inspect compatible stored procedures."
            ) from error
        for objective_root in objective_roots:
            if objective_root.is_symlink() or not objective_root.is_dir():
                raise CompiledProcedureStoreError(
                    "A stored procedure objective must be a regular directory."
                )
            with self._locked(objective_root):
                path = objective_root / "procedure.json"
                if not path.exists():
                    continue
                bundle = _read_bundle(path)
                if bundle.compatibility != compatibility:
                    raise CompiledProcedureStoreError(
                        "A stored procedure candidate is incompatible."
                    )
                bundles.append(bundle)
        return tuple(bundles)

    def save(
        self,
        compatibility: MemoryCompatibility,
        evidence: EvidenceDocument,
        procedure: ReplayProcedure,
    ) -> StoredProcedureBundle:
        """Atomically replace the current compatible procedure."""
        bundle = StoredProcedureBundle(
            compatibility=compatibility,
            evidence=evidence,
            procedure=procedure,
        )
        procedure_root = self._procedure_root(
            compatibility,
            evidence.task.objective_id,
            evidence.task.objective,
        )
        with self._locked(procedure_root):
            _write_bundle(procedure_root, bundle)
        return bundle

    def _procedure_root(
        self,
        compatibility: MemoryCompatibility,
        objective_id: str,
        objective: str,
    ) -> Path:
        compatibility_key = _compatibility_key(compatibility)
        objective_key = sha256(
            _canonical_bytes({"objective_id": objective_id, "objective": objective})
        ).hexdigest()[:32]
        return self.root / "procedures" / compatibility_key / objective_key

    @contextmanager
    def _locked(self, procedure_root: Path) -> Generator[None, None, None]:
        descriptor: int | None = None
        try:
            _prepare_procedure_root(self.root, procedure_root)
            lock_path = procedure_root / ".procedure.lock"
            if lock_path.is_symlink():
                raise CompiledProcedureStoreError("The compiled procedure lock cannot be symbolic.")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_path, flags, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                descriptor = None
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                yield
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except CompiledProcedureStoreError:
            raise
        except OSError as error:
            raise CompiledProcedureStoreError(
                "Could not lock the compiled procedure store."
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _write_bundle(root: Path, bundle: StoredProcedureBundle) -> None:
    bundle_payload = bundle.model_dump(mode="json")
    bundle_content = _canonical_bytes(bundle_payload)
    envelope = _StoredProcedureEnvelope(
        bundle_sha256=sha256(bundle_content).hexdigest(),
        bundle=bundle,
    )
    content = _canonical_bytes(envelope.model_dump(mode="json")) + b"\n"
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=".procedure-")
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        destination = root / "procedure.json"
        if destination.is_symlink():
            raise CompiledProcedureStoreError("The compiled procedure snapshot cannot be symbolic.")
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except CompiledProcedureStoreError:
        raise
    except OSError as error:
        raise CompiledProcedureStoreError(
            "Could not persist the compiled procedure snapshot."
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_bundle(path: Path) -> StoredProcedureBundle:
    try:
        if path.is_symlink() or not path.is_file():
            raise CompiledProcedureStoreError("The compiled procedure must be a regular file.")
        payload = json.loads(path.read_bytes())
        envelope = _StoredProcedureEnvelope.model_validate(payload)
        content = _canonical_bytes(payload["bundle"])
    except CompiledProcedureStoreError:
        raise
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValidationError) as error:
        raise CompiledProcedureStoreError("The compiled procedure snapshot is invalid.") from error
    if sha256(content).hexdigest() != envelope.bundle_sha256:
        raise CompiledProcedureStoreError("The compiled procedure snapshot digest does not match.")
    return envelope.bundle


def _compatibility_key(compatibility: MemoryCompatibility) -> str:
    return sha256(_canonical_bytes(compatibility.model_dump(mode="json"))).hexdigest()[:32]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _prepare_procedure_root(store_root: Path, procedure_root: Path) -> None:
    try:
        store_root.mkdir(parents=True, exist_ok=True)
        if store_root.is_symlink() or not store_root.is_dir():
            raise CompiledProcedureStoreError("The compiled procedure root must be a directory.")
        relative = procedure_root.relative_to(store_root)
        current = store_root
        for part in relative.parts:
            current /= part
            if current.exists():
                if current.is_symlink() or not current.is_dir():
                    raise CompiledProcedureStoreError(
                        "The compiled procedure path must contain only directories."
                    )
            else:
                current.mkdir()
        procedure_root.resolve().relative_to(store_root.resolve())
    except CompiledProcedureStoreError:
        raise
    except (OSError, ValueError) as error:
        raise CompiledProcedureStoreError(
            "Could not prepare the compiled procedure directory."
        ) from error
