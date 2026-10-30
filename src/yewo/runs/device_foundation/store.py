"""Persist deterministic action receipts and one workflow result."""

from pathlib import Path

from yewo.artifacts import AtomicArtifactWriter, ArtifactWriteError
from yewo.runs.device_foundation.records import (
    DeviceFoundationActionReceipt,
    DeviceFoundationResult,
)
from yewo.runs.errors import RunCoordinationError


class DeviceFoundationStore:
    """Persist payload-free workflow evidence beneath one run root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._writer = AtomicArtifactWriter(root)

    def persist_action(self, receipt: DeviceFoundationActionReceipt) -> Path:
        """Verify observation ownership and persist one ordered action receipt."""
        self._require_observation(receipt.before_observation_id)
        self._require_observation(receipt.after_observation_id)
        path = Path("assets/actions") / (f"{receipt.action_index:06d}-{receipt.action_id}.json")
        self._write(path, receipt.model_dump_json(indent=2))
        return path

    def persist_result(self, result: DeviceFoundationResult) -> Path:
        """Persist one immutable summary after all attempted action receipts."""
        self._require_observation(result.initial_observation_id)
        self._require_observation(result.final_observation_id)
        for action_id in (
            *result.completed_action_ids,
            *((result.rejected_action_id,) if result.rejected_action_id else ()),
        ):
            matches = tuple((self.root / "assets/actions").glob(f"*-{action_id}.json"))
            if len(matches) != 1 or matches[0].is_symlink():
                raise RunCoordinationError(
                    f"A device-foundation result has no unique action receipt: {action_id}"
                )
        path = Path("assets/device-foundation/result.json")
        self._write(path, result.model_dump_json(indent=2))
        return path

    def _require_observation(self, observation_id: str) -> None:
        path = self.root / "assets/observations" / f"{observation_id}.json"
        if not path.is_file() or path.is_symlink():
            raise RunCoordinationError(
                f"Device-foundation evidence references a missing observation: {observation_id}"
            )

    def _write(self, path: Path, content: str) -> None:
        try:
            self._writer.write(path, f"{content}\n".encode())
        except ArtifactWriteError as error:
            raise RunCoordinationError(
                f"Could not persist device-foundation evidence: {path.as_posix()}"
            ) from error
