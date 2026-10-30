"""Binary media transfer through ADB."""

from pathlib import Path
from tempfile import TemporaryDirectory

from yewo.devices.android._file_commands import AdbFileCommands
from yewo.devices.media.errors import MediaTransferError


class AdbMediaTransport:
    """Transfer media without using the Appium connection."""

    def __init__(self, files: AdbFileCommands, connection_id: str) -> None:
        self._files = files
        self._connection_id = connection_id

    def push(self, remote_path: str, content: bytes) -> None:
        with TemporaryDirectory(prefix="yewo-media-push-") as directory:
            source = Path(directory) / "media"
            source.write_bytes(content)
            try:
                self._files.push_file(
                    self._connection_id,
                    str(source),
                    remote_path,
                )
            except Exception as error:  # noqa: BLE001 - platform command boundary
                raise MediaTransferError("ADB could not push the media file.") from error

    def pull(self, remote_path: str, maximum_bytes: int) -> bytes:
        with TemporaryDirectory(prefix="yewo-media-pull-") as directory:
            destination = Path(directory) / "media"
            self.pull_to_path(remote_path, destination, maximum_bytes)
            content = destination.read_bytes()
        if not content:
            raise MediaTransferError("ADB returned an empty media file.")
        return content

    def pull_to_path(
        self,
        remote_path: str,
        destination: Path,
        maximum_bytes: int,
        *,
        timeout_seconds: float = 120,
    ) -> None:
        """Pull one bounded remote file without loading it into memory."""
        try:
            self._files.pull_file(
                self._connection_id,
                remote_path,
                str(destination),
                timeout_seconds=timeout_seconds,
            )
            size = destination.stat().st_size
        except Exception as error:  # noqa: BLE001 - platform and filesystem boundary
            raise MediaTransferError("ADB could not pull the media file.") from error
        if size < 1:
            raise MediaTransferError("ADB returned an empty media file.")
        if size > maximum_bytes:
            raise MediaTransferError("ADB media exceeds the configured byte limit.")
