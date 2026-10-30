"""ADB removal for paths that must survive an Appium failure."""

from yewo.devices.android._file_commands import AdbFileCommands
from yewo.devices.remote_files.errors import RemoteTemporaryFileError


class AdbRemoteFileRemover:
    """Remove run-owned Android files without the Appium connection."""

    def __init__(self, files: AdbFileCommands, connection_id: str) -> None:
        self._files = files
        self._connection_id = connection_id

    def remove_file(self, remote_path: str) -> None:
        try:
            self._files.remove_file(self._connection_id, remote_path)
        except Exception as error:  # noqa: BLE001 - platform command boundary
            raise RemoteTemporaryFileError(
                "ADB could not remove a run-owned remote temporary file."
            ) from error
