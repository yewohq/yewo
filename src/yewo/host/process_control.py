"""Process startup and teardown that own their children on every host."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import subprocess
import sys
from typing import Protocol


_STOP_TIMEOUT_SECONDS = 5.0
_TASKKILL_TIMEOUT_SECONDS = 5.0


class ManagedProcess(Protocol):
    """One started child whose lifetime a Yewo run owns."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def stop_process_tree(
    process: ManagedProcess,
    *,
    timeout_seconds: float = _STOP_TIMEOUT_SECONDS,
) -> None:
    """Stop one child and every process it started."""
    if process.poll() is not None:
        return
    request_process_tree_stop(process)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


if sys.platform == "win32":

    def start_detached_process(
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start one child in its own process group and without a console."""
        return subprocess.Popen(  # noqa: S603 - the command is an owned argument list
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=None if environment is None else dict(environment),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        )

    def request_process_tree_stop(process: ManagedProcess) -> None:
        """Ask the host to stop one process and its children."""
        # A Windows stop request does not reach the children of a process, so
        # the host tool stops the whole tree first.
        stop_result = subprocess.run(  # noqa: S603 - fixed executable and argument list
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            timeout=_TASKKILL_TIMEOUT_SECONDS,
            check=False,
        )
        if stop_result.returncode:
            process.terminate()

else:

    def start_detached_process(
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start one child in a new session so a terminal signal misses it."""
        return subprocess.Popen(  # noqa: S603 - the command is an owned argument list
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=None if environment is None else dict(environment),
            start_new_session=True,
        )

    def request_process_tree_stop(process: ManagedProcess) -> None:
        """Ask the host to stop one process and its children."""
        process.terminate()
