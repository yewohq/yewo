"""Acquire a healthy Appium server with explicit process ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit
from urllib.request import urlopen

from yewo.automation.appium_runtime import AppiumRuntime, resolve_appium_runtime
from yewo.automation.errors import AppiumServerError, AppiumStartupTimeoutError
from yewo.run_narration import current_narrator
from yewo.host import ManagedProcess, start_detached_process, stop_process_tree
from yewo.resources import ResourceLease, ResourceOwnership


_MAX_STATUS_BYTES = 1024 * 1024


@dataclass(frozen=True)
class AppiumServerConfig:
    """Settings for attaching to or starting one Appium server."""

    log_path: Path | None = None
    server_url: str = "http://127.0.0.1:4723"
    start_if_unavailable: bool = True
    startup_timeout_seconds: float = 30
    probe_timeout_seconds: float = 0.75
    poll_interval_seconds: float = 0.25
    allow_insecure_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _parse_server_url(self.server_url)
        for name, value in (
            ("startup timeout", self.startup_timeout_seconds),
            ("probe timeout", self.probe_timeout_seconds),
            ("poll interval", self.poll_interval_seconds),
        ):
            if value <= 0:
                raise ValueError(f"The Appium {name} must be greater than zero.")
        if any(not feature.strip() or "," in feature for feature in self.allow_insecure_features):
            raise ValueError("Each insecure Appium feature must be one nonempty name.")


class AppiumServerLease(ResourceLease):
    """A healthy Appium endpoint and its optional run-owned process."""

    def __init__(
        self,
        server_url: str,
        ownership: ResourceOwnership,
        *,
        log_path: Path | None = None,
        release: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(ownership, release)
        self.server_url = server_url.rstrip("/")
        self.log_path = log_path


class _ProcessStarter(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> ManagedProcess: ...


class _StatusProbe(Protocol):
    def __call__(self, status_url: str, *, timeout_seconds: float) -> bool: ...


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class _SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def _probe_status(status_url: str, *, timeout_seconds: float) -> bool:
    try:
        with urlopen(status_url, timeout=timeout_seconds) as response:  # noqa: S310
            if response.status != 200:
                return False
            body = response.read(_MAX_STATUS_BYTES + 1)
            if len(body) > _MAX_STATUS_BYTES:
                return False
            payload = json.loads(body)
    except (HTTPError, URLError, OSError, ValueError, TypeError):
        return False
    value = payload.get("value") if isinstance(payload, dict) else None
    return isinstance(value, dict) and value.get("ready") is True


def _parse_server_url(server_url: str) -> SplitResult:
    try:
        url_parts = urlsplit(server_url)
        port = url_parts.port
    except ValueError as error:
        raise ValueError("The Appium server URL has an invalid port.") from error
    if (
        url_parts.scheme not in {"http", "https"}
        or not url_parts.hostname
        or port is None
        or url_parts.username is not None
        or url_parts.password is not None
        or url_parts.query
        or url_parts.fragment
    ):
        raise ValueError("The Appium server URL must contain only a scheme, host, port, and path.")
    return url_parts


def _status_url(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/status"


def _is_local_http(url_parts: SplitResult) -> bool:
    return url_parts.scheme == "http" and url_parts.hostname in {"127.0.0.1", "::1", "localhost"}


def _trace_appium(message: str) -> None:
    narrator = current_narrator()
    if narrator is not None:
        narrator.trace("appium-server", message)


class AppiumServerManager:
    """Attach to an Appium endpoint or start and own a local process."""

    def __init__(
        self,
        *,
        runtime_resolver: Callable[[], AppiumRuntime] = resolve_appium_runtime,
        process_starter: _ProcessStarter = start_detached_process,
        status_probe: _StatusProbe = _probe_status,
        clock: _Clock | None = None,
    ) -> None:
        self._runtime_resolver = runtime_resolver
        self._process_starter = process_starter
        self._status_probe = status_probe
        self._clock = clock or _SystemClock()

    def acquire(self, config: AppiumServerConfig) -> AppiumServerLease:
        """Return a healthy server without taking ownership of an existing one."""
        status_url = _status_url(config.server_url)
        if self._is_ready(status_url, config):
            _trace_appium(f"server={config.server_url} ownership=external state=ready")
            return AppiumServerLease(config.server_url, ResourceOwnership.EXTERNAL)
        if not config.start_if_unavailable:
            raise AppiumServerError(f"No ready Appium server exists at {config.server_url}.")
        url_parts = _parse_server_url(config.server_url)
        if not _is_local_http(url_parts):
            raise AppiumServerError(
                "Yewo can start Appium only on a local HTTP endpoint. "
                "Start the configured remote or HTTPS server separately."
            )
        log_path = config.log_path
        if log_path is None:
            raise AppiumServerError("A log path is required to start an Appium server.")
        runtime = self._runtime_resolver()
        _trace_appium(
            f"server={config.server_url} node={runtime.node_executable} "
            f"appium_script={runtime.appium_script} state=starting"
        )
        self._prepare_log(log_path)
        command = self._command(config, url_parts, runtime, log_path)
        try:
            process = self._process_starter(command, environment=runtime.environment)
        except OSError as error:
            raise AppiumServerError("Could not start the Appium process.") from error
        try:
            ownership = self._wait_until_ready(process, status_url, config)
        except BaseException:
            with suppress(Exception):
                stop_process_tree(process)
            raise
        if ownership is ResourceOwnership.EXTERNAL:
            _trace_appium(f"server={config.server_url} ownership=external state=ready")
            return AppiumServerLease(config.server_url, ownership)
        _trace_appium(f"server={config.server_url} ownership=run state=ready")
        return AppiumServerLease(
            config.server_url,
            ResourceOwnership.RUN,
            log_path=log_path,
            release=lambda: stop_process_tree(process),
        )

    def _is_ready(self, status_url: str, config: AppiumServerConfig) -> bool:
        return self._status_probe(
            status_url,
            timeout_seconds=config.probe_timeout_seconds,
        )

    def _wait_until_ready(
        self,
        process: ManagedProcess,
        status_url: str,
        config: AppiumServerConfig,
    ) -> ResourceOwnership:
        deadline = self._clock.monotonic() + config.startup_timeout_seconds
        while True:
            if process.poll() is not None:
                if self._is_ready(status_url, config):
                    return ResourceOwnership.EXTERNAL
                raise AppiumServerError(f"Appium stopped during startup. See {config.log_path}.")
            if self._is_ready(status_url, config):
                return ResourceOwnership.RUN
            if self._clock.monotonic() >= deadline:
                raise AppiumStartupTimeoutError(
                    f"Appium did not become ready within "
                    f"{config.startup_timeout_seconds:g} seconds. See {config.log_path}."
                )
            self._clock.sleep(config.poll_interval_seconds)

    @staticmethod
    def _prepare_log(log_path: Path) -> None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AppiumServerError(
                f"Could not create the Appium log directory: {log_path.parent}"
            ) from error

    @staticmethod
    def _command(
        config: AppiumServerConfig,
        server_url_parts: SplitResult,
        runtime: AppiumRuntime,
        log_path: Path,
    ) -> tuple[str, ...]:
        host = server_url_parts.hostname
        port = server_url_parts.port
        if host is None or port is None:
            raise AppiumServerError("The Appium server URL has no host or port.")
        command = [
            str(runtime.node_executable),
            str(runtime.appium_script),
            "--address",
            host,
            "--port",
            str(port),
            "--log",
            str(log_path),
            "--log-level",
            "warn",
            "--log-no-colors",
        ]
        base_path = server_url_parts.path.rstrip("/")
        if base_path:
            command.extend(("--base-path", base_path))
        if config.allow_insecure_features:
            command.extend(("--allow-insecure", ",".join(config.allow_insecure_features)))
        return tuple(command)
