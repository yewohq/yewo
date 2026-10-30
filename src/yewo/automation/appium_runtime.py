"""Resolve an installed Appium server and compatible Node runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from yewo.automation.errors import AppiumRuntimeError


NODE_BINARY_ENV = "YEWO_NODE_BINARY"
APPIUM_BINARY_ENV = "YEWO_APPIUM_BINARY"
_CHECK_TIMEOUT_SECONDS = 10
_NODE_COMPATIBILITY_SCRIPT = (
    "a=require('module').createRequire;b=process.argv[1];c=a(b);d=c(b);e=c('semver');"
    "f=d.engines&&d.engines.node;if(!f||!e.satisfies(process.versions.node,f)){"
    "process.stderr.write(`Node_${process.versions.node}_does_not_satisfy_${f||'unknown'}`);"
    "process.exit(1)}"
)


@dataclass(frozen=True)
class AppiumRuntime:
    """Executables and environment verified for starting Appium."""

    node_executable: Path
    appium_script: Path
    appium_version: str
    environment: dict[str, str]


class _RuntimeRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        environment: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]: ...


def _run_runtime_command(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
        check=False,
    )


def _existing_unique_paths(candidates: Sequence[str | Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    known: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if resolved in known or not resolved.is_file():
            continue
        known.add(resolved)
        paths.append(resolved)
    return tuple(paths)


def _node_candidates(environment: Mapping[str, str]) -> tuple[Path, ...]:
    configured = environment.get(NODE_BINARY_ENV)
    if configured:
        return _existing_unique_paths((configured,))
    discovered: list[str | Path] = []
    if active := shutil.which("node", path=environment.get("PATH")):
        discovered.append(active)
    for root in (Path("/opt/homebrew/opt"), Path("/usr/local/opt")):
        if root.is_dir():
            discovered.extend(sorted(root.glob("node@*/bin/node"), reverse=True))
    nvm_root = Path(environment.get("NVM_DIR", str(Path.home() / ".nvm")))
    versions = nvm_root / "versions" / "node"
    if versions.is_dir():
        discovered.extend(sorted(versions.glob("*/bin/node"), reverse=True))
    return _existing_unique_paths(discovered)


def _appium_package_json(appium_script: Path) -> Path | None:
    for directory in (appium_script.parent, *appium_script.parents):
        package = directory / "package.json"
        if package.is_file():
            return package
    return None


def resolve_appium_runtime(
    environment: Mapping[str, str] | None = None,
    *,
    runner: _RuntimeRunner = _run_runtime_command,
) -> AppiumRuntime:
    """Return a Node and Appium pair that passes local compatibility checks."""
    process_environment = dict(os.environ if environment is None else environment)
    configured_appium = process_environment.get(APPIUM_BINARY_ENV)
    discovered_appium = configured_appium or shutil.which(
        "appium", path=process_environment.get("PATH")
    )
    if not discovered_appium:
        raise AppiumRuntimeError(
            f"Appium is not installed or is not on PATH. Install Appium or set {APPIUM_BINARY_ENV}."
        )
    appium_script = Path(discovered_appium).expanduser()
    if not appium_script.is_file():
        raise AppiumRuntimeError(f"The configured Appium script does not exist: {appium_script}")
    appium_script = appium_script.resolve()
    node_candidates = _node_candidates(process_environment)
    if not node_candidates:
        raise AppiumRuntimeError(
            f"Yewo found no Node executable. Install Node or set {NODE_BINARY_ENV}."
        )

    package_json = _appium_package_json(appium_script)
    failures: list[str] = []
    for node_executable in node_candidates:
        runtime_environment = _runtime_environment(process_environment, node_executable)
        if package_json is not None and not _node_is_compatible(
            runner,
            node_executable,
            package_json,
            runtime_environment,
            failures,
        ):
            continue
        version = _appium_version(
            runner,
            node_executable,
            appium_script,
            runtime_environment,
            failures,
        )
        if version is not None:
            return AppiumRuntime(
                node_executable=node_executable,
                appium_script=appium_script,
                appium_version=version,
                environment=runtime_environment,
            )
    attempted = "; ".join(failures)
    raise AppiumRuntimeError(
        "No installed Node runtime can start Appium. Set "
        f"{NODE_BINARY_ENV} to a compatible executable. Tried: {attempted}"
    )


def _runtime_environment(environment: dict[str, str], node: Path) -> dict[str, str]:
    configured = dict(environment)
    existing_path = configured.get("PATH", "")
    configured["PATH"] = os.pathsep.join(filter(None, (str(node.parent), existing_path)))
    return configured


def _node_is_compatible(
    runner: _RuntimeRunner,
    node: Path,
    package_json: Path,
    environment: Mapping[str, str],
    failures: list[str],
) -> bool:
    result = _checked_runtime_command(
        runner,
        [str(node), "-e", _NODE_COMPATIBILITY_SCRIPT, str(package_json)],
        environment,
        node,
        failures,
    )
    return result is not None and result.returncode == 0


def _appium_version(
    runner: _RuntimeRunner,
    node: Path,
    appium_script: Path,
    environment: Mapping[str, str],
    failures: list[str],
) -> str | None:
    result = _checked_runtime_command(
        runner,
        [str(node), str(appium_script), "--version"],
        environment,
        node,
        failures,
    )
    if result is None:
        return None
    if result.returncode == 0:
        version = result.stdout.strip()
        if version:
            return version
        failures.append(f"{node}: Appium returned an empty version")
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    failures.append(f"{node}: {detail[-1] if detail else 'incompatible'}")
    return None


def _checked_runtime_command(
    runner: _RuntimeRunner,
    command: Sequence[str],
    environment: Mapping[str, str],
    node: Path,
    failures: list[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        result = runner(
            command,
            timeout_seconds=_CHECK_TIMEOUT_SECONDS,
            environment=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        failures.append(f"{node}: {type(error).__name__}")
        return None
    if result.returncode and "--version" not in command:
        detail = (result.stderr or result.stdout).strip()
        failures.append(f"{node}: {detail or 'incompatible'}")
    return result
