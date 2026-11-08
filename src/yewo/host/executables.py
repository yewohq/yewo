"""Executable and SDK discovery for each supported host."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import sys


ANDROID_SDK_ENVIRONMENT = ("ANDROID_HOME", "ANDROID_SDK_ROOT")
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".bat", ".cmd")
_POSIX_EXECUTABLE_SUFFIXES = ("",)
_NODE_SCRIPT_SHIM_SUFFIXES = (".cmd", ".bat", ".ps1")
_COMMAND_SHIM_SUFFIXES = (".cmd", ".bat")


def executable_suffixes() -> tuple[str, ...]:
    """Return the file suffixes an executable carries on this host."""
    if sys.platform == "win32":
        return _WINDOWS_EXECUTABLE_SUFFIXES
    return _POSIX_EXECUTABLE_SUFFIXES


def build_host_command(executable: Path | str, *arguments: str) -> tuple[str, ...]:
    """Return an argument list that starts one executable, shim included.

    Windows cannot start a command shim directly. The command interpreter
    starts it instead.
    """
    path = Path(executable)
    if sys.platform == "win32" and path.suffix.lower() in _COMMAND_SHIM_SUFFIXES:
        return ("cmd", "/c", str(path), *arguments)
    return (str(path), *arguments)


def resolve_host_executable(name: str, search_path: str | None = None) -> Path | None:
    """Return one executable from the host search path."""
    executable_path = shutil.which(name, path=search_path)
    return Path(executable_path) if executable_path else None


def resolve_sdk_executable(directory: Path, name: str) -> Path | None:
    """Return one executable inside an SDK directory, with its host suffix."""
    for suffix in executable_suffixes():
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def default_android_sdk_root() -> Path | None:
    """Return the standard Android SDK location for this host."""
    if sys.platform == "win32":
        local_application_data = os.environ.get("LOCALAPPDATA")
        if not local_application_data:
            return None
        return Path(local_application_data) / "Android" / "Sdk"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Android" / "sdk"
    return Path.home() / "Android" / "Sdk"


def resolve_android_sdk_root(environment: Mapping[str, str]) -> Path | None:
    """Return the configured Android SDK, or this host's standard location."""
    for name in ANDROID_SDK_ENVIRONMENT:
        configured_sdk_root = environment.get(name)
        if configured_sdk_root and Path(configured_sdk_root).expanduser().is_dir():
            return Path(configured_sdk_root).expanduser().resolve()
    default = default_android_sdk_root()
    return default.resolve() if default is not None and default.is_dir() else None


def node_installation_roots() -> tuple[Path, ...]:
    """Return directories that hold additional Node installations."""
    if sys.platform == "win32":
        program_files = os.environ.get("PROGRAMFILES")
        application_data = os.environ.get("APPDATA")
        roots = [Path(program_files) / "nodejs"] if program_files else []
        if application_data:
            roots.append(Path(application_data) / "nvm")
        return tuple(roots)
    if sys.platform == "darwin":
        return (Path("/opt/homebrew/opt"), Path("/usr/local/opt"))
    return (Path("/home/linuxbrew/.linuxbrew/opt"), Path("/usr/local/opt"))


def node_executable_patterns() -> tuple[str, ...]:
    """Return glob patterns that find Node beneath an installation root."""
    if sys.platform == "win32":
        return ("node.exe", "v*/node.exe")
    return ("node", "node@*/bin/node")


def is_node_script_shim(path: Path) -> bool:
    """Return whether a path is a host shim rather than a Node script."""
    return path.suffix.lower() in _NODE_SCRIPT_SHIM_SUFFIXES


def resolve_node_package_entry_point(shim_path: Path, package_name: str) -> Path | None:
    """Return the Node entry point a host shim starts.

    Windows installs an npm command shim, not a link to the script. Yewo
    reads the entry point from the installed package.
    """
    for directory in (shim_path.parent, *shim_path.parents):
        for library in ("node_modules", "lib/node_modules"):
            package_root = directory / library / package_name
            if not package_root.is_dir():
                continue
            for relative in ("index.js", f"{package_name}.js", "build/lib/main.js"):
                candidate = package_root / relative
                if candidate.is_file():
                    return candidate
    return None


def resolve_emulator_executable(
    environment: Mapping[str, str],
    search_path: str | None = None,
) -> Path | None:
    """Return the Android emulator executable for this host."""
    emulator_path = resolve_host_executable("emulator", search_path)
    if emulator_path is not None:
        return emulator_path
    sdk = resolve_android_sdk_root(environment)
    if sdk is None:
        return None
    return resolve_sdk_executable(sdk / "emulator", "emulator")


def resolve_android_sdk_tool(
    name: str,
    sdk: Path,
    search_path: str | None = None,
) -> Path | None:
    """Return one Android SDK tool from the search path or the SDK layout."""
    executable_path = resolve_host_executable(name, search_path)
    if executable_path is not None:
        return executable_path.resolve()
    for relative in ("platform-tools", "emulator", "tools/bin"):
        candidate = resolve_sdk_executable(sdk / relative, name)
        if candidate is not None:
            return candidate
    return None


def resolve_build_tools_executables(sdk: Path, name: str) -> tuple[Path, ...]:
    """Return every build-tools executable with one name, newest release last."""
    build_tools = sdk / "build-tools"
    if not build_tools.is_dir():
        return ()
    matches: list[Path] = []
    for release in build_tools.iterdir():
        if not release.is_dir():
            continue
        candidate = resolve_sdk_executable(release, name)
        if candidate is not None:
            matches.append(candidate)
    return tuple(matches)


def sorted_existing_paths(candidates: Sequence[str | Path]) -> tuple[Path, ...]:
    """Return each unique file that exists, in candidate order."""
    paths: list[Path] = []
    known: set[Path] = set()
    for candidate in candidates:
        try:
            resolved_path = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if resolved_path in known or not resolved_path.is_file():
            continue
        known.add(resolved_path)
        paths.append(resolved_path)
    return tuple(paths)
