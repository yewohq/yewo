"""Locating the app binary and journey files to work with.

Yewo runs from wherever your project is, so paths are resolved against the
working directory and the journey file rather than assumed. Nothing here is
tied to a particular toolchain: any .apk, .app or .ipa is a candidate.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from yewo.errors import UsageError

logger = logging.getLogger(__name__)

ANDROID_SUFFIXES = (".apk",)
IOS_SUFFIXES = (".app", ".ipa")

JOURNEY_SUFFIXES = (".yaml", ".yml", ".json")

NOT_JOURNEYS = {
    "pubspec.yaml",
    "pubspec.lock",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "docker-compose.yaml",
    "docker-compose.yml",
    "codemagic.yaml",
    "analysis_options.yaml",
}

JOURNEY_KEYS = {"journey", "journeys", "workflows", "security_rules"}

# Where builds usually land, searched before a wider sweep.
BUILD_DIR_NAMES = ("build", "out", "dist", "outputs", "artifacts", "target")

MAX_SEARCH_DEPTH = 6
IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".dart_tool", "Pods"}


def _suffixes_for(platform: str) -> tuple[str, ...]:
    return IOS_SUFFIXES if platform == "ios" else ANDROID_SUFFIXES


def _walk(root: Path, depth: int):
    """Yields directories under root, skipping noise and bounding the depth."""
    if depth < 0 or not root.is_dir() or root.name in IGNORED_DIRS:
        return
    yield root
    try:
        children = sorted(root.iterdir())
    except (PermissionError, OSError):
        return
    for child in children:
        if child.is_dir() and child.name not in IGNORED_DIRS:
            # An .app bundle is a directory, so do not descend into it.
            if child.suffix in IOS_SUFFIXES:
                continue
            yield from _walk(child, depth - 1)


def _unique_dirs(search_dirs) -> list[Path]:
    seen: list[Path] = []
    for directory in search_dirs:
        if directory is None:
            continue
        resolved = Path(directory).resolve()
        if resolved not in seen:
            seen.append(resolved)
    return seen


def find_app_binaries(platform: str, *search_dirs: Optional[Path]) -> list[Path]:
    """Every app binary for the platform under the given directories, newest first."""
    suffixes = _suffixes_for(platform)
    found: dict[Path, float] = {}

    for start in _unique_dirs(search_dirs):
        for current in _walk(start, MAX_SEARCH_DEPTH):
            for entry in current.iterdir() if current.is_dir() else []:
                if entry.suffix in suffixes and entry.exists():
                    try:
                        found[entry.resolve()] = entry.stat().st_mtime
                    except OSError:
                        continue

    ranked = sorted(found, key=lambda path: found[path], reverse=True)
    # Binaries inside a build directory beat stray copies elsewhere.
    ranked.sort(key=lambda path: 0 if any(p in BUILD_DIR_NAMES for p in path.parts) else 1)
    return ranked


def _digest(path: Path) -> str:
    """Content digest for a binary, which on iOS is a bundle directory."""
    if not path.is_dir():
        return _file_digest(path)

    executable = _bundle_executable(path)
    if executable is not None:
        return _file_digest(executable)
    return _bundle_manifest_digest(path)


def _bundle_executable(bundle: Path) -> Optional[Path]:
    """The binary inside an .app, which is what actually changes between builds."""
    manifest = bundle / "Info.plist"
    if not manifest.is_file():
        return None
    try:
        import plistlib

        with manifest.open("rb") as handle:
            name = plistlib.load(handle).get("CFBundleExecutable")
    except Exception:
        return None
    candidate = bundle / name if name else None
    return candidate if candidate and candidate.is_file() else None


def _bundle_manifest_digest(bundle: Path) -> str:
    """Falls back to every file's path and size when no executable is named."""
    sha = hashlib.sha256()
    try:
        for entry in sorted(bundle.rglob("*")):
            if entry.is_file():
                sha.update(str(entry.relative_to(bundle)).encode())
                sha.update(str(entry.stat().st_size).encode())
    except OSError:
        return str(bundle)
    return sha.hexdigest()


def _file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                sha.update(chunk)
    except OSError:
        return str(path)
    return sha.hexdigest()


def distinct_binaries(paths: list[Path]) -> list[list[Path]]:
    """Groups paths holding the same bytes together, keeping the ranked order.

    Build tools routinely copy the same artifact to two places, such as a
    build output and a copy made beside it. That is not a choice worth
    bothering anyone about.
    """
    groups: list[list[Path]] = []
    signatures: dict[tuple[int, str], int] = {}

    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        same_size = [
            other
            for other in paths
            if other is not path and _safe_size(other) == size and other.is_dir() == path.is_dir()
        ]
        signature = (size, _digest(path) if same_size else str(path))
        index = signatures.get(signature)
        if index is None:
            signatures[signature] = len(groups)
            groups.append([path])
        else:
            groups[index].append(path)
    return groups


def _safe_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _built_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def looks_like_journey(path: Path) -> bool:
    """Whether a file actually describes a journey, rather than merely being YAML."""
    if path.name in NOT_JOURNEYS or path.name.startswith("."):
        return False
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(loaded, list):
        loaded = loaded[0] if loaded else None
    if not isinstance(loaded, dict):
        return False
    return bool(JOURNEY_KEYS & set(loaded))


def find_journey_files(directory: Path) -> list[Path]:
    """Journey files sitting directly in a directory, newest first."""
    candidates = [
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix in JOURNEY_SUFFIXES and looks_like_journey(path)
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_journey_file(journey_path: Optional[str]) -> Path:
    """The journey file to run, looking in the working directory when none is named.

    Raises:
        UsageError: If the directory holds no journey file, or several, because
            Yewo cannot choose between them on the user's behalf.
    """
    if journey_path:
        return Path(journey_path)

    candidates = find_journey_files(Path.cwd())
    if len(candidates) == 1:
        logger.info("Using journey file %s", candidates[0].name)
        return candidates[0]
    if not candidates:
        raise UsageError(
            f"No journey file found in {Path.cwd()}.\nPass one: yewo test <journey.yaml>"
        )
    listed = "\n".join(f"  {path.name}" for path in candidates)
    raise UsageError(
        f"Several journey files found in {Path.cwd()}:\n{listed}\n"
        "Pass the one you want: yewo test <journey.yaml>"
    )


def resolve_app_path(
    explicit: Optional[str], platform: str, *search_dirs: Optional[Path]
) -> Path:
    """Resolves the app binary, searching nearby when none was given.

    A relative path is tried against the working directory first, then against
    the other search directories. A journey file can then name a binary beside
    itself, however Yewo was invoked.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            for directory in search_dirs:
                if directory is None:
                    continue
                relative = Path(directory) / candidate
                if relative.exists():
                    return relative
        raise UsageError(
            f"App binary not found: {explicit}\n"
            "Pass --app-path with a path relative to this directory, or an absolute one."
        )

    discovered = find_app_binaries(platform, *search_dirs)
    if discovered:
        groups = distinct_binaries(discovered)
        chosen = groups[0][0]
        copies = groups[0][1:]

        if len(groups) == 1:
            logger.info("Using app binary %s (built %s)", chosen, _built_at(chosen))
            if copies:
                logger.debug(
                    "The same build is also at %s", ", ".join(str(c) for c in copies)
                )
        else:
            logger.warning(
                "Found %d different app binaries. Using the one built most "
                "recently: %s (%s)",
                len(groups),
                chosen,
                _built_at(chosen),
            )
            for group in groups[1:4]:
                logger.warning("  not using %s (%s)", group[0], _built_at(group[0]))
            logger.warning("  pass --app-path to choose a different one")
        return chosen

    looked = "\n".join(f"  {directory}" for directory in _unique_dirs(search_dirs))
    suffixes = " or ".join(_suffixes_for(platform))
    raise UsageError(
        f"No {platform} app binary ({suffixes}) found under:\n{looked}\n"
        "Build the app, then pass --app-path with a relative or absolute path "
        "to it — or set it once with: yewo config set app_path <path>"
    )
