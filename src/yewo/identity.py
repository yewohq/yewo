"""What produced a run: the commit, the build, the app and the device.

A report that says a scenario failed, and cannot say which build it failed
on, is not evidence of a defect. Two runs of one journey differ by the app,
the device and the commit first. They differ by what Yewo decided second.

Every reader here is allowed to find nothing. A machine with no git, an app
Yewo cannot parse, and a device that reports no model are all ordinary.
Each one reports `None` rather than guessing or raising. A run must not end
because its own identity was unreadable.
"""

import hashlib
import logging
import os
import plistlib
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

from pydantic import BaseModel, Field

from yewo.utils.subprocess_runner import run_command

if TYPE_CHECKING:
    from yewo.agent.device.platform import Platform

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
"""How much of an artifact is hashed at a time, so a large build is streamed."""

SERIAL_VISIBLE_CHARS = 4
"""Characters of a device serial left legible. Enough to tell two apart."""

VERSION_NAME = re.compile(r"versionName='([^']*)'")
VERSION_CODE = re.compile(r"versionCode='([^']*)'")
PACKAGE_NAME = re.compile(r"package: name='([^']*)'")


NAMESPACE = uuid.UUID("6f1b4a52-0d3c-5e8a-9b27-1c4d6e8f0a35")
"""The namespace every derived identifier is computed under."""


def new_uuid() -> str:
    """One identifier for something that happened: a run, a step, a call."""
    return str(uuid.uuid4())


def derived_uuid(*parts: object) -> str:
    """One identifier for a part of something already identified.

    An expected outcome is the third outcome of one scenario, and nothing
    else. Derived rather than drawn, so two runs of one locked scenario name
    its outcomes the same way. No field has to carry an identifier the model
    would otherwise be asked to write.
    """
    return str(uuid.uuid5(NAMESPACE, ":".join(str(part) for part in parts)))


class GitIdentity(BaseModel):
    """The commit a run tested, where a repository could be found."""

    sha: Optional[str] = None
    branch: Optional[str] = None
    pull_request: Optional[str] = None
    dirty: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the checkout held changes the commit does not. A run "
            "of edited files is not evidence about the commit it names. "
            "None where no repository could be read."
        ),
    )
    diff_sha256: Optional[str] = Field(
        default=None,
        description=(
            "The uncommitted diff, hashed. Two dirty runs of the same "
            "commit are the same run only if this matches. Absent on a "
            "clean checkout."
        ),
    )


class CiIdentity(BaseModel):
    """The build that ran Yewo, where one did."""

    provider: Optional[str] = None
    run_url: Optional[str] = None


class AppIdentity(BaseModel):
    """The build under test, read from the artifact rather than from a setting."""

    package: Optional[str] = None
    version: Optional[str] = None
    build: Optional[str] = None
    artifact_sha256: Optional[str] = None
    installed: bool = Field(
        default=True,
        description=(
            "Whether this run installed the artifact these fields describe. "
            "False under --reuse-app, where the app on the device was put "
            "there by something else and can be a different build."
        ),
    )


class DeviceIdentity(BaseModel):
    """The target a run used, as the driver reported it."""

    model: Optional[str] = None
    os_version: Optional[str] = None
    emulator: Optional[bool] = None
    serial: Optional[str] = Field(
        default=None,
        description="Masked to its last few characters. It names hardware.",
    )


def _read(*args: str, timeout: float = 10) -> Optional[str]:
    """One short command's output, or nothing at all.

    Nothing is a legitimate answer for every caller here: no git, no SDK, or a
    directory that is not a repository.
    """
    try:
        result = run_command(list(args), timeout=timeout, log_command=False)
    except Exception as exc:  # noqa: BLE001 - identity must not fail a run
        logger.debug("Could not run %s: %s", args[0], exc)
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def git_identity(directory: Optional[Path] = None) -> GitIdentity:
    """The commit at `directory`, taken from the build first and git second.

    A build names the commit it checked out, and that is the commit under
    test. Most builds leave a detached checkout. Such a checkout reports its
    branch as `HEAD` to git, and by name to the environment.
    """
    repo_path = str((directory or Path.cwd()).resolve())
    environment = _ci_git()
    if environment.sha:
        # The build names the commit. Whether the checkout still matches it
        # is a different question, and a job that writes a file before
        # running is as unreproducible as a laptop that did.
        return environment.model_copy(update=_changes(repo_path))

    branch = _read("git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    sha = _read("git", "-C", repo_path, "rev-parse", "HEAD")
    return GitIdentity(
        sha=sha,
        branch=None if branch == "HEAD" else branch,
        pull_request=None,
        **(_changes(repo_path) if sha else {}),
    )


def _changes(repo_path: str) -> dict:
    """Whether the checkout holds what its commit does not, and which.

    None for both where no repository answered. A run outside a checkout
    has no commit to be dirty against, and saying `False` would claim it
    was clean.
    """
    changed = _diff(repo_path)
    if changed is None:
        return {"dirty": None, "diff_sha256": None}
    return {
        "dirty": bool(changed),
        "diff_sha256": (
            hashlib.sha256(changed.encode("utf-8")).hexdigest() if changed else None
        ),
    }


def _diff(repo_path: str) -> Optional[str]:
    """Everything the checkout holds that its commit does not.

    Tracked changes, and untracked files by name and by content. A name
    alone gives two different versions of one new file the same hash. The
    hash exists to tell two dirty runs apart.

    What this returns is hashed and discarded. It is the working tree, and
    a run's artifacts are not the place to keep a copy of it.
    """
    tracked = _read("git", "-C", repo_path, "diff", "HEAD")
    listed = _read("git", "-C", repo_path, "ls-files", "--others", "--exclude-standard")
    if tracked is None and listed is None:
        return None
    return f"{tracked or ''}\n{_untracked(repo_path, listed)}".strip()


def _untracked(repo_path: str, listed: Optional[str]) -> str:
    """Each untracked file, as its path and the hash of what is in it."""
    untracked_entries = []
    for name in (listed or "").splitlines():
        if not name.strip():
            continue
        digest = artifact_sha256(Path(repo_path) / name) or "unreadable"
        untracked_entries.append(f"{name} {digest}")
    return "\n".join(untracked_entries)


def _ci_git() -> GitIdentity:
    """What the build says about the commit, for the providers Yewo reads."""
    if os.getenv("GITHUB_ACTIONS"):
        reference = os.getenv("GITHUB_REF", "")
        pull = reference.split("/")[2] if reference.startswith("refs/pull/") else None
        return GitIdentity(
            sha=os.getenv("GITHUB_SHA") or None,
            branch=os.getenv("GITHUB_HEAD_REF") or os.getenv("GITHUB_REF_NAME") or None,
            pull_request=pull,
        )
    if os.getenv("GITLAB_CI"):
        return GitIdentity(
            sha=os.getenv("CI_COMMIT_SHA") or None,
            branch=os.getenv("CI_COMMIT_REF_NAME") or None,
            pull_request=os.getenv("CI_MERGE_REQUEST_IID") or None,
        )
    return GitIdentity()


def ci_identity() -> CiIdentity:
    """The build this run belongs to, so a failure links back to its job."""
    if os.getenv("GITHUB_ACTIONS"):
        server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
        repository = os.getenv("GITHUB_REPOSITORY")
        run = os.getenv("GITHUB_RUN_ID")
        url = f"{server}/{repository}/actions/runs/{run}" if repository and run else None
        return CiIdentity(provider="github_actions", run_url=url)
    if os.getenv("GITLAB_CI"):
        return CiIdentity(provider="gitlab_ci", run_url=os.getenv("CI_PIPELINE_URL"))
    if os.getenv("CI"):
        return CiIdentity(provider="unnamed")
    return CiIdentity()


def artifact_sha256(path: Path) -> Optional[str]:
    """The build's content hash, for a file or for a bundle directory.

    A bundle is hashed over its own paths as well as its bytes, so moving a
    file inside it changes the hash. Walked in sorted order, because a
    filesystem's own order is not stable between machines.
    """
    digest = hashlib.sha256()
    try:
        if path.is_dir():
            for member in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(str(member.relative_to(path)).encode("utf-8"))
                digest.update(b"\0")
                _absorb(digest, member)
        else:
            _absorb(digest, path)
    except OSError as exc:
        logger.debug("Could not hash %s: %s", path, exc)
        return None
    return digest.hexdigest()


def _absorb(digest, path: Path) -> None:
    """Streams one file into the hash, so a large build is not read at once."""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)


def app_identity(app_path: Optional[str], installed: bool = True) -> AppIdentity:
    """The build's own account of itself, read from the artifact.

    Read from the file rather than from the device. A device reports the
    package it runs and never the version that produced it.

    Args:
        app_path: The artifact on disk.
        installed: False where this run did not install it. The artifact is
            then a candidate and not the app under test, so nothing about it
            is recorded. A hash of a build that is not running is worse than
            no hash. A reader would trust it.
    """
    if not app_path:
        return AppIdentity(installed=installed)
    if not installed:
        # Under --reuse-app the app on the device was put there by something
        # else. A rebuilt debug APK keeps its version, so the artifact beside
        # the journey file can differ from what is running with nothing to
        # say so.
        return AppIdentity(installed=False)
    path = Path(app_path)
    if not path.exists():
        return AppIdentity()

    identity = _android_identity(path) if path.suffix == ".apk" else _apple_identity(path)
    return identity.model_copy(update={"artifact_sha256": artifact_sha256(path)})


def _android_identity(apk: Path) -> AppIdentity:
    """The package and version inside an APK, through aapt."""
    from yewo.agent.device.adb_control import AdbControl

    try:
        aapt = AdbControl.find_aapt()
    except Exception as exc:  # noqa: BLE001 - an absent SDK is not a run failure
        logger.debug("No aapt to read %s: %s", apk, exc)
        return AppIdentity()

    badging = _read(aapt, "dump", "badging", str(apk), timeout=30)
    if not badging:
        return AppIdentity()
    return AppIdentity(
        package=_matched(PACKAGE_NAME, badging),
        version=_matched(VERSION_NAME, badging),
        build=_matched(VERSION_CODE, badging),
    )


def _matched(pattern: re.Pattern, text: str) -> Optional[str]:
    found = pattern.search(text)
    return found.group(1) if found else None


def _apple_identity(bundle: Path) -> AppIdentity:
    """The bundle id and version from an app bundle's Info.plist.

    An `.ipa` is a zip and is not opened here. Nothing is read rather than
    half-read, so a reader is never shown a version that came from elsewhere.
    """
    info = bundle / "Info.plist"
    if not info.is_file():
        return AppIdentity()
    try:
        with info.open("rb") as handle:
            plist = plistlib.load(handle)
    except Exception as exc:  # noqa: BLE001 - a malformed plist is not a run failure
        logger.debug("Could not read %s: %s", info, exc)
        return AppIdentity()
    return AppIdentity(
        package=plist.get("CFBundleIdentifier"),
        version=plist.get("CFBundleShortVersionString"),
        build=plist.get("CFBundleVersion"),
    )


def masked_serial(serial: Optional[str]) -> Optional[str]:
    """A serial reduced to its last few characters.

    A physical device's serial names the hardware, and a manifest is uploaded
    and shared. The tail is kept so two targets in one matrix read apart.
    """
    if not serial:
        return None
    if len(serial) <= SERIAL_VISIBLE_CHARS:
        return serial
    return "…" + serial[-SERIAL_VISIBLE_CHARS:]


class IdentifiedDevice(Protocol):
    """What this module reads off a session to describe the target."""

    platform: "Platform"
    serial: Optional[str]
    os_version: Optional[str]
    model: Optional[str]


def device_identity(device_control: Optional[IdentifiedDevice]) -> DeviceIdentity:
    """The target in front of this run, as the driver described it.

    Guarded whole. A session that answers some of this and not the rest is
    still worth recording. A run must not end because its target could not be
    described.
    """
    if device_control is None:
        return DeviceIdentity()
    try:
        return DeviceIdentity(
            model=device_control.model,
            os_version=device_control.os_version,
            emulator=_emulator(device_control),
            serial=masked_serial(device_control.serial),
        )
    except Exception as exc:  # noqa: BLE001 - identity must not fail a run
        logger.debug("Could not describe the target: %s", exc)
        return DeviceIdentity()


def _emulator(device_control: IdentifiedDevice) -> Optional[bool]:
    """Whether this is a virtual device, where that can be established.

    Guarded on its own. It is the one read here that asks a command line. A
    failure in it must not take the model and the version with it.
    """
    from yewo.testing.judge.capabilities import target_of

    try:
        return target_of(device_control).emulator
    except Exception as exc:  # noqa: BLE001 - identity must not fail a run
        logger.debug("Could not establish the kind of target: %s", exc)
        return None
