"""Individual environment checks behind `yewo doctor`.

Each check is read-only and returns a CheckResult rather than raising, so a
broken environment produces a full report instead of stopping at the first
problem. Anything that can be fixed by running a command carries that
command as its hint.
"""

import os
import platform as platform_module
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from yewo import __version__
from yewo.agent.perception.rapidocr_diagnostics import diagnose_rapidocr
from yewo.defaults import DEFAULT_MODEL
from yewo.utils.platform_checks import MIN_EMULATOR_FREE_GB, free_disk_gb
from yewo.utils.subprocess_runner import run_command
from yewo.utils.vision_deps import VISION_PACKAGES, missing_vision_packages

MIN_NODE_MAJOR = 18
MIN_PYTHON = (3, 12)

ANDROID_SDK_ENV_VARS = ("ANDROID_HOME", "ANDROID_SDK_ROOT")
MACOS_DEFAULT_SDK = Path.home() / "Library" / "Android" / "sdk"

PATH_EXPORT_HINT = 'export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"'


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    status: Status
    detail: str = ""
    hint: Optional[str] = None


@dataclass
class CheckSection:
    """A named group of related checks."""

    title: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]


def is_macos() -> bool:
    return platform_module.system() == "Darwin"


def _capture(cmd: list[str], timeout: float = 20.0) -> Optional[subprocess.CompletedProcess]:
    """Runs a command quietly, returning None when the binary is missing or hangs."""
    try:
        return run_command(cmd, timeout=timeout, log_command=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def _streams(result: subprocess.CompletedProcess) -> str:
    """Appium and friends split output across stdout and stderr. Both are searched."""
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] >= MIN_PYTHON:
        return CheckResult("python", Status.OK, version)
    return CheckResult(
        "python",
        Status.FAIL,
        f"{version} is too old",
        f"Yewo needs Python {'.'.join(map(str, MIN_PYTHON))} or newer",
    )


def check_yewo() -> CheckResult:
    location = Path(__file__).resolve().parents[1]
    return CheckResult("yewo", Status.OK, f"{__version__} ({location})")


def check_vision_deps() -> CheckResult:
    missing = missing_vision_packages()
    if not missing:
        return CheckResult("vision deps", Status.OK, ", ".join(VISION_PACKAGES))
    return CheckResult(
        "vision deps",
        Status.FAIL,
        f"missing: {', '.join(missing)}",
        "yewo setup   (or: uv sync --extra vision)",
    )


def check_rapidocr_runtime() -> CheckResult:
    """Inspect the pinned OCR runtime without initializing or downloading models."""
    missing = missing_vision_packages()
    unavailable = [package for package in ("rapidocr", "onnxruntime") if package in missing]
    if unavailable:
        return CheckResult(
            "RapidOCR runtime",
            Status.SKIP,
            f"cannot inspect while missing: {', '.join(unavailable)}",
        )
    diagnostic = diagnose_rapidocr()
    if not diagnostic.is_ok:
        return CheckResult(
            "RapidOCR runtime",
            Status.FAIL,
            diagnostic.detail,
            diagnostic.hint,
        )
    versions = dict(diagnostic.versions)
    version_detail = ", ".join(
        f"{package} {versions[package]}" for package in ("rapidocr", "onnxruntime")
    )
    return CheckResult(
        "RapidOCR runtime",
        Status.OK,
        f"{version_detail}; {', '.join(diagnostic.providers)}",
    )


def check_credentials(model: str = DEFAULT_MODEL) -> CheckResult:
    """Reports whether the configured model's credentials are in the environment."""
    from yewo.llm.credentials import missing_credentials

    try:
        missing = missing_credentials(model)
    except ImportError as exc:  # pragma: no cover - litellm is a base dependency
        return CheckResult("credentials", Status.WARN, f"could not load litellm: {exc}")

    if not missing:
        return CheckResult("credentials", Status.OK, f"{model} is configured")

    hint = f"Add {'='.join([missing[0], '...'])} to a .env file in this directory"
    if missing[0] == "OPENROUTER_API_KEY":
        hint += " (https://openrouter.ai/keys)"
    return CheckResult(
        "credentials",
        Status.FAIL,
        f"{model} needs {', '.join(missing)}",
        hint,
    )


def check_disk_space() -> CheckResult:
    """An emulator or simulator needs several GB free. Model weights need a few hundred MB."""
    free = free_disk_gb()
    if free < MIN_EMULATOR_FREE_GB:
        return CheckResult(
            "disk space",
            Status.FAIL,
            f"{free:.1f}GB free",
            f"Emulators need at least {MIN_EMULATOR_FREE_GB:.0f}GB free to boot",
        )
    if free < MIN_EMULATOR_FREE_GB * 2:
        return CheckResult(
            "disk space",
            Status.WARN,
            f"{free:.1f}GB free",
            "Tight for an emulator plus screenshots and recordings",
        )
    return CheckResult("disk space", Status.OK, f"{free:.1f}GB free")


def check_node() -> CheckResult:
    if not shutil.which("node"):
        return CheckResult("node", Status.FAIL, "not installed", "brew install node")
    result = _capture(["node", "--version"])
    raw = (result.stdout or "").strip() if result else ""
    try:
        major = int(raw.lstrip("v").split(".")[0])
    except (ValueError, IndexError):
        return CheckResult("node", Status.WARN, f"unrecognised version: {raw or 'unknown'}")
    if major >= MIN_NODE_MAJOR:
        return CheckResult("node", Status.OK, raw)
    return CheckResult(
        "node",
        Status.FAIL,
        f"{raw} is too old (need v{MIN_NODE_MAJOR}+)",
        "brew install node",
    )


def check_appium() -> CheckResult:
    if not shutil.which("appium"):
        return CheckResult(
            "appium", Status.FAIL, "not installed", "yewo setup   (or: npm install -g appium)"
        )
    result = _capture(["appium", "--version"])
    version = (result.stdout or "").strip() if result else ""
    return CheckResult("appium", Status.OK, version or "installed")


def check_appium_driver(driver: str, needed: bool = True) -> CheckResult:
    """Checks one Appium driver. Drivers for platforms you don't target are informational."""
    name = f"driver: {driver}"
    if not shutil.which("appium"):
        return CheckResult(name, Status.SKIP, "appium not installed")
    result = _capture(["appium", "driver", "list", "--installed"])
    if result is None:
        return CheckResult(name, Status.WARN, "could not list drivers")
    if driver in _streams(result):
        return CheckResult(name, Status.OK, "installed")
    return CheckResult(
        name,
        Status.FAIL if needed else Status.WARN,
        "not installed",
        f"yewo setup   (or: appium driver install {driver})",
    )


def android_sdk_path() -> Optional[Path]:
    """Resolves the Android SDK from the environment, falling back to the macOS default."""
    for var in ANDROID_SDK_ENV_VARS:
        value = os.getenv(var)
        if value and Path(value).is_dir():
            return Path(value)
    if is_macos() and MACOS_DEFAULT_SDK.is_dir():
        return MACOS_DEFAULT_SDK
    return None


def check_android_sdk() -> CheckResult:
    sdk = android_sdk_path()
    if sdk is None:
        return CheckResult(
            "android sdk",
            Status.FAIL,
            "not found",
            "Install it via Android Studio, then: export ANDROID_HOME=$HOME/Library/Android/sdk",
        )
    if not any(os.getenv(var) for var in ANDROID_SDK_ENV_VARS):
        return CheckResult(
            "android sdk",
            Status.WARN,
            f"found at {sdk}, but ANDROID_HOME is not set",
            f'export ANDROID_HOME="{sdk}"',
        )
    return CheckResult("android sdk", Status.OK, str(sdk))


def _sdk_binary(sdk: Path, name: str) -> Optional[Path]:
    for relative in (f"platform-tools/{name}", f"emulator/{name}", f"tools/bin/{name}"):
        candidate = sdk / relative
        if candidate.exists():
            return candidate
    return None


def resolve_android_binary(name: str) -> Optional[Path]:
    """Finds an SDK binary on PATH, or inside the SDK as a fallback."""
    found = shutil.which(name)
    if found:
        return Path(found)
    sdk = android_sdk_path()
    return _sdk_binary(sdk, name) if sdk else None


def check_android_binary(name: str) -> CheckResult:
    if shutil.which(name):
        return CheckResult(name, Status.OK, "on PATH")
    sdk = android_sdk_path()
    if sdk and _sdk_binary(sdk, name):
        return CheckResult(
            name,
            Status.WARN,
            "in the SDK but not on PATH",
            PATH_EXPORT_HINT,
        )
    return CheckResult(
        name,
        Status.FAIL,
        "not found",
        "Install 'platform-tools' and 'emulator' from the Android Studio SDK Manager",
    )


def check_aapt() -> CheckResult:
    """aapt reads the package id and launch activity out of the APK under test."""
    sdk = android_sdk_path()
    if sdk is None:
        return CheckResult("aapt", Status.SKIP, "no android sdk")
    build_tools = sdk / "build-tools"
    if build_tools.is_dir():
        for candidate in sorted(build_tools.glob("*/aapt"), reverse=True):
            return CheckResult("aapt", Status.OK, str(candidate))
    return CheckResult(
        "aapt",
        Status.FAIL,
        "not found in build-tools",
        "Install any 'Android SDK Build-Tools' version from the SDK Manager",
    )


def check_avds(expected_device: Optional[str] = None) -> CheckResult:
    """Lists configured AVDs, warning when the device Yewo would target is absent."""
    from yewo.config import load_config

    expected = expected_device or load_config().get("device")
    emulator = resolve_android_binary("emulator")
    if emulator is None:
        return CheckResult("avds", Status.SKIP, "emulator not found")
    result = _capture([str(emulator), "-list-avds"])
    if result is None:
        return CheckResult("avds", Status.WARN, "could not list AVDs")
    avds = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if avds and expected not in avds:
        return CheckResult(
            "avds",
            Status.WARN,
            f"{', '.join(avds)} (no '{expected}')",
            f"Yewo targets '{expected}'. Run: yewo config set device {avds[0]}",
        )
    if avds:
        return CheckResult("avds", Status.OK, ", ".join(avds))
    return CheckResult(
        "avds",
        Status.FAIL,
        "none configured",
        f"Create one with: yewo devices create {expected}",
    )


def check_xcode() -> CheckResult:
    if not is_macos():
        return CheckResult("xcode", Status.SKIP, "iOS automation requires macOS")
    result = _capture(["xcode-select", "-p"])
    path = (result.stdout or "").strip() if result else ""
    if result is not None and result.returncode == 0 and path:
        return CheckResult("xcode", Status.OK, path)
    return CheckResult(
        "xcode",
        Status.FAIL,
        "command-line tools not installed",
        "xcode-select --install",
    )


def check_applesimutils() -> CheckResult:
    """Whether the host can read an iOS permission back.

    `simctl privacy` sets a decision and reports nothing. The read is
    `mobile: getPermission`, which runs this program on the host. Without it
    a permission scenario is skipped, because a state nothing confirms is not
    evidence.

    A warning rather than a failure. Every other iOS capability works
    without it.
    """
    if not is_macos():
        return CheckResult("applesimutils", Status.SKIP, "iOS automation requires macOS")
    if not shutil.which("applesimutils"):
        return CheckResult(
            "applesimutils",
            Status.WARN,
            "not installed, so iOS permission scenarios are skipped",
            "yewo setup   (or: brew tap wix/brew && brew install applesimutils)",
        )
    result = _capture(["applesimutils", "--version"])
    version = (result.stdout or "").strip() if result else ""
    return CheckResult("applesimutils", Status.OK, version or "installed")


def check_simulators() -> CheckResult:
    if not is_macos():
        return CheckResult("simulators", Status.SKIP, "iOS automation requires macOS")
    result = _capture(["xcrun", "simctl", "list", "devices", "available"])
    if result is None:
        return CheckResult("simulators", Status.SKIP, "xcrun unavailable")
    iphones = [
        line.strip()
        for line in (result.stdout or "").splitlines()
        if "iPhone" in line and "unavailable" not in line
    ]
    if iphones:
        return CheckResult("simulators", Status.OK, f"{len(iphones)} iPhone simulator(s) available")
    return CheckResult(
        "simulators",
        Status.FAIL,
        "no iPhone simulators available",
        "Install a simulator runtime: Xcode > Settings > Components",
    )


def run_checks(platforms: tuple[str, ...] = ("android", "ios")) -> list[CheckSection]:
    """Runs every check for the requested platforms and groups the results."""
    sections = [
        CheckSection(
            "Yewo",
            [
                check_python(),
                check_yewo(),
                check_vision_deps(),
                check_rapidocr_runtime(),
                check_credentials(),
                check_disk_space(),
            ],
        )
    ]

    appium = CheckSection("Appium", [check_node(), check_appium()])
    appium.results.append(check_appium_driver("uiautomator2", needed="android" in platforms))
    if is_macos():
        appium.results.append(check_appium_driver("xcuitest", needed="ios" in platforms))
    sections.append(appium)

    if "android" in platforms:
        sections.append(
            CheckSection(
                "Android",
                [
                    check_android_sdk(),
                    check_android_binary("adb"),
                    check_android_binary("emulator"),
                    check_aapt(),
                    check_avds(),
                ],
            )
        )

    if "ios" in platforms:
        sections.append(
            CheckSection(
                "iOS",
                [check_xcode(), check_simulators(), check_applesimutils()],
            )
        )

    return sections
