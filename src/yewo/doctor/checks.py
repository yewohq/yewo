"""Read-only project, model, automation, and device checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform as platform_module
import shutil
import subprocess
import sys

from yewo.automation.appium_runtime import (
    AppiumRuntime,
    parse_installed_appium_drivers,
    resolve_appium_runtime,
)
from yewo.automation.errors import AppiumRuntimeError
from yewo.devices import DeviceDescriptor
from yewo.devices._commands import CommandRunner, run_command
from yewo.devices.errors import DeviceInventoryError
from yewo.devices.ios.inventory import (
    IosDeviceInventory,
    IosPhysicalDeviceInventory,
    SimctlDeviceInventory,
)
from yewo.models import ModelProfile
from yewo.models.profiles import missing_model_credentials
from yewo.project_config import (
    DEFAULT_PLATFORM,
    LoadedYewoProject,
    YewoProjectCompileError,
    YewoProjectError,
    compile_yewo_run_configuration,
    load_nearest_yewo_project,
)
from yewo.doctor.records import DoctorCheck, DoctorReport, DoctorSection, DoctorStatus


MINIMUM_PYTHON = (3, 12)
MINIMUM_FREE_GB = 10
ANDROID_SDK_ENVIRONMENT = ("ANDROID_HOME", "ANDROID_SDK_ROOT")

CredentialCheck = Callable[[ModelProfile], tuple[str, ...]]


@dataclass(frozen=True)
class _ProjectInspection:
    checks: tuple[DoctorCheck, ...]
    project: LoadedYewoProject | None
    platform: str | None
    device_name: str | None


def run_doctor(
    start: Path | None = None,
    requested_platform: str | None = None,
    *,
    credential_check: CredentialCheck = missing_model_credentials,
    runner: CommandRunner = run_command,
    environment: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Inspect every relevant prerequisite without changing local state."""
    configured_environment = dict(os.environ if environment is None else environment)
    project = _inspect_project(start, credential_check)
    platforms = _selected_platforms(requested_platform, project.platform)
    sections = [
        DoctorSection(title="Project", checks=project.checks),
        DoctorSection(title="System", checks=_system_checks(start)),
        DoctorSection(
            title="Appium",
            checks=_appium_checks(platforms, runner, configured_environment),
        ),
    ]
    if "android" in platforms:
        sections.append(
            DoctorSection(
                title="Android",
                checks=_android_checks(
                    _expected_device(project, "android"),
                    runner,
                    configured_environment,
                ),
            )
        )
    if "ios" in platforms:
        sections.append(
            DoctorSection(
                title="iOS",
                checks=_ios_checks(_expected_device(project, "ios"), runner),
            )
        )
    return DoctorReport(sections=tuple(sections))


def _inspect_project(start: Path | None, credential_check: CredentialCheck) -> _ProjectInspection:
    checks: list[DoctorCheck] = []
    try:
        project = load_nearest_yewo_project(start)
    except YewoProjectError as error:
        checks.append(_fail("yewo.yaml", str(error), "Create a yewo.yaml for this project."))
        return _ProjectInspection(tuple(checks), None, None, None)
    except Exception as error:  # noqa: BLE001 - Doctor must report every project loader failure
        checks.append(
            _fail(
                "yewo.yaml",
                f"could not inspect the project: {type(error).__name__}",
                "Fix yewo.yaml, then run yewo doctor again.",
            )
        )
        return _ProjectInspection(tuple(checks), None, None, None)

    checks.append(_ok("yewo.yaml", str(project.source_path)))
    settings = project.settings
    if settings.app_path is not None:
        checks.append(_application_artifact(settings.app_path, settings.platform))
    try:
        configuration = compile_yewo_run_configuration(project, "Doctor configuration probe.")
    except YewoProjectCompileError as error:
        checks.append(_fail("run configuration", str(error)))
    else:
        checks.append(_ok("run configuration", "complete and internally compatible"))
        checks.extend(_credential_checks(configuration.model_profiles.profiles, credential_check))
    return _ProjectInspection(
        tuple(checks),
        project,
        settings.platform,
        settings.device_name,
    )


def _application_artifact(path: Path, platform: str | None) -> DoctorCheck:
    if not path.exists():
        return _fail(
            "application artifact", f"not found: {path}", "Build the app or fix app in yewo.yaml."
        )
    expected = (
        {"android": {".apk"}, "ios": {".app", ".ipa"}}.get(platform)
        if platform is not None
        else None
    )
    if expected is not None and path.suffix.lower() not in expected:
        suffixes = ", ".join(sorted(expected))
        return _fail(
            "application artifact",
            f"{path} does not match {platform}",
            f"Use one of these artifact types: {suffixes}.",
        )
    if path.suffix.lower() == ".app" and not path.is_dir():
        return _fail("application artifact", f"iOS .app is not a directory: {path}")
    if path.suffix.lower() != ".app" and not path.is_file():
        return _fail("application artifact", f"artifact is not a file: {path}")
    return _ok("application artifact", str(path))


def _credential_checks(
    profiles: Sequence[ModelProfile],
    credential_check: CredentialCheck,
) -> tuple[DoctorCheck, ...]:
    grouped: dict[str, list[ModelProfile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.model, []).append(profile)
    checks: list[DoctorCheck] = []
    for model, matching in grouped.items():
        roles = ", ".join(profile.role.value for profile in matching)
        try:
            missing = credential_check(matching[0])
        except Exception:  # noqa: BLE001 - provider configuration boundary
            checks.append(_fail(f"model credentials: {model}", "could not inspect credentials"))
            continue
        if missing:
            names = ", ".join(missing)
            checks.append(
                _fail(
                    f"model credentials: {model}",
                    f"missing {names} for {roles}",
                    f"Add {missing[0]}=... to the project .env file.",
                )
            )
        else:
            checks.append(_ok(f"model credentials: {model}", f"configured for {roles}"))
    return tuple(checks)


def _system_checks(start: Path | None) -> tuple[DoctorCheck, ...]:
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    python = (
        _ok("python", python_version)
        if sys.version_info[:2] >= MINIMUM_PYTHON
        else _fail("python", f"{python_version} is too old", "Install Python 3.12 or newer.")
    )
    try:
        yewo_version = version("yewo")
    except PackageNotFoundError:
        yewo = _fail("yewo", "package metadata is unavailable", "Install the Yewo package.")
    else:
        yewo = _ok("yewo", yewo_version)
    disk_root = (start or Path.cwd()).resolve()
    try:
        free_gb = shutil.disk_usage(disk_root).free / (1024**3)
    except OSError as error:
        disk = _fail("disk space", f"could not inspect {disk_root}: {error}")
    else:
        disk = (
            _ok("disk space", f"{free_gb:.1f} GB free")
            if free_gb >= MINIMUM_FREE_GB
            else _fail(
                "disk space",
                f"{free_gb:.1f} GB free",
                f"Free at least {MINIMUM_FREE_GB} GB for devices and run artifacts.",
            )
        )
    return python, yewo, disk


def _appium_checks(
    platforms: tuple[str, ...],
    runner: CommandRunner,
    environment: Mapping[str, str],
) -> tuple[DoctorCheck, ...]:
    try:
        runtime = resolve_appium_runtime(environment)
    except AppiumRuntimeError as error:
        return (
            _fail(
                "node + appium runtime",
                str(error),
                "Install a compatible Node and Appium runtime.",
            ),
            *tuple(
                _skip(f"driver: {_driver(platform)}", "Appium is unavailable")
                for platform in platforms
            ),
        )
    checks = [
        _ok(
            "node + appium runtime",
            f"Appium {runtime.appium_version}. Node {runtime.node_executable}",
        )
    ]
    installed = _installed_appium_drivers(runtime, runner)
    if installed is None:
        checks.extend(
            _warn(f"driver: {_driver(platform)}", "could not list drivers")
            for platform in platforms
        )
    else:
        for platform in platforms:
            driver = _driver(platform)
            checks.append(
                _ok(f"driver: {driver}", "installed")
                if driver in installed
                else _fail(
                    f"driver: {driver}",
                    "not installed",
                    f"appium driver install {driver}",
                )
            )
    return tuple(checks)


def _installed_appium_drivers(
    runtime: AppiumRuntime,
    runner: CommandRunner,
) -> frozenset[str] | None:
    result = _capture(
        runner,
        [
            str(runtime.node_executable),
            str(runtime.appium_script),
            "driver",
            "list",
            "--installed",
            "--json",
        ],
    )
    if result is None or result.returncode:
        return None
    return parse_installed_appium_drivers(result.stdout or "")


def _android_checks(
    expected_device: str | None,
    runner: CommandRunner,
    environment: Mapping[str, str],
) -> tuple[DoctorCheck, ...]:
    sdk = _android_sdk(environment)
    if sdk is None:
        return (
            _fail(
                "Android SDK",
                "not found",
                "Set ANDROID_HOME to an installed Android SDK.",
            ),
            _skip("adb", "Android SDK is unavailable"),
            _skip("emulator", "Android SDK is unavailable"),
            _skip("aapt", "Android SDK is unavailable"),
            _skip("configured AVDs", "Android SDK is unavailable"),
        )
    checks = [_ok("Android SDK", str(sdk))]
    adb = _android_binary("adb", sdk, environment)
    emulator = _android_binary("emulator", sdk, environment)
    checks.append(_binary_check("adb", adb))
    checks.append(_binary_check("emulator", emulator))
    checks.append(_aapt_check(sdk))
    if emulator is None:
        checks.append(_skip("configured AVDs", "emulator is unavailable"))
    else:
        checks.append(_avd_check(emulator, expected_device, runner))
    return tuple(checks)


def _avd_check(
    emulator: Path,
    expected: str | None,
    runner: CommandRunner,
) -> DoctorCheck:
    result = _capture(runner, [str(emulator), "-list-avds"])
    if result is None or result.returncode:
        return _warn("configured AVDs", "could not list Android virtual devices")
    names = tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
    if expected is not None and expected in names:
        return _ok("configured AVDs", f"{expected} is available")
    if not names:
        return _warn(
            "configured AVDs",
            "none configured. An attached physical device can still satisfy the run",
        )
    detail = ", ".join(names)
    if expected is not None:
        return _warn(
            "configured AVDs",
            f"'{expected}' is not an AVD. Configured AVDs: {detail}",
            "Make sure that the configured name identifies an attached physical device.",
        )
    return _ok("configured AVDs", detail)


def _ios_checks(
    expected_device: str | None,
    runner: CommandRunner,
) -> tuple[DoctorCheck, ...]:
    if platform_module.system() != "Darwin":
        return (
            _skip("Xcode", "iOS automation requires macOS"),
            _skip("devices", "iOS automation requires macOS"),
            _skip("applesimutils", "iOS automation requires macOS"),
        )
    xcode = _capture(runner, ["xcode-select", "-p"])
    checks = [
        _ok("Xcode", (xcode.stdout or "").strip())
        if xcode is not None and xcode.returncode == 0 and (xcode.stdout or "").strip()
        else _fail("Xcode", "command-line tools are unavailable", "xcode-select --install")
    ]
    try:
        devices = IosDeviceInventory(
            SimctlDeviceInventory(runner),
            IosPhysicalDeviceInventory(runner),
        ).list_devices()
    except DeviceInventoryError as error:
        checks.append(_fail("devices", str(error)))
    else:
        checks.append(_device_check(devices, expected_device))
    applesimutils = shutil.which("applesimutils")
    checks.append(
        _ok("applesimutils", applesimutils)
        if applesimutils
        else _warn(
            "applesimutils",
            "not installed. Permission-state verification is unavailable",
            "brew tap wix/brew && brew install applesimutils",
        )
    )
    return tuple(checks)


def _device_check(
    devices: Sequence[DeviceDescriptor],
    expected: str | None,
) -> DoctorCheck:
    if not devices:
        return _fail("devices", "none found")
    labels = ", ".join(device.display_name for device in devices)
    if expected is None:
        return _ok("devices", labels)
    matching = tuple(
        device
        for device in devices
        if expected in {device.target_id, device.display_name, device.connection_id}
    )
    if matching:
        device = matching[0]
        return _ok("devices", f"{expected}: {device.readiness.value}")
    return _fail(
        "devices",
        f"configured device '{expected}' was not found. Available: {labels}",
        "Update device in yewo.yaml or create/connect the configured device.",
    )


def _android_sdk(environment: Mapping[str, str]) -> Path | None:
    for name in ANDROID_SDK_ENVIRONMENT:
        raw = environment.get(name)
        if raw and Path(raw).expanduser().is_dir():
            return Path(raw).expanduser().resolve()
    default = Path.home() / "Library" / "Android" / "sdk"
    return default.resolve() if platform_module.system() == "Darwin" and default.is_dir() else None


def _android_binary(
    name: str,
    sdk: Path,
    environment: Mapping[str, str],
) -> Path | None:
    found = shutil.which(name, path=environment.get("PATH", ""))
    if found:
        return Path(found).resolve()
    relatives = (f"platform-tools/{name}", f"emulator/{name}", f"tools/bin/{name}")
    return next(
        (candidate for relative in relatives if (candidate := sdk / relative).is_file()), None
    )


def _binary_check(name: str, path: Path | None) -> DoctorCheck:
    return _ok(name, str(path)) if path is not None else _fail(name, "not found")


def _aapt_check(sdk: Path) -> DoctorCheck:
    candidates = sorted(
        (
            *tuple((sdk / "build-tools").glob("*/aapt")),
            *tuple((sdk / "build-tools").glob("*/aapt2")),
        ),
        reverse=True,
    )
    return (
        _ok("aapt", str(candidates[0]))
        if candidates
        else _fail("aapt", "not found", "Install Android SDK Build-Tools.")
    )


def _selected_platforms(requested: str | None, configured: str | None) -> tuple[str, ...]:
    selected = requested or configured or DEFAULT_PLATFORM
    if selected == "all":
        return "android", "ios"
    return (selected,)


def _expected_device(project: _ProjectInspection, platform: str) -> str | None:
    if project.platform not in {None, platform}:
        return None
    return project.device_name


def _driver(platform: str) -> str:
    return "uiautomator2" if platform == "android" else "xcuitest"


def _capture(
    runner: CommandRunner,
    command: Sequence[str],
    timeout_seconds: float = 20,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return runner(command, timeout_seconds=timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        return None


def _ok(name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status=DoctorStatus.OK, detail=detail)


def _warn(name: str, detail: str, hint: str | None = None) -> DoctorCheck:
    return DoctorCheck(name=name, status=DoctorStatus.WARN, detail=detail, hint=hint)


def _fail(name: str, detail: str, hint: str | None = None) -> DoctorCheck:
    return DoctorCheck(name=name, status=DoctorStatus.FAIL, detail=detail, hint=hint)


def _skip(name: str, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status=DoctorStatus.SKIP, detail=detail)
