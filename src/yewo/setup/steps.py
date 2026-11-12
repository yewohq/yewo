"""Explicit, previewable mutations behind ``yewo setup``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from yewo.automation.appium_runtime import (
    AppiumRuntime,
    parse_installed_appium_drivers,
    resolve_appium_runtime,
)
from yewo.automation.errors import AppiumRuntimeError
from yewo.host import build_host_command, resolve_host_executable, supports_apple_device_tooling
from yewo.project_config import (
    DEFAULT_PLATFORM,
    YewoProjectError,
    find_yewo_project,
    load_yewo_project,
)
from yewo.setup.records import SetupReport, SetupStatus, SetupStep


PROJECT_TEMPLATE = """\
device:
  name: {device}
  platform: {platform}

app: {app}

model: openrouter/deepseek/deepseek-v4.1-flash

grounder:
  model: openrouter/bytedance/ui-tars-1.5-7b
  protocol: source_pixels
"""

ENV_TEMPLATE = """\
# Credentials for models configured in yewo.yaml.
# OPENROUTER_API_KEY=

# Add other provider credentials when you select those providers.
# ANTHROPIC_API_KEY=
# GEMINI_API_KEY=
"""

_PLATFORM_TEMPLATE = {
    "android": ("Pixel_8", "build/app/outputs/flutter-apk/app-debug.apk"),
    "ios": ("iPhone 17 Pro", "build/ios/iphonesimulator/Runner.app"),
}
_DRIVERS = {"android": "uiautomator2", "ios": "xcuitest"}

Confirm = Callable[[str], bool]
_INSTALL_TIMEOUT_SECONDS = 900
_LIST_TIMEOUT_SECONDS = 30


class SetupRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]: ...


def _run(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    capture_output: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=capture_output,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


@dataclass(frozen=True)
class SetupContext:
    root: Path
    requested_platform: str | None
    dry_run: bool
    assume_yes: bool
    confirm: Confirm
    runner: SetupRunner = _run
    environment: Mapping[str, str] | None = None

    def approved(self, prompt: str) -> bool:
        return self.assume_yes or self.confirm(prompt)


def run_setup(context: SetupContext) -> SetupReport:
    """Create missing project files and prepare the requested automation tools."""
    environment = dict(os.environ if context.environment is None else context.environment)
    project_path = find_yewo_project(context.root)
    platforms, project_root, project_step = _project_plan(context, project_path)
    steps = [
        project_step,
        _env_step(context, project_root),
        _gitignore_step(context, project_root),
    ]
    appium_step, runtime = _appium_step(context, environment)
    steps.append(appium_step)
    for platform in platforms:
        steps.append(_driver_step(context, platform, runtime))
    if "ios" in platforms:
        steps.append(_applesimutils_step(context))
    return SetupReport(platforms=platforms, steps=tuple(steps))


def _project_plan(
    context: SetupContext,
    existing_config_path: Path | None,
) -> tuple[tuple[str, ...], Path, SetupStep]:
    if existing_config_path is not None:
        try:
            project = load_yewo_project(existing_config_path)
        except YewoProjectError as error:
            platforms = _platforms(context.requested_platform, None)
            return platforms, existing_config_path.parent, _fail("yewo.yaml", str(error))
        platforms = _platforms(context.requested_platform, project.settings.platform)
        return (
            platforms,
            existing_config_path.parent,
            _ok("yewo.yaml", f"using {existing_config_path}"),
        )

    platforms = _platforms(context.requested_platform, None)
    target = context.root / "yewo.yaml"
    if len(platforms) != 1:
        return (
            platforms,
            context.root,
            _warn(
                "yewo.yaml",
                "not created because a starter project needs one platform",
                "Run setup with --platform android or --platform ios.",
            ),
        )
    platform = platforms[0]
    device, app = _PLATFORM_TEMPLATE[platform]
    content = PROJECT_TEMPLATE.format(platform=platform, device=device, app=app)
    if context.dry_run:
        return platforms, context.root, _planned("yewo.yaml", f"would create {target}")
    if not context.approved(f"Create a starter project at {target}?"):
        return platforms, context.root, _warn("yewo.yaml", "creation declined")
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as error:
        return platforms, context.root, _fail("yewo.yaml", f"could not create {target}: {error}")
    return platforms, context.root, _ok("yewo.yaml", f"created {target}")


def _env_step(context: SetupContext, project_root: Path) -> SetupStep:
    target = project_root / ".env"
    if target.exists():
        return _ok(".env", f"using {target}")
    if context.dry_run:
        return _planned(".env", f"would create {target}")
    if not context.approved(f"Create a credential template at {target}?"):
        return _warn(".env", "creation declined")
    try:
        target.write_text(ENV_TEMPLATE, encoding="utf-8")
    except OSError as error:
        return _fail(".env", f"could not create {target}: {error}")
    return _warn(
        ".env",
        f"created {target}",
        "Add the credentials required by yewo.yaml.",
    )


def _gitignore_step(context: SetupContext, project_root: Path) -> SetupStep:
    target = project_root / ".gitignore"
    try:
        content = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError as error:
        return _fail(".gitignore", f"could not read {target}: {error}")
    entries = {line.strip() for line in content.splitlines()}
    if entries.intersection({".env", "/.env"}):
        return _ok(".gitignore", ".env is ignored")
    if context.dry_run:
        return _planned(".gitignore", f"would add .env to {target}")
    if not context.approved(f"Add .env to {target}?"):
        return _warn(".gitignore", ".env ignore entry declined")
    prefix = "" if not content or content.endswith("\n") else "\n"
    try:
        target.write_text(f"{content}{prefix}.env\n", encoding="utf-8")
    except OSError as error:
        return _fail(".gitignore", f"could not update {target}: {error}")
    return _ok(".gitignore", f"added .env to {target}")


def _appium_step(
    context: SetupContext,
    environment: Mapping[str, str],
) -> tuple[SetupStep, AppiumRuntime | None]:
    runtime = _resolved_runtime(environment)
    if runtime is not None:
        return (
            _ok("Appium", f"{runtime.appium_version} with {runtime.node_executable}"),
            runtime,
        )
    npm_path = resolve_host_executable("npm", environment.get("PATH"))
    if npm_path is None:
        return (
            _fail("Appium", "npm is unavailable", "Install a supported Node.js release."),
            None,
        )
    command = build_host_command(npm_path, "install", "--global", "appium")
    if context.dry_run:
        return _planned("Appium", "would install Appium", command), None
    if not context.approved("Install Appium globally with npm?"):
        return _warn("Appium", "installation declined", command=command), None
    try:
        result = context.runner(
            command,
            timeout_seconds=_INSTALL_TIMEOUT_SECONDS,
            capture_output=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("Appium", "npm installation timed out", command=command), None
    if result.returncode:
        return _fail(
            "Appium",
            _command_failure_detail(result, "npm installation failed"),
            "Review the npm output above, then retry setup.",
            command,
        ), None
    runtime = _resolved_runtime(environment)
    if runtime is None:
        return (
            _fail(
                "Appium",
                "installed, but no compatible Node and Appium pair was found",
                "Run yewo doctor for the compatibility details.",
            ),
            None,
        )
    return _ok("Appium", f"installed {runtime.appium_version}"), runtime


def _resolved_runtime(environment: Mapping[str, str]) -> AppiumRuntime | None:
    try:
        return resolve_appium_runtime(environment)
    except AppiumRuntimeError:
        return None


def _driver_step(
    context: SetupContext,
    platform: str,
    runtime: AppiumRuntime | None,
) -> SetupStep:
    driver = _DRIVERS[platform]
    label = f"driver: {driver}"
    if runtime is None:
        command = ("appium", "driver", "install", driver)
        return (
            _planned(label, "would install after Appium", command)
            if context.dry_run
            else _skip(label, "Appium is unavailable")
        )
    base = (str(runtime.node_executable), str(runtime.appium_script), "driver")
    listing_command = (*base, "list", "--installed", "--json")
    try:
        listing = context.runner(
            listing_command,
            timeout_seconds=_LIST_TIMEOUT_SECONDS,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return _fail(label, "driver listing timed out", command=listing_command)
    installed_drivers = parse_installed_appium_drivers(listing.stdout or "")
    if listing.returncode:
        return _fail(label, "could not list installed drivers", command=listing_command)
    if installed_drivers is None:
        return _fail(label, "Appium returned an invalid driver listing", command=listing_command)
    if driver in installed_drivers:
        return _ok(label, "already installed")
    command = (*base, "install", driver)
    if context.dry_run:
        return _planned(label, "would install", command)
    if not context.approved(f"Install the Appium {driver} driver?"):
        return _warn(label, "installation declined", command=command)
    try:
        result = context.runner(
            command,
            timeout_seconds=_INSTALL_TIMEOUT_SECONDS,
            capture_output=False,
        )
    except subprocess.TimeoutExpired:
        return _fail(label, "installation timed out", command=command)
    if result.returncode:
        return _fail(
            label,
            _command_failure_detail(result, "installation failed"),
            "Review the Appium output above, then retry setup.",
            command,
        )
    return _ok(label, "installed")


def _applesimutils_step(context: SetupContext) -> SetupStep:
    if not supports_apple_device_tooling():
        return _skip("applesimutils", "iOS permission tooling requires macOS")
    applesimutils_path = shutil.which("applesimutils")
    if applesimutils_path:
        return _ok("applesimutils", applesimutils_path)
    brew = shutil.which("brew")
    if brew is None:
        return _warn(
            "applesimutils",
            "Homebrew is unavailable",
            "Install applesimutils manually to verify iOS permissions.",
        )
    command = (brew, "install", "wix/brew/applesimutils")
    if context.dry_run:
        return _planned("applesimutils", "would install", command)
    if not context.approved("Install applesimutils with Homebrew?"):
        return _warn("applesimutils", "installation declined", command=command)
    try:
        result = context.runner(
            command,
            timeout_seconds=_INSTALL_TIMEOUT_SECONDS,
            capture_output=False,
        )
    except subprocess.TimeoutExpired:
        return _warn("applesimutils", "installation timed out", command=command)
    if result.returncode:
        return _warn(
            "applesimutils",
            "installation failed",
            "Install applesimutils manually to verify iOS permissions.",
            command,
        )
    return _ok("applesimutils", "installed")


def _platforms(requested_platform: str | None, configured_platform: str | None) -> tuple[str, ...]:
    selected_platform = requested_platform or configured_platform or DEFAULT_PLATFORM
    return ("android", "ios") if selected_platform == "all" else (selected_platform,)


def _command_failure_detail(
    result: subprocess.CompletedProcess[str],
    fallback: str,
) -> str:
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        return fallback
    maximum_length = 500
    return output if len(output) <= maximum_length else f"{output[: maximum_length - 3]}..."


def _ok(name: str, detail: str) -> SetupStep:
    return SetupStep(name=name, status=SetupStatus.OK, detail=detail)


def _planned(name: str, detail: str, command: tuple[str, ...] = ()) -> SetupStep:
    return SetupStep(name=name, status=SetupStatus.PLANNED, detail=detail, command=command)


def _warn(
    name: str,
    detail: str,
    hint: str | None = None,
    command: tuple[str, ...] = (),
) -> SetupStep:
    return SetupStep(name=name, status=SetupStatus.WARN, detail=detail, hint=hint, command=command)


def _fail(
    name: str,
    detail: str,
    hint: str | None = None,
    command: tuple[str, ...] = (),
) -> SetupStep:
    return SetupStep(name=name, status=SetupStatus.FAIL, detail=detail, hint=hint, command=command)


def _skip(name: str, detail: str) -> SetupStep:
    return SetupStep(name=name, status=SetupStatus.SKIP, detail=detail)
