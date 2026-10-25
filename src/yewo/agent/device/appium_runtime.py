"""Resolve an installed Appium server with a compatible Node runtime."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

NODE_BINARY_ENV = "YEWO_NODE_BINARY"
APPIUM_BINARY_ENV = "YEWO_APPIUM_BINARY"
RUNTIME_CHECK_TIMEOUT_SECONDS = 10

NODE_COMPATIBILITY_SCRIPT = """
const {createRequire} = require('module');
const packagePath = process.argv[1];
const fromAppium = createRequire(packagePath);
const appiumPackage = fromAppium(packagePath);
const semver = fromAppium('semver');
const requirement = appiumPackage.engines && appiumPackage.engines.node;
if (!requirement || !semver.satisfies(process.versions.node, requirement)) {
  process.stderr.write(`Node ${process.versions.node} does not satisfy ${requirement || 'unknown'}`);
  process.exit(1);
}
""".strip()


@dataclass(frozen=True)
class AppiumRuntime:
    """The executables and environment used to start Appium."""

    node_binary: Path
    appium_script: Path
    appium_version: str
    environment: dict[str, str]


def _unique_existing(paths) -> list[Path]:
    unique: list[Path] = []
    known: set[Path] = set()
    for candidate in paths:
        path = Path(candidate).expanduser()
        try:
            identity = path.resolve()
        except OSError:
            continue
        if identity in known or not identity.is_file():
            continue
        known.add(identity)
        unique.append(identity)
    return unique


def _node_candidates(environment: dict[str, str]) -> list[Path]:
    configured = environment.get(NODE_BINARY_ENV)
    if configured:
        return _unique_existing([configured])

    discovered: list[str | Path] = []
    active = shutil.which("node", path=environment.get("PATH"))
    if active:
        discovered.append(active)
    for root in (Path("/opt/homebrew/opt"), Path("/usr/local/opt")):
        if root.is_dir():
            discovered.extend(sorted(root.glob("node@*/bin/node"), reverse=True))
    nvm_root = Path(environment.get("NVM_DIR", str(Path.home() / ".nvm")))
    versions = nvm_root / "versions" / "node"
    if versions.is_dir():
        discovered.extend(sorted(versions.glob("*/bin/node"), reverse=True))
    return _unique_existing(discovered)


def _appium_package_json(appium_script: Path) -> Path | None:
    for directory in (appium_script.parent, *appium_script.parents):
        package = directory / "package.json"
        if package.is_file():
            return package
    return None


def resolve_appium_runtime(environment: dict[str, str] | None = None) -> AppiumRuntime:
    """Find an installed Node that can execute the installed Appium script."""
    process_environment = dict(os.environ if environment is None else environment)
    configured_appium = process_environment.get(APPIUM_BINARY_ENV)
    appium = configured_appium or shutil.which(
        "appium", path=process_environment.get("PATH")
    )
    if not appium or not Path(appium).expanduser().is_file():
        raise RuntimeError(
            "Appium is not installed or is not on PATH. Install Appium or set "
            f"{APPIUM_BINARY_ENV}."
        )
    appium_script = Path(appium).expanduser().resolve()
    appium_package = _appium_package_json(appium_script)
    candidates = _node_candidates(process_environment)
    if not candidates:
        raise RuntimeError(
            "Node is not installed or no executable was found. Install a Node "
            f"version supported by Appium or set {NODE_BINARY_ENV}."
        )

    failures: list[str] = []
    for node_binary in candidates:
        runtime_environment = dict(process_environment)
        existing_path = runtime_environment.get("PATH", "")
        runtime_environment["PATH"] = os.pathsep.join(
            filter(None, (str(node_binary.parent), existing_path))
        )
        try:
            if appium_package is not None:
                compatible = subprocess.run(
                    [
                        str(node_binary),
                        "-e",
                        NODE_COMPATIBILITY_SCRIPT,
                        str(appium_package),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=RUNTIME_CHECK_TIMEOUT_SECONDS,
                    env=runtime_environment,
                    check=False,
                )
                if compatible.returncode != 0:
                    reason = (compatible.stderr or compatible.stdout).strip()
                    failures.append(f"{node_binary}: {reason or 'incompatible'}")
                    continue
            checked = subprocess.run(
                [str(node_binary), str(appium_script), "--version"],
                capture_output=True,
                text=True,
                timeout=RUNTIME_CHECK_TIMEOUT_SECONDS,
                env=runtime_environment,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{node_binary}: {type(exc).__name__}")
            continue
        if checked.returncode == 0:
            return AppiumRuntime(
                node_binary=node_binary,
                appium_script=appium_script,
                appium_version=checked.stdout.strip() or "unknown",
                environment=runtime_environment,
            )
        reason = (checked.stderr or checked.stdout).strip().splitlines()
        failures.append(f"{node_binary}: {reason[-1] if reason else 'incompatible'}")

    attempted = "; ".join(failures)
    raise RuntimeError(
        "No installed Node runtime can start Appium. Set "
        f"{NODE_BINARY_ENV} to a compatible Node executable. Tried: {attempted}"
    )
