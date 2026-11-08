"""Tests for resolving a compatible local Appium runtime."""

from collections.abc import Mapping, Sequence
from collections import deque
from pathlib import Path
import subprocess

import pytest

from yewo.automation import AppiumRuntimeError, resolve_appium_runtime
from yewo.automation.appium_runtime import (
    APPIUM_BINARY_ENV,
    NODE_BINARY_ENV,
    parse_installed_appium_drivers,
)


class _RuntimeRunnerStub:
    def __init__(self, responses: Sequence[tuple[str, str, int]]) -> None:
        self._responses = deque(responses)
        self.calls: list[tuple[tuple[str, ...], float, Mapping[str, str]]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        environment: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((tuple(command), timeout_seconds, environment))
        stdout, stderr, returncode = self._responses.popleft()
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _runtime_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    node = tmp_path / "bin" / "node"
    appium_root = tmp_path / "lib" / "node_modules" / "appium"
    appium_script = appium_root / "build" / "lib" / "main.js"
    package_json = appium_root / "package.json"
    node.parent.mkdir(parents=True)
    appium_script.parent.mkdir(parents=True)
    node.touch()
    appium_script.touch()
    package_json.write_text("{}", encoding="utf-8")
    return node, appium_script, package_json


def test_runtime_resolver_checks_node_compatibility_and_appium_version(
    tmp_path: Path,
) -> None:
    node, appium_script, package_json = _runtime_files(tmp_path)
    runner = _RuntimeRunnerStub((("", "", 0), ("3.1.0\n", "", 0)))

    runtime = resolve_appium_runtime(
        {
            NODE_BINARY_ENV: str(node),
            APPIUM_BINARY_ENV: str(appium_script),
            "PATH": "/usr/bin",
        },
        runner=runner,
    )

    assert runtime.node_executable == node
    assert runtime.appium_script == appium_script
    assert runtime.appium_version == "3.1.0"
    assert runtime.environment["PATH"].startswith(str(node.parent))
    assert runner.calls[0][0][-1] == str(package_json)
    assert runner.calls[1][0] == (str(node), str(appium_script), "--version")


def test_runtime_resolver_rejects_an_incompatible_node(tmp_path: Path) -> None:
    node, appium_script, _ = _runtime_files(tmp_path)
    runner = _RuntimeRunnerStub((("", "Node is too old", 1),))

    with pytest.raises(AppiumRuntimeError, match="Node is too old"):
        resolve_appium_runtime(
            {
                NODE_BINARY_ENV: str(node),
                APPIUM_BINARY_ENV: str(appium_script),
            },
            runner=runner,
        )


def test_runtime_resolver_rejects_a_missing_appium_script(tmp_path: Path) -> None:
    node = tmp_path / "node"
    node.touch()

    with pytest.raises(AppiumRuntimeError, match="does not exist"):
        resolve_appium_runtime(
            {
                NODE_BINARY_ENV: str(node),
                APPIUM_BINARY_ENV: str(tmp_path / "missing-appium"),
            }
        )


def test_runtime_resolver_rejects_an_empty_appium_version(tmp_path: Path) -> None:
    node, appium_script, _ = _runtime_files(tmp_path)
    runner = _RuntimeRunnerStub((("", "", 0), ("\n", "", 0)))

    with pytest.raises(AppiumRuntimeError, match="empty version"):
        resolve_appium_runtime(
            {
                NODE_BINARY_ENV: str(node),
                APPIUM_BINARY_ENV: str(appium_script),
            },
            runner=runner,
        )


def test_driver_listing_parser_requires_exact_installed_driver_names() -> None:
    installed_drivers = parse_installed_appium_drivers(
        '{"uiautomator2-extra":{"installed":true},"uiautomator2":{"installed":false}}'
    )

    assert installed_drivers == frozenset({"uiautomator2-extra"})


@pytest.mark.parametrize("output", ("", "Appium says uiautomator2 is installed", "[]"))
def test_driver_listing_parser_rejects_non_object_json(output: str) -> None:
    assert parse_installed_appium_drivers(output) is None


def test_runtime_resolver_replaces_a_host_shim_with_the_installed_script(
    tmp_path: Path,
) -> None:
    node, appium_script, _ = _runtime_files(tmp_path)
    shim = tmp_path / "bin" / "appium.cmd"
    shim.touch()
    runner = _RuntimeRunnerStub([("", "", 0), ("2.11.3", "", 0)])

    runtime = resolve_appium_runtime(
        {APPIUM_BINARY_ENV: str(shim), NODE_BINARY_ENV: str(node)},
        runner=runner,
    )

    assert runtime.appium_script == appium_script
    assert str(shim) not in runner.calls[-1][0]


def test_runtime_resolver_reports_a_host_shim_without_an_installed_script(
    tmp_path: Path,
) -> None:
    node = tmp_path / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.touch()
    shim = tmp_path / "bin" / "appium.cmd"
    shim.touch()

    with pytest.raises(AppiumRuntimeError, match="no Appium script"):
        resolve_appium_runtime(
            {APPIUM_BINARY_ENV: str(shim), NODE_BINARY_ENV: str(node)},
            runner=_RuntimeRunnerStub([]),
        )
