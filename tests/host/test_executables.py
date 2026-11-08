"""Tests for executable and SDK discovery on each supported host."""

from __future__ import annotations

from pathlib import Path

import pytest

from host.windows_simulation import load_host_module_as
from yewo.host import (
    is_node_script_shim,
    resolve_android_sdk_root,
    resolve_android_sdk_tool,
    resolve_build_tools_executables,
    resolve_node_package_entry_point,
    resolve_sdk_executable,
)


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_sdk_executable_is_found_without_a_suffix_on_posix(tmp_path: Path) -> None:
    _make_executable(tmp_path / "platform-tools" / "adb")

    assert resolve_sdk_executable(tmp_path / "platform-tools", "adb") is not None
    assert resolve_sdk_executable(tmp_path / "platform-tools", "fastboot") is None


def test_build_tools_executables_are_found_in_every_release(tmp_path: Path) -> None:
    _make_executable(tmp_path / "build-tools" / "34.0.0" / "aapt")
    _make_executable(tmp_path / "build-tools" / "35.0.0" / "aapt")
    (tmp_path / "build-tools" / "notes.txt").write_text("ignored", encoding="utf-8")

    aapt_paths = resolve_build_tools_executables(tmp_path, "aapt")

    assert {path.parent.name for path in aapt_paths} == {"34.0.0", "35.0.0"}
    assert resolve_build_tools_executables(tmp_path, "aapt2") == ()


def test_build_tools_discovery_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert resolve_build_tools_executables(tmp_path / "absent", "aapt") == ()


def test_sdk_root_prefers_the_configured_environment(tmp_path: Path) -> None:
    configured = tmp_path / "configured-sdk"
    configured.mkdir()

    assert resolve_android_sdk_root({"ANDROID_HOME": str(configured)}) == configured.resolve()
    assert resolve_android_sdk_root({"ANDROID_SDK_ROOT": str(configured)}) == configured.resolve()
    assert resolve_android_sdk_root({"ANDROID_HOME": str(tmp_path / "absent")}) != configured


def test_sdk_tool_falls_back_to_the_sdk_layout(tmp_path: Path) -> None:
    _make_executable(tmp_path / "platform-tools" / "adb")
    _make_executable(tmp_path / "tools" / "bin" / "sdkmanager")

    assert resolve_android_sdk_tool("adb", tmp_path, "") is not None
    assert resolve_android_sdk_tool("sdkmanager", tmp_path, "") is not None
    assert resolve_android_sdk_tool("emulator", tmp_path, "") is None


def test_sdk_tool_prefers_the_search_path(tmp_path: Path) -> None:
    on_path = tmp_path / "bin"
    _make_executable(on_path / "adb")
    _make_executable(tmp_path / "platform-tools" / "adb")

    adb_path = resolve_android_sdk_tool("adb", tmp_path, str(on_path))

    assert adb_path == (on_path / "adb").resolve()


def test_node_script_shims_are_recognized_by_suffix() -> None:
    assert is_node_script_shim(Path("appium.cmd"))
    assert is_node_script_shim(Path("appium.BAT"))
    assert not is_node_script_shim(Path("appium"))
    assert not is_node_script_shim(Path("index.js"))


def test_node_package_entry_point_is_read_from_the_installed_package(tmp_path: Path) -> None:
    package_root = tmp_path / "node_modules" / "appium"
    package_root.mkdir(parents=True)
    (package_root / "index.js").write_text("", encoding="utf-8")
    shim = tmp_path / "appium.cmd"
    shim.write_text("", encoding="utf-8")

    assert resolve_node_package_entry_point(shim, "appium") == package_root / "index.js"


def test_node_package_entry_point_is_absent_without_an_installed_package(tmp_path: Path) -> None:
    shim = tmp_path / "appium.cmd"
    shim.write_text("", encoding="utf-8")

    assert resolve_node_package_entry_point(shim, "appium") is None


@pytest.mark.parametrize(
    ("platform_name", "expected_suffixes"),
    [
        ("win32", (".exe", ".bat", ".cmd")),
        ("darwin", ("",)),
        ("linux", ("",)),
    ],
)
def test_executable_suffixes_match_the_host(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected_suffixes: tuple[str, ...],
) -> None:
    module = load_host_module_as(monkeypatch, "executables", platform_name)

    assert module.executable_suffixes() == expected_suffixes


def test_windows_sdk_default_follows_local_application_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\becca\AppData\Local")
    module = load_host_module_as(monkeypatch, "executables", "win32")

    assert module.default_android_sdk_root() == Path(
        r"C:\Users\becca\AppData\Local"
    ) / "Android" / "Sdk"


def test_windows_sdk_default_is_absent_without_local_application_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    module = load_host_module_as(monkeypatch, "executables", "win32")

    assert module.default_android_sdk_root() is None


def test_linux_sdk_default_is_the_home_directory_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_host_module_as(monkeypatch, "executables", "linux")

    assert module.default_android_sdk_root() == Path.home() / "Android" / "Sdk"


def test_macos_sdk_default_is_the_library_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_host_module_as(monkeypatch, "executables", "darwin")

    assert module.default_android_sdk_root() == Path.home() / "Library" / "Android" / "sdk"


def test_linux_node_roots_include_linuxbrew(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_host_module_as(monkeypatch, "executables", "linux")

    assert Path("/home/linuxbrew/.linuxbrew/opt") in module.node_installation_roots()


def test_macos_node_roots_include_homebrew(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_host_module_as(monkeypatch, "executables", "darwin")

    assert Path("/opt/homebrew/opt") in module.node_installation_roots()


def test_windows_node_roots_follow_the_host_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROGRAMFILES", r"C:\Program Files")
    monkeypatch.setenv("APPDATA", r"C:\Users\becca\AppData\Roaming")
    module = load_host_module_as(monkeypatch, "executables", "win32")

    roots = module.node_installation_roots()

    assert Path(r"C:\Program Files") / "nodejs" in roots
    assert Path(r"C:\Users\becca\AppData\Roaming") / "nvm" in roots
    assert "v*/node.exe" in module.node_executable_patterns()


def test_posix_host_command_starts_the_executable_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_host_module_as(monkeypatch, "executables", "linux")

    assert module.build_host_command("/usr/bin/npm", "install", "--global", "appium") == (
        "/usr/bin/npm",
        "install",
        "--global",
        "appium",
    )


def test_windows_host_command_starts_a_shim_through_the_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_host_module_as(monkeypatch, "executables", "win32")

    assert module.build_host_command(r"C:\npm\npm.cmd", "install") == (
        "cmd",
        "/c",
        r"C:\npm\npm.cmd",
        "install",
    )


def test_windows_host_command_starts_a_real_executable_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_host_module_as(monkeypatch, "executables", "win32")

    assert module.build_host_command(r"C:\nodejs\node.exe", "--version") == (
        r"C:\nodejs\node.exe",
        "--version",
    )
