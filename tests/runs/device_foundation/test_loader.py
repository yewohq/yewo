"""Tests for bounded deterministic-script loading."""

import json
from pathlib import Path

import pytest

from yewo.devices import DevicePlatform
from yewo.runs.device_foundation import (
    MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES,
    DeviceFoundationScriptError,
    load_device_foundation_script,
)


def _script_payload() -> dict[str, object]:
    return {
        "script_id": "script_1",
        "platform": "android",
        "actions": [{"action_id": "wait_1", "kind": "wait", "duration_seconds": 0}],
    }


def test_loads_one_valid_json_script(tmp_path: Path) -> None:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(_script_payload()))

    script = load_device_foundation_script(path)

    assert script.script_id == "script_1"
    assert script.platform is DevicePlatform.ANDROID


def test_rejects_content_larger_than_the_bounded_read(tmp_path: Path) -> None:
    path = tmp_path / "script.json"
    path.write_bytes(b"x" * (MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES + 1))

    with pytest.raises(DeviceFoundationScriptError, match="one-megabyte"):
        load_device_foundation_script(path)


def test_rejects_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_script_payload()))
    link = tmp_path / "script.json"
    link.symlink_to(target)

    with pytest.raises(DeviceFoundationScriptError, match="regular file"):
        load_device_foundation_script(link)


def test_invalid_content_is_not_repeated_in_the_error(tmp_path: Path) -> None:
    path = tmp_path / "script.json"
    sensitive_value = "secret-value"
    path.write_text(sensitive_value)

    with pytest.raises(DeviceFoundationScriptError) as captured:
        load_device_foundation_script(path)

    assert sensitive_value not in str(captured.value)
