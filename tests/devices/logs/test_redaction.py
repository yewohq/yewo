"""Tests for declared-value device-log redaction."""

from yewo.devices.logs import DeclaredValueDeviceLogScrubber


def test_declared_values_are_replaced_longest_first() -> None:
    scrubber = DeclaredValueDeviceLogScrubber(("token", "token-extended"))

    result = scrubber("token-extended token token-extended")

    assert result.text == "[yewo-redacted] [yewo-redacted] [yewo-redacted]"
    assert result.redaction_count == 3
    assert result.policy_id == "yewo-declared-values-v1"


def test_replacement_text_is_not_processed_as_another_sensitive_value() -> None:
    scrubber = DeclaredValueDeviceLogScrubber(("secret", "[yewo-redacted]"))

    result = scrubber("secret")

    assert result.text == "[yewo-redacted]"
    assert result.redaction_count == 1
