"""Tests for deterministic iOS diagnostic classification."""

import json

from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.devices.ios.diagnostics import (
    ios_report_contains_application,
    parse_ios_diagnostic_report,
)


def _report(details: dict[str, object]) -> str:
    header = {
        "bundleID": "com.example.app",
        "timestamp": "2026-10-30T12:00:00Z",
    }
    return json.dumps(header) + "\n" + json.dumps(details)


def test_watchdog_termination_is_an_application_anr() -> None:
    report = parse_ios_diagnostic_report(
        "Example.ips",
        _report(
            {
                "exceptionType": "EXC_CRASH (SIGKILL)",
                "terminationReason": "Namespace SPRINGBOARD, Code 0x8badf00d",
            }
        ),
    )

    assert report.failure_kind is ApplicationFailureKind.ANR
    assert report.occurred_at is not None


def test_fatal_exception_is_an_application_crash() -> None:
    report = parse_ios_diagnostic_report(
        "Example.ips",
        _report({"exceptionType": "EXC_BAD_ACCESS (SIGSEGV)"}),
    )

    assert report.failure_kind is ApplicationFailureKind.CRASH


def test_real_apple_timestamp_and_qualified_termination_fields_are_preserved() -> None:
    content = (
        json.dumps(
            {
                "bundleID": "com.example.app",
                "timestamp": "2026-10-30 12:00:00.00 +0000",
                "code": "unrelated-header-code",
            }
        )
        + "\n"
        + json.dumps(
            {
                "exception": {"type": "EXC_BAD_ACCESS"},
                "termination": {
                    "code": 11,
                    "indicator": "Segmentation fault: 11",
                },
            }
        )
    )

    report = parse_ios_diagnostic_report("Example.ips", content)

    assert report.occurred_at is not None
    assert report.occurred_at.isoformat() == "2026-10-30T12:00:00+00:00"
    assert report.platform_code == "11"
    assert report.platform_reason == "EXC_BAD_ACCESS"


def test_unrelated_bare_code_is_not_reported_as_a_termination_code() -> None:
    content = json.dumps(
        {
            "bundleID": "com.example.app",
            "timestamp": "2026-10-30T12:00:00Z",
            "code": "unrelated-header-code",
            "exceptionType": "EXC_BAD_ACCESS",
        }
    )

    report = parse_ios_diagnostic_report("Example.ips", content)

    assert report.platform_code is None


def test_nonfatal_report_is_not_promoted_to_a_crash() -> None:
    report = parse_ios_diagnostic_report(
        "Example.ips",
        _report(
            {
                "exceptionType": "EXC_CRASH",
                "exceptionNote": "NON-FATAL CONDITION (this is NOT a crash)",
            }
        ),
    )

    assert report.failure_kind is ApplicationFailureKind.UNCLASSIFIED


def test_application_identity_search_is_structural() -> None:
    assert ios_report_contains_application(_report({}), "com.example.app")
    assert not ios_report_contains_application(_report({}), "com.other.app")
