"""Deterministic iOS package-policy checks from Info.plist and entitlements."""

from typing import Any

from yewo.agent.device.platform import Platform
from yewo.policy import (
    PolicyCheckId,
    PolicyCheckReport,
    PolicyCheckStatus,
    bounded_policy_text,
)

INFO_SOURCE = "Info.plist"
ENTITLEMENT_SOURCE = "codesign -d --entitlements :-"


def _report(
    check_id: PolicyCheckId,
    status: PolicyCheckStatus,
    evidence: str,
    source: str = INFO_SOURCE,
) -> PolicyCheckReport:
    return PolicyCheckReport(
        check_id=check_id.value,
        status=status,
        platform=Platform.IOS,
        source=bounded_policy_text(source, 1024),
        evidence=bounded_policy_text(evidence, 4096),
    )


def _development(
    entitlements: dict[str, Any] | None, problem: tuple[PolicyCheckStatus, str] | None
) -> PolicyCheckReport:
    if entitlements is None:
        status, evidence = problem or (
            PolicyCheckStatus.EVIDENCE_MISSING,
            "The signed entitlements could not be read.",
        )
        return _report(PolicyCheckId.IOS_GET_TASK_ALLOW, status, evidence, ENTITLEMENT_SOURCE)
    enabled = bool(entitlements.get("get-task-allow", False))
    return _report(
        PolicyCheckId.IOS_GET_TASK_ALLOW,
        PolicyCheckStatus.FAILED if enabled else PolicyCheckStatus.PASSED,
        f"Signed entitlement get-task-allow is {'true' if enabled else 'false or absent'}.",
        ENTITLEMENT_SOURCE,
    )


def _file_sharing(info: dict[str, Any]) -> PolicyCheckReport:
    enabled = [
        key
        for key in ("UIFileSharingEnabled", "LSSupportsOpeningDocumentsInPlace")
        if bool(info.get(key, False))
    ]
    return _report(
        PolicyCheckId.IOS_FILE_SHARING,
        PolicyCheckStatus.FAILED if enabled else PolicyCheckStatus.PASSED,
        (
            "Enabled document exposure keys: " + ", ".join(enabled)
            if enabled
            else "File sharing and in-place document opening are disabled or absent."
        ),
    )


def _transport_security(info: dict[str, Any]) -> PolicyCheckReport:
    ats = info.get("NSAppTransportSecurity") or {}
    unsafe = [
        key
        for key in (
            "NSAllowsArbitraryLoads",
            "NSAllowsArbitraryLoadsInWebContent",
            "NSAllowsArbitraryLoadsForMedia",
        )
        if bool(ats.get(key, False))
    ]
    domains = ats.get("NSExceptionDomains") or {}
    unsafe.extend(
        f"NSExceptionDomains.{domain}.NSExceptionAllowsInsecureHTTPLoads"
        for domain, settings in domains.items()
        if isinstance(settings, dict)
        and bool(settings.get("NSExceptionAllowsInsecureHTTPLoads", False))
    )
    return _report(
        PolicyCheckId.IOS_APP_TRANSPORT_SECURITY,
        PolicyCheckStatus.FAILED if unsafe else PolicyCheckStatus.PASSED,
        (
            "Insecure App Transport Security exceptions: " + ", ".join(unsafe[:20])
            if unsafe
            else "No broad or insecure App Transport Security exception is enabled."
        ),
    )


def _external_entries(
    info: dict[str, Any],
    entitlements: dict[str, Any] | None,
    entitlement_problem: tuple[PolicyCheckStatus, str] | None,
) -> PolicyCheckReport:
    schemes = []
    for entry in info.get("CFBundleURLTypes") or []:
        if isinstance(entry, dict):
            schemes.extend(str(item) for item in entry.get("CFBundleURLSchemes") or [])
    domains = list((entitlements or {}).get("com.apple.developer.associated-domains") or [])
    if schemes:
        evidence = "Custom URL schemes require explicit review: " + ", ".join(schemes[:20])
        status = PolicyCheckStatus.FAILED
    elif entitlements is None:
        status, problem = entitlement_problem or (
            PolicyCheckStatus.EVIDENCE_MISSING,
            "The signed entitlements could not be read.",
        )
        evidence = f"No custom URL schemes are declared. {problem}"
    else:
        evidence = "No custom URL schemes are declared."
        status = PolicyCheckStatus.PASSED
    if domains:
        evidence += " Associated domains: " + ", ".join(str(item) for item in domains[:20])
    return _report(PolicyCheckId.IOS_EXTERNAL_ENTRY_POINTS, status, evidence)


def evaluate_ios_metadata(
    info: dict[str, Any],
    entitlements: dict[str, Any] | None,
    entitlement_problem: tuple[PolicyCheckStatus, str] | None = None,
) -> list[PolicyCheckReport]:
    """Evaluate the iOS rule families without launching the app."""
    return [
        _development(entitlements, entitlement_problem),
        _file_sharing(info),
        _transport_security(info),
        _external_entries(info, entitlements, entitlement_problem),
    ]
