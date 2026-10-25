"""Read mobile packages and produce deterministic run-level policy evidence."""

import plistlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.platform import Platform
from yewo.android_policy import evaluate_android_manifest, parse_manifest
from yewo.errors import UsageError
from yewo.ios_policy import evaluate_ios_metadata
from yewo.policy import (
    POLICY_CHECK_IDS_BY_PLATFORM,
    PolicyCheckId,
    PolicyCheckReport,
    PolicyCheckStatus,
    bounded_policy_text,
)
from yewo.utils.secrets import SecretScrubber
from yewo.utils.subprocess_runner import SubprocessRunError, run_command

KNOWN_CHECK_IDS = {check_id.value for check_id in PolicyCheckId}
MAX_IOS_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024


def _uniform(
    platform: Platform, status: PolicyCheckStatus, source: str, evidence: str
) -> list[PolicyCheckReport]:
    return [
        PolicyCheckReport(
            check_id=check_id,
            status=status,
            platform=platform,
            source=bounded_policy_text(source, 1024),
            evidence=bounded_policy_text(evidence, 4096),
        )
        for check_id in POLICY_CHECK_IDS_BY_PLATFORM[platform]
    ]


def _ios_info(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Info.plist must contain a dictionary")
    ats = payload.get("NSAppTransportSecurity")
    if ats is not None and not isinstance(ats, dict):
        raise ValueError("NSAppTransportSecurity must be a dictionary")
    for key in (
        "NSAllowsArbitraryLoads",
        "NSAllowsArbitraryLoadsInWebContent",
        "NSAllowsArbitraryLoadsForMedia",
    ):
        if ats and key in ats and not isinstance(ats[key], bool):
            raise ValueError(f"{key} must be a boolean")
    domains = (ats or {}).get("NSExceptionDomains")
    if domains is not None and not isinstance(domains, dict):
        raise ValueError("NSExceptionDomains must be a dictionary")
    for settings in (domains or {}).values():
        if not isinstance(settings, dict):
            raise ValueError("each NSExceptionDomains entry must be a dictionary")
        insecure = settings.get("NSExceptionAllowsInsecureHTTPLoads")
        if insecure is not None and not isinstance(insecure, bool):
            raise ValueError("NSExceptionAllowsInsecureHTTPLoads must be a boolean")
    for key in ("UIFileSharingEnabled", "LSSupportsOpeningDocumentsInPlace"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be a boolean")
    url_types = payload.get("CFBundleURLTypes")
    if url_types is not None and not isinstance(url_types, list):
        raise ValueError("CFBundleURLTypes must be a list")
    for entry in url_types or []:
        if not isinstance(entry, dict):
            raise ValueError("each CFBundleURLTypes entry must be a dictionary")
        schemes = entry.get("CFBundleURLSchemes")
        if schemes is not None and (
            not isinstance(schemes, list)
            or any(not isinstance(scheme, str) for scheme in schemes)
        ):
            raise ValueError("CFBundleURLSchemes must be a list of strings")
    return payload


def _ios_entitlements(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("the entitlement property list is not a dictionary")
    domains = payload.get("com.apple.developer.associated-domains")
    if domains is not None and (
        not isinstance(domains, list)
        or any(not isinstance(domain, str) for domain in domains)
    ):
        raise ValueError("associated-domains must be a list of strings")
    get_task_allow = payload.get("get-task-allow")
    if get_task_allow is not None and not isinstance(get_task_allow, bool):
        raise ValueError("get-task-allow must be a boolean")
    return payload


def _android_xml_references(xmltree: str) -> set[str]:
    manifest = parse_manifest(xmltree)
    application = manifest.child("application")
    if application is None:
        return set()
    return {
        reference
        for name in ("networkSecurityConfig", "fullBackupContent", "dataExtractionRules")
        if (reference := application.attrs.get(name)) and reference.startswith("@0x")
    }


def _resource_path(resource_lines: list[str], resource_id: str) -> str | None:
    matched_index = next(
        (
            index
            for index, line in enumerate(resource_lines)
            if f"resource {resource_id} " in line and "xml/" in line
        ),
        None,
    )
    if matched_index is None:
        return None
    file_line = next(
        (
            line
            for line in resource_lines[matched_index + 1 : matched_index + 5]
            if "(file) res/" in line
        ),
        None,
    )
    if file_line:
        return file_line.split("(file)", 1)[1].split()[0]
    resource_name = resource_lines[matched_index].split("xml/", 1)[1].split()[0]
    return f"res/xml/{resource_name}.xml"


def _android_resource_xml(path: Path, xmltree: str) -> dict[str, str]:
    references = _android_xml_references(xmltree)
    if not references:
        return {}
    try:
        aapt2 = AdbControl.find_build_tool("aapt2")
        resources = run_command(
            [aapt2, "dump", "resources", str(path)],
            check=True,
            timeout=30,
            log_command=False,
        ).stdout
        resource_lines = resources.splitlines()
        xml_by_reference = {}
        for reference in sorted(references):
            resource_path = _resource_path(resource_lines, reference.removeprefix("@"))
            if resource_path is None:
                continue
            try:
                xml_by_reference[reference] = run_command(
                    [aapt2, "dump", "xmltree", "--file", resource_path, str(path)],
                    check=True,
                    timeout=30,
                    log_command=False,
                ).stdout
            except (SubprocessRunError, subprocess.TimeoutExpired):
                continue
        return xml_by_reference
    except (SubprocessRunError, subprocess.TimeoutExpired, ValueError):
        return {}


def _android(path: Path) -> list[PolicyCheckReport]:
    if path.suffix.lower() != ".apk":
        return _uniform(
            Platform.ANDROID,
            PolicyCheckStatus.UNSUPPORTED,
            str(path),
            "Android package policy requires an .apk artifact.",
        )
    xmltree = AdbControl.dump_manifest(str(path))
    if not xmltree:
        return _uniform(
            Platform.ANDROID,
            PolicyCheckStatus.EVIDENCE_MISSING,
            "aapt2 dump xmltree --file AndroidManifest.xml",
            "aapt2 could not read AndroidManifest.xml from the APK.",
        )
    try:
        return evaluate_android_manifest(xmltree, _android_resource_xml(path, xmltree))
    except ValueError as exc:
        return _uniform(
            Platform.ANDROID,
            PolicyCheckStatus.EVIDENCE_MISSING,
            "aapt2 dump xmltree --file AndroidManifest.xml",
            f"AndroidManifest.xml evidence was malformed: {exc}",
        )


def _ipa_bundle(archive: zipfile.ZipFile) -> str | None:
    candidates = sorted(
        {
            "/".join(PurePosixPath(name).parts[:2])
            for name in archive.namelist()
            if len(PurePosixPath(name).parts) >= 3
            and PurePosixPath(name).parts[0] == "Payload"
            and PurePosixPath(name).parts[1].endswith(".app")
        }
    )
    return candidates[0] if len(candidates) == 1 else None


def _safe_extract_bundle(archive: zipfile.ZipFile, prefix: str, destination: Path) -> Path:
    bundle_members = [
        member
        for member in archive.infolist()
        if member.filename.startswith(prefix + "/")
    ]
    if sum(member.file_size for member in bundle_members) > MAX_IOS_BUNDLE_BYTES:
        raise ValueError("the iOS app bundle is larger than the inspection limit")
    for member in bundle_members:
        parts = PurePosixPath(member.filename).parts
        if ".." in parts or not member.filename.startswith(prefix + "/"):
            continue
        archive.extract(member, destination)
    return destination / prefix


def _signed_entitlements(bundle: Path) -> tuple[dict | None, tuple[PolicyCheckStatus, str] | None]:
    codesign = shutil.which("codesign")
    if not codesign:
        return None, (
            PolicyCheckStatus.UNSUPPORTED,
            "Signed entitlement inspection requires the macOS codesign tool.",
        )
    try:
        result = run_command(
            [codesign, "-d", "--entitlements", ":-", str(bundle)],
            check=True,
            timeout=30,
            log_command=False,
        )
    except (SubprocessRunError, subprocess.TimeoutExpired) as exc:
        return None, (
            PolicyCheckStatus.EVIDENCE_MISSING,
            f"codesign could not read the signed entitlements: {exc}",
        )
    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    start = combined.find("<?xml")
    end = combined.find("</plist>", start)
    if start < 0 or end < 0:
        return None, (
            PolicyCheckStatus.EVIDENCE_MISSING,
            "codesign returned no entitlement property list.",
        )
    try:
        payload = plistlib.loads(combined[start : end + len("</plist>")].encode())
        return _ios_entitlements(payload), None
    except (plistlib.InvalidFileException, ValueError) as exc:
        return None, (
            PolicyCheckStatus.EVIDENCE_MISSING,
            f"codesign returned malformed entitlement evidence: {exc}",
        )


def _ios_app(path: Path) -> tuple[dict, Path]:
    info_path = path / "Info.plist"
    with info_path.open("rb") as handle:
        return _ios_info(plistlib.load(handle)), path


def _ios_ipa(path: Path, destination: Path) -> tuple[dict, Path]:
    with zipfile.ZipFile(path) as archive:
        prefix = _ipa_bundle(archive)
        if prefix is None:
            raise ValueError("the IPA must contain exactly one Payload/*.app bundle")
        with archive.open(prefix + "/Info.plist") as handle:
            info = _ios_info(plistlib.load(handle))
        return info, _safe_extract_bundle(archive, prefix, destination)


def _ios(path: Path) -> list[PolicyCheckReport]:
    if path.suffix.lower() not in {".app", ".ipa"}:
        return _uniform(
            Platform.IOS,
            PolicyCheckStatus.UNSUPPORTED,
            str(path),
            "iOS package policy requires an .app bundle or .ipa archive.",
        )
    try:
        with tempfile.TemporaryDirectory(prefix="yewo-policy-") as directory:
            info, bundle = (
                _ios_app(path)
                if path.suffix.lower() == ".app"
                else _ios_ipa(path, Path(directory))
            )
            entitlements, problem = _signed_entitlements(bundle)
            return evaluate_ios_metadata(info, entitlements, problem)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError, plistlib.InvalidFileException) as exc:
        return _uniform(
            Platform.IOS,
            PolicyCheckStatus.EVIDENCE_MISSING,
            str(path),
            f"The iOS package metadata could not be read: {exc}",
        )


def _waive(
    checks: list[PolicyCheckReport], waivers: dict[str, str]
) -> list[PolicyCheckReport]:
    unknown = set(waivers) - KNOWN_CHECK_IDS
    if unknown:
        raise UsageError("Unknown policy waiver check ID(s): " + ", ".join(sorted(unknown)))
    waived = []
    for check in checks:
        reason = waivers.get(check.check_id)
        if reason and check.status is PolicyCheckStatus.FAILED:
            waived.append(
                PolicyCheckReport(
                    **{
                        **check.model_dump(),
                        "status": PolicyCheckStatus.WAIVED,
                        "waiver_reason": reason,
                    }
                )
            )
        else:
            waived.append(check)
    return waived


def evaluate_package_policy(
    path: Path,
    platform: Platform,
    waivers: dict[str, str],
    scrubber: SecretScrubber,
) -> list[PolicyCheckReport]:
    """Inspect one build before Appium starts, then mask its bounded evidence."""
    checks = _waive(_android(path) if platform is Platform.ANDROID else _ios(path), waivers)
    return [
        PolicyCheckReport(
            **{
                **check.model_dump(),
                "source": bounded_policy_text(scrubber.scrub(check.source), 1024),
                "evidence": bounded_policy_text(scrubber.scrub(check.evidence), 4096),
                "waiver_reason": (
                    bounded_policy_text(scrubber.scrub(check.waiver_reason), 2048)
                    if check.waiver_reason
                    else None
                ),
            }
        )
        for check in checks
    ]
