"""Deterministic Android package-policy checks from an aapt2 manifest dump."""

import re
from dataclasses import dataclass, field

from yewo.agent.device.platform import Platform
from yewo.policy import (
    PolicyCheckId,
    PolicyCheckReport,
    PolicyCheckStatus,
    bounded_policy_text,
)

SOURCE = "aapt2 dump xmltree --file AndroidManifest.xml"
ELEMENT = re.compile(r"^(\s*)E: ([^\s(]+)")
ATTRIBUTE = re.compile(r"^(\s*)A: ([^\s(=]+)(?:\([^)]*\))?=(.*)$")
RAW = re.compile(r'\(Raw: "([^"]*)"\)')
QUOTED = re.compile(r'^"([^"]*)"')
HEX = re.compile(r"0x([0-9a-fA-F]+)$")
PLATFORM_PERMISSION_LEVEL_BY_NAME = {
    # Flutter's profile installer exports a receiver behind this framework permission.
    "android.permission.DUMP": "signature|privileged|development",
}


@dataclass
class ManifestNode:
    """One element in aapt2's indentation-based XML tree."""

    tag: str
    indent: int
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["ManifestNode"] = field(default_factory=list)

    def child(self, tag: str) -> "ManifestNode | None":
        return next((item for item in self.children if item.tag == tag), None)


def _value(raw: str) -> str:
    raw_value = RAW.search(raw)
    if raw_value:
        return raw_value.group(1)
    quoted = QUOTED.search(raw.strip())
    if quoted:
        return quoted.group(1)
    return raw.strip()


def parse_xmltree(xmltree: str, root_tag: str) -> ManifestNode:
    """Parse the stable element/attribute subset emitted by aapt2."""
    roots: list[ManifestNode] = []
    stack: list[ManifestNode] = []
    for line in xmltree.splitlines():
        element = ELEMENT.match(line)
        if element:
            indent, tag = len(element.group(1)), element.group(2)
            while stack and stack[-1].indent >= indent:
                stack.pop()
            node = ManifestNode(tag, indent)
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
            continue
        attribute = ATTRIBUTE.match(line)
        if attribute and stack:
            name = attribute.group(2).rsplit(":", 1)[-1]
            stack[-1].attrs[name] = _value(attribute.group(3))
    root = next((node for node in roots if node.tag == root_tag), None)
    if root is None:
        raise ValueError(f"aapt2 output did not contain a {root_tag} element")
    return root


def parse_manifest(xmltree: str) -> ManifestNode:
    """Parse an aapt2 AndroidManifest.xml dump."""
    return parse_xmltree(xmltree, "manifest")


def _boolean(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"true", "0xffffffff"}:
        return True
    if lowered in {"false", "0x0"}:
        return False
    matched = HEX.search(lowered)
    return bool(int(matched.group(1), 16)) if matched else default


def _integer(value: str | None, default: int) -> int:
    if not value:
        return default
    matched = HEX.search(value.strip())
    if matched:
        return int(matched.group(1), 16)
    try:
        return int(value)
    except ValueError:
        return default


def _report(
    check_id: PolicyCheckId,
    status: PolicyCheckStatus,
    evidence: str,
    source: str = SOURCE,
) -> PolicyCheckReport:
    return PolicyCheckReport(
        check_id=check_id.value,
        status=status,
        platform=Platform.ANDROID,
        source=bounded_policy_text(source, 1024),
        evidence=bounded_policy_text(evidence, 4096),
    )


def _debuggable(application: ManifestNode) -> PolicyCheckReport:
    enabled = _boolean(application.attrs.get("debuggable"), False)
    return _report(
        PolicyCheckId.ANDROID_DEBUGGABLE,
        PolicyCheckStatus.FAILED if enabled else PolicyCheckStatus.PASSED,
        f"application android:debuggable is {'true' if enabled else 'false (including default)' }.",
    )


def _rule_entry(node: ManifestNode) -> str:
    domain = node.attrs.get("domain", "unknown-domain")
    path = node.attrs.get("path", ".")
    flags = node.attrs.get("requireFlags")
    return f"{node.tag} {domain}/{path}" + (f" [{flags}]" if flags else "")


def _backup_section(name: str, section: ManifestNode | None) -> str:
    if section is None:
        return f"{name}: section absent, so default exposure remains"
    includes = [child for child in section.children if child.tag == "include"]
    excludes = [child for child in section.children if child.tag == "exclude"]
    scope = (
        f"limited to {len(includes)} include rule(s)"
        if includes
        else "default exposure remains"
    )
    entries = ", ".join(_rule_entry(child) for child in (includes + excludes)[:12])
    encryption = section.attrs.get("disableIfNoEncryptionCapabilities")
    details = f"{name}: {scope}; {len(excludes)} exclusion(s)"
    if encryption is not None:
        details += f"; encryption required={encryption}"
    return details + (f" ({entries})" if entries else "")


def _backup_resource_summary(reference: str, xmltree: str) -> str:
    for root_tag in ("full-backup-content", "data-extraction-rules"):
        try:
            root = parse_xmltree(xmltree, root_tag)
        except ValueError:
            continue
        if root_tag == "full-backup-content":
            return _backup_section(f"{reference} legacy backup", root)
        summaries = [
            _backup_section(name, root.child(name))
            for name in ("cloud-backup", "device-transfer")
        ]
        summaries.extend(
            _backup_section(
                f"cross-platform-transfer[{section.attrs.get('platform', 'unknown')}]",
                section,
            )
            for section in root.children
            if section.tag == "cross-platform-transfer"
        )
        return "; ".join(summaries)
    return f"{reference}: the backup-rule root is not recognized"


def _backup(
    application: ManifestNode, referenced_xml: dict[str, str]
) -> PolicyCheckReport:
    allowed = _boolean(application.attrs.get("allowBackup"), True)
    summaries = []
    for name in ("fullBackupContent", "dataExtractionRules"):
        reference = application.attrs.get(name)
        if not reference:
            continue
        if reference == "false":
            summaries.append(f"{name}=false")
        elif reference in referenced_xml:
            summaries.append(_backup_resource_summary(reference, referenced_xml[reference]))
        else:
            summaries.append(f"{name}={reference}, but the resource could not be read")
    evidence = (
        f"application android:allowBackup is {'true (including default)' if allowed else 'false'}."
    )
    if summaries:
        evidence += " " + "; ".join(summaries) + "."
    return _report(
        PolicyCheckId.ANDROID_BACKUP,
        PolicyCheckStatus.FAILED if allowed else PolicyCheckStatus.PASSED,
        evidence,
    )


def _network_cleartext(
    xmltree: str, default: bool
) -> tuple[PolicyCheckStatus, str]:
    root = parse_xmltree(xmltree, "network-security-config")
    base = root.child("base-config")
    base_allowed = _boolean(
        base.attrs.get("cleartextTrafficPermitted") if base else None, default
    )
    unsafe = ["base-config"] if base_allowed else []

    def visit(node: ManifestNode, inherited: bool) -> None:
        allowed = _boolean(node.attrs.get("cleartextTrafficPermitted"), inherited)
        if node.tag == "domain-config" and allowed:
            declaration = node.attrs.get("cleartextTrafficPermitted", "inherited")
            unsafe.append(f"domain-config ({declaration})")
        for child in node.children:
            if child.tag == "domain-config":
                visit(child, allowed)

    for child in root.children:
        if child.tag == "domain-config":
            visit(child, base_allowed)
    status = PolicyCheckStatus.FAILED if unsafe else PolicyCheckStatus.PASSED
    evidence = (
        "Network security configuration permits cleartext: " + ", ".join(unsafe[:20])
        if unsafe
        else "Network security configuration does not permit cleartext traffic."
    )
    return status, evidence


def _cleartext(
    manifest: ManifestNode,
    application: ManifestNode,
    referenced_xml: dict[str, str],
) -> PolicyCheckReport:
    uses_sdk = manifest.child("uses-sdk")
    target = _integer(uses_sdk.attrs.get("targetSdkVersion") if uses_sdk else None, 1)
    default = target <= 27
    network_config = application.attrs.get("networkSecurityConfig")
    if network_config:
        xmltree = referenced_xml.get(network_config)
        if xmltree:
            try:
                status, evidence = _network_cleartext(xmltree, default)
            except ValueError as exc:
                status = PolicyCheckStatus.EVIDENCE_MISSING
                evidence = f"Network security resource {network_config} was malformed: {exc}"
            return _report(
                PolicyCheckId.ANDROID_CLEARTEXT,
                status,
                evidence,
                f"{SOURCE}; network security resource {network_config}",
            )
        return _report(
            PolicyCheckId.ANDROID_CLEARTEXT,
            PolicyCheckStatus.EVIDENCE_MISSING,
            f"android:networkSecurityConfig {network_config} requires resource inspection.",
        )
    allowed = _boolean(application.attrs.get("usesCleartextTraffic"), default)
    return _report(
        PolicyCheckId.ANDROID_CLEARTEXT,
        PolicyCheckStatus.FAILED if allowed else PolicyCheckStatus.PASSED,
        (
            f"application android:usesCleartextTraffic resolves to {'true' if allowed else 'false'} "
            f"for targetSdkVersion {target}."
        ),
    )


def _has_launcher_filter(component: ManifestNode) -> bool:
    for intent_filter in (child for child in component.children if child.tag == "intent-filter"):
        names = {
            child.attrs.get("name")
            for child in intent_filter.children
            if child.tag in {"action", "category"}
        }
        if {
            "android.intent.action.MAIN",
            "android.intent.category.LAUNCHER",
        }.issubset(names):
            return True
    return False


def _is_exported(component: ManifestNode, target: int) -> bool:
    explicit = component.attrs.get("exported")
    if explicit is not None:
        return _boolean(explicit, False)
    if component.tag == "provider":
        return target <= 16
    return any(child.tag == "intent-filter" for child in component.children)


def _declared_permissions(manifest: ManifestNode) -> dict[str, str]:
    return {
        name: permission.attrs.get("protectionLevel", "normal")
        for permission in manifest.children
        if permission.tag == "permission" and (name := permission.attrs.get("name"))
    }


def _permission_strength(
    name: str, declared_levels: dict[str, str]
) -> tuple[bool | None, str]:
    level = PLATFORM_PERMISSION_LEVEL_BY_NAME.get(name)
    if level is None and name.startswith("android.permission."):
        return None, "platform permission protection level is not packaged"
    if level is None:
        level = declared_levels.get(name)
    if level is None:
        return None, "permission definition is not packaged"
    normalized = level.lower()
    if any(term in normalized for term in ("signature", "knownsigner", "internal")):
        return True, level
    base = _integer(level, 0) & 0xF
    if base in {2, 3, 4}:
        return True, level
    return False, level


def _component_permissions(
    component: ManifestNode, inherited_permission: str | None
) -> tuple[list[str], list[str]]:
    general = component.attrs.get("permission") or inherited_permission
    if general:
        return [general], []
    if component.tag != "provider":
        return [], ["all access"]
    read_permission = component.attrs.get("readPermission")
    write_permission = component.attrs.get("writePermission")
    permissions = [item for item in (read_permission, write_permission) if item]
    missing = []
    if not read_permission:
        missing.append("read access")
    if not write_permission:
        missing.append("write access")
    return permissions, missing


def _external_entries(manifest: ManifestNode, application: ManifestNode) -> PolicyCheckReport:
    uses_sdk = manifest.child("uses-sdk")
    target = _integer(uses_sdk.attrs.get("targetSdkVersion") if uses_sdk else None, 1)
    inherited_permission = application.attrs.get("permission")
    declared_levels = _declared_permissions(manifest)
    findings = []
    unresolved = []
    protected = []
    component_tags = {"activity", "activity-alias", "service", "receiver", "provider"}
    for component in (item for item in application.children if item.tag in component_tags):
        if not _is_exported(component, target) or _has_launcher_filter(component):
            continue
        component_name = f"{component.tag} {component.attrs.get('name', '(unnamed)')}"
        permissions, missing_access = _component_permissions(component, inherited_permission)
        findings.extend(f"{component_name}: no permission for {access}" for access in missing_access)
        for permission in permissions:
            is_strong, level = _permission_strength(permission, declared_levels)
            description = f"{component_name}: {permission} ({level})"
            if is_strong is True:
                protected.append(description)
            elif is_strong is False:
                findings.append(description)
            else:
                unresolved.append(description)
    if findings:
        status = PolicyCheckStatus.FAILED
        evidence = "Unprotected or weak exported components: " + ", ".join(findings[:20])
    elif unresolved:
        status = PolicyCheckStatus.EVIDENCE_MISSING
        evidence = "Exported component permission strength is unknown: " + ", ".join(
            unresolved[:20]
        )
    else:
        status = PolicyCheckStatus.PASSED
        evidence = (
            "Protected exported components: " + ", ".join(protected[:20])
            if protected
            else "No non-launcher exported component is exposed."
        )
    return _report(
        PolicyCheckId.ANDROID_EXPORTED_COMPONENTS,
        status,
        evidence,
    )


def evaluate_android_manifest(
    xmltree: str, referenced_xml: dict[str, str] | None = None
) -> list[PolicyCheckReport]:
    """Evaluate the Android rule families without installing the APK."""
    manifest = parse_manifest(xmltree)
    application = manifest.child("application")
    if application is None:
        raise ValueError("AndroidManifest.xml did not contain an application element")
    return [
        _debuggable(application),
        _backup(application, referenced_xml or {}),
        _cleartext(manifest, application, referenced_xml or {}),
        _external_entries(manifest, application),
    ]
