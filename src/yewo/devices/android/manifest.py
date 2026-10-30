"""Find launcher entries in an AAPT2 manifest tree."""

from __future__ import annotations

from dataclasses import dataclass
import re


MAIN_ACTION = "android.intent.action.MAIN"
LAUNCHER_CATEGORY = "android.intent.category.LAUNCHER"
_ELEMENT = re.compile(r"^(\s*)E: (\S+)")
_NAME_ATTRIBUTE = re.compile(r'A: .*?android:name\([^)]*\)="([^"]*)"')
_ENABLED_ATTRIBUTE = re.compile(r"A: .*?android:enabled\([^)]*\)=(\S+)")


@dataclass(frozen=True)
class LauncherEntry:
    """One activity or alias that an Android launcher can start."""

    name: str
    enabled: bool


def _qualified_name(name: str, package_id: str) -> str:
    return f"{package_id}{name}" if name.startswith(".") else name


def launcher_entries(xml_tree: str, package_id: str) -> tuple[LauncherEntry, ...]:
    """Return activities and aliases that declare MAIN and LAUNCHER."""
    entries: list[LauncherEntry] = []
    name: str | None = None
    enabled = True
    depth = -1
    nested = False
    has_main = False
    has_launcher = False

    def record_entry() -> None:
        if name and has_main and has_launcher:
            entries.append(LauncherEntry(_qualified_name(name, package_id), enabled))

    for line in xml_tree.splitlines():
        element = _ELEMENT.match(line)
        if element:
            indent = len(element.group(1))
            tag = element.group(2)
            if depth >= 0 and indent <= depth:
                record_entry()
                name, enabled, depth = None, True, -1
                nested, has_main, has_launcher = False, False, False
            if tag in ("activity", "activity-alias"):
                name, enabled, depth = None, True, indent
                nested, has_main, has_launcher = False, False, False
            elif depth >= 0:
                nested = True
            continue
        if depth < 0:
            continue
        name_attribute = _NAME_ATTRIBUTE.search(line)
        if name_attribute:
            value = name_attribute.group(1)
            if not nested:
                name = name or value
            elif value == MAIN_ACTION:
                has_main = True
            elif value == LAUNCHER_CATEGORY:
                has_launcher = True
            continue
        enabled_attribute = _ENABLED_ATTRIBUTE.search(line)
        if enabled_attribute and not nested:
            enabled = enabled_attribute.group(1).strip().lower() != "false"
    record_entry()
    return tuple(entries)


def enabled_launcher(xml_tree: str, package_id: str) -> str | None:
    """Return the sole enabled launcher entry when the manifest is clear."""
    enabled = [entry.name for entry in launcher_entries(xml_tree, package_id) if entry.enabled]
    return enabled[0] if len(enabled) == 1 else None
